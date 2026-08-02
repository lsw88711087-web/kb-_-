"""외부 벤치마크 대조 + 애블레이션.

(A) 분쟁조정 사례 정답셋 대조 — 적합성 판정 적중률
    arm 3종: single_norag(단발·근거없음) / single(단발·RAG) / debate(3진영 디베이트·RAG)
    평가 시 해당 사례 문서는 검색에서 제외한다(정답 누출 방지).

(B) KOSIS 보유율 대조 — 세그먼트별 가입의향률의 순위 일치도(Spearman) 및 MAE
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..agents.debate import DebateConfig, run_debate, run_ensemble, single_shot
from ..agents.schema import Suitability
from ..config import BENCHMARK_DIR, OUTPUT_DIR
from ..llm import LLMClient, LLMError
from ..personas.loader import PersonaSource, load_personas, require_nemotron_personas
from ..personas.schema import Persona
from ..products.schema import Product
from ..rag.retriever import get_retriever
from .confidence import aggregate
from .simulate import Mode, SimulationReport, simulate_product

CASES_PATH = BENCHMARK_DIR / "dispute_cases.json"
HOLDING_PATH = BENCHMARK_DIR / "segment_holding_rates.json"

Arm = Literal[
    "naive", "single_norag", "single_lawonly", "single", "debate", "debate_lawonly", "ensemble"
]
# naive:          규칙·RAG 없음 (순응 편향의 실재 여부 확인용 최하위 비교군)
# single_lawonly: 법조문·가이드라인만 검색. 조정례를 뺀다.
#                 조정례 문서는 제목에 `— 판정 {label}`이 붙어 있어 이웃 사례의 정답이
#                 프롬프트에 노출된다(실측: 검색 결과의 72%가 조정례, 그중 74%가 fail/warn.
#                 깨끗한 대출 케이스는 검색된 조정례 3건이 전부 fail).
#                 single과의 차이가 곧 '조정례 라벨 노출'의 기여분이다.
# debate_lawonly: 같은 처리를 디베이트에 적용. **single_lawonly만으로는 디베이트를 설명할 수 없다.**
#                 실측(sev_B, 깨끗한 2건): single 오탐 3개는 debate 오탐 6개의 부분집합이고
#                 debate가 3개를 더 만든다. 두 arm은 동일한 근거 문서를 보므로, 그 3개는
#                 근거가 원인일 수 없다(같은 문서를 본 single이 제기하지 않았다).
#                 회의론자 역할이 만든 몫이라 single_lawonly의 사정권 밖이다.
# ensemble:       단발 + 디베이트 병행. 두 방식이 서로 다른 우려를 놓친다는 실측에 근거한다
#                 (합집합이 정답 우려 29/30, 단독은 24). 라벨은 더 정확·안정한 단발에서 취한다.
ARMS: tuple[Arm, ...] = ("naive", "single_norag", "single", "debate", "ensemble")

# 조정례를 뺀 검색 대상. corpus.Doc.kind와 맞춰야 한다.
LAW_ONLY_KINDS: tuple[str, ...] = ("law", "guideline")

RISKY = {"warn", "fail"}


class BenchCase(BaseModel):
    case_id: str
    title: str
    label: Suitability
    principle: list[str] = Field(default_factory=list)
    product: Product
    persona: Persona
    facts: str = Field(
        default="",
        description=(
            "판매 정황(설명 방식·권유 경위). 정답 라벨은 '판매 행위가 적법했는가'를 "
            "판정한 것이므로 이 정보 없이는 모델이 '상품이 위험한가'만 보고 답하게 된다. "
            "decision_gist(판단요지)는 정답에 해당하므로 절대 전달하지 않는다."
        ),
    )


def load_cases() -> list[BenchCase]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    out = []
    for c in payload["cases"]:
        out.append(
            BenchCase(
                case_id=c["case_id"],
                title=c["title"],
                label=c["label"],
                principle=c.get("principle", []),
                product=Product(**c["product"]),
                persona=Persona(**c["persona"]),
                facts=c.get("facts", ""),
            )
        )
    return out


class CaseOutcome(BaseModel):
    case_id: str
    title: str
    gold: Suitability
    pred: Suitability
    exact: bool
    risk_correct: bool = Field(description="위험(warn/fail) 대 정상(pass) 이분류 일치")
    principle_recall: float = Field(
        description="정답 위반원칙 중 예측이 잡아낸 비율 (시드별 재현율의 평균). "
        "gold=pass 케이스에서는 무의미하므로 arm 집계에서 제외된다"
    )
    principle_recall_union: float = Field(
        default=0.0, description="시드 합집합 기준 재현율 (참고용 상한)"
    )
    principle_applicable: bool = Field(
        default=True,
        description="위반원칙 채점 대상인가. gold=pass는 위반이 없어 대상이 아니다",
    )
    false_principle_claims: float = Field(
        default=0.0,
        description="gold=pass인데 위반원칙을 주장한 평균 개수. 과잉 경고의 직접 측정",
    )
    judge_schema_repairs: int = Field(
        default=0, description="심판 응답 스키마 교정이 필요했던 횟수"
    )
    confidence: float
    intent_mean: float
    n_llm_calls: int
    elapsed_sec: float
    # 우려 단위 채점을 위해 산출 텍스트를 보존한다 (라벨만으로는 우려 발굴력을 못 잰다)
    predicted_principles: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""
    dropped_concerns: list[str] = Field(
        default_factory=list, description="사실팩 계산값과 모순되어 기각된 우려 (환각 발생량 지표)"
    )
    concerns: list[dict] = Field(
        default_factory=list, description="구조화된 우려 (유형·심각도·앵커). 유형 일치로 채점한다"
    )


class ArmScore(BaseModel):
    arm: Arm
    n: int
    accuracy: float
    risk_accuracy: float
    macro_f1: float
    principle_recall: float = Field(
        description="실제 위반이 있는 케이스(gold≠pass)만의 평균. pass는 위반이 없어 제외한다"
    )
    principle_recall_union: float = 0.0
    false_principle_rate: float = Field(
        default=0.0,
        description="gold=pass 케이스에서 주장한 위반원칙 평균 개수. 낮을수록 좋다",
    )
    judge_schema_repairs: int = 0
    evidence_empty_rate: float = Field(
        default=0.0, description="근거를 하나도 제시하지 않은 케이스 비율 (품질 지표)"
    )
    failed_cases: list[str] = Field(
        default_factory=list, description="LLM 응답 실패로 채점에서 제외된 케이스"
    )
    mean_confidence: float
    total_llm_calls: int
    total_elapsed_sec: float
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    outcomes: list[CaseOutcome] = Field(default_factory=list)


class AblationReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    generated_at: str
    backend: str = ""
    model_small: str = ""
    model_judge: str = ""
    n_seeds: int = 1
    screen_situation: bool = Field(
        default=False,
        description="판매 정황 모순 스크리닝(FDM_SCREEN_SITUATION)을 켜고 돌렸는가. "
        "결과 파일만 보고 조건을 알 수 있어야 비교가 성립한다",
    )
    seed_base: int = Field(
        default=7000,
        description="시드 시작값. 이 값을 바꿔 재실행하면 표집 노이즈를 측정할 수 있다",
    )
    arms: list[ArmScore] = Field(default_factory=list)
    delta_accuracy_vs_single: float = 0.0
    delta_risk_accuracy_vs_single: float = 0.0
    caveats: list[str] = Field(default_factory=list)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else OUTPUT_DIR / "ablation.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "AblationReport":
        return AblationReport(**json.loads(Path(path).read_text(encoding="utf-8")))


def _macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {"pass", "warn", "fail"}
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
            continue
        if tp == 0:
            continue  # 정답·예측 모두 없는 클래스는 제외
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(statistics.fmean(f1s), 3) if f1s else 0.0


def _principle_recall(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 1.0
    hit = sum(1 for g in gold if any(g in p or p in g for p in pred))
    return hit / len(gold)


def _mean_applicable(outcomes: list, field: str) -> float:
    """위반원칙 채점 대상 케이스만 평균한다. 대상이 없으면 0.0."""
    vals = [getattr(o, field) for o in outcomes if o.principle_applicable]
    return statistics.fmean(vals) if vals else 0.0


def has_violation(label: str) -> bool:
    """정답 라벨에 실제 위반이 있는가.

    정답셋의 `principle` 필드는 라벨에 따라 의미가 다르다.
      - fail/warn: 실제로 **위반한** 원칙
      - pass     : 쟁점이 되었을 뿐 **위반하지 않은** 원칙
                   (CASE-006 "판매원칙 위반이 없다", CASE-008 "적합한 권유로 판단")
    둘을 같이 채점하면 깨끗한 케이스에서 위반을 주장해야 점수를 받는다.
    실측: CASE-008에서 모델이 '적합성 위반' 주장을 뺐는데 재현율이 100%→0%로
    떨어져, 옳은 자제가 성능 하락으로 기록됐다(ensemble 83.3%→75.0%의 전부).
    """
    return label != "pass"


def _run_ensemble(
    case: BenchCase,
    *,
    seed: int,
    cfg: DebateConfig,
    client: LLMClient,
    retriever,
    exclude: set[str],
):
    """벤치마크용 얇은 래퍼. 실제 구현은 `agents.debate.run_ensemble`에 있다.

    시뮬레이션 경로(리포트·UI)와 **같은 함수를 써야** 두 경로가 갈라지지 않는다.
    이전에는 이 로직이 벤치마크에만 있어서 리포트에는 ensemble이 아예 없었다.
    """
    return run_ensemble(
        case.product, case.persona,
        segment=case.case_id, seed=seed, config=cfg, client=client,
        retriever=retriever, exclude_doc_ids=exclude, situation=case.facts,
    )


def run_case_arm(
    case: BenchCase,
    arm: Arm,
    *,
    n_seeds: int = 1,
    config: DebateConfig | None = None,
    client: LLMClient | None = None,
    seed_base: int = 7000,
) -> CaseOutcome:
    cfg = config or DebateConfig()
    client = client or LLMClient()
    # law-only arm은 검색 대상만 다르다. 원본 cfg를 건드리면 다른 arm까지 오염되므로 복제한다.
    if arm in {"single_lawonly", "debate_lawonly"}:
        cfg = replace(cfg, retrieve_kinds=LAW_ONLY_KINDS)
    retriever = get_retriever(cfg.use_dense)
    exclude = {case.case_id}  # 정답 누출 방지

    runs = []
    for i in range(n_seeds):
        seed = seed_base + 137 * i
        if arm == "ensemble":
            runs.append(
                _run_ensemble(
                    case, seed=seed, cfg=cfg, client=client, retriever=retriever, exclude=exclude
                )
            )
        elif arm in {"debate", "debate_lawonly"}:
            runs.append(
                run_debate(
                    case.product, case.persona, segment=case.case_id, seed=seed,
                    config=cfg, client=client, retriever=retriever, exclude_doc_ids=exclude,
                    situation=case.facts,
                )
            )
        else:
            runs.append(
                single_shot(
                    case.product, case.persona, segment=case.case_id, seed=seed,
                    config=cfg, client=client, retriever=retriever, exclude_doc_ids=exclude,
                    with_rag=(arm in {"single", "single_lawonly"}),
                    use_rules=(arm != "naive"),
                    situation=case.facts,
                )
            )
    cr = aggregate(runs)
    pred = cr.modal_suitability
    n_calls = sum(len(r.turns) for r in runs)
    return CaseOutcome(
        case_id=case.case_id,
        title=case.title,
        gold=case.label,
        pred=pred,
        exact=pred == case.label,
        risk_correct=(pred in RISKY) == (case.label in RISKY),
        # 시드별로 각각 재현율을 구한 뒤 평균한다. 여러 시드의 위반원칙을 합집합으로
        # 모으면 "시드를 늘릴수록 재현율이 올라가는" 지표가 되어 arm 간 비교가 왜곡된다.
        principle_recall=round(
            statistics.fmean(
                _principle_recall(case.principle, r.verdict.violated_principles) for r in runs
            ),
            3,
        ),
        # 참고용: 합집합 기준(=시드 중 한 번이라도 잡았는가). 상한을 보여준다.
        principle_recall_union=round(
            _principle_recall(
                case.principle, [x for r in runs for x in r.verdict.violated_principles]
            ),
            3,
        ),
        principle_applicable=has_violation(case.label),
        # pass 케이스에서 위반을 주장한 개수. 재현율로 보상하지 않고 별도로 센다.
        false_principle_claims=(
            0.0
            if has_violation(case.label)
            else round(statistics.fmean(len(r.verdict.violated_principles) for r in runs), 2)
        ),
        judge_schema_repairs=sum(1 for r in runs if r.judge_schema_repaired),
        confidence=cr.confidence,
        intent_mean=cr.intent_mean,
        n_llm_calls=n_calls,
        elapsed_sec=round(sum(r.elapsed_sec for r in runs), 2),
        predicted_principles=sorted({x for r in runs for x in r.verdict.violated_principles}),
        risks=[x for r in runs for x in r.verdict.risks],
        evidence=[x for r in runs for x in r.verdict.evidence],
        recommendations=[x for r in runs for x in r.verdict.recommendations],
        summary=" ".join(r.verdict.summary for r in runs),
        dropped_concerns=[x for r in runs for x in r.dropped_concerns],
        # 구조화 우려: 채점을 키워드 추측이 아니라 유형 일치로 하기 위해 보존한다
        concerns=[c.model_dump() for r in runs for c in r.verdict.concerns],
    )


def run_ablation(
    *,
    arms: tuple[Arm, ...] = ARMS,
    n_seeds: int = 1,
    seed_base: int = 7000,
    limit: int | None = None,
    config: DebateConfig | None = None,
    progress: bool = True,
) -> AblationReport:
    from ..config import SETTINGS

    cases = load_cases()[: limit or None]
    client = LLMClient()
    scores: list[ArmScore] = []

    for arm in arms:
        outcomes = []
        failed: list[str] = []
        for case in cases:
            # 한 케이스의 LLM 실패가 전체 실행을 날리지 않게 격리한다.
            # 실패 케이스는 값을 조작하지 않고 채점에서 제외하고 별도 보고한다.
            try:
                oc = run_case_arm(
                    case, arm, n_seeds=n_seeds, config=config, client=client,
                    seed_base=seed_base,
                )
            except LLMError as e:
                failed.append(case.case_id)
                if progress:
                    print(f"  [{arm}] {case.case_id} 실패 → 채점 제외: {str(e)[:90]}", flush=True)
                continue
            outcomes.append(oc)
            if progress:
                mark = "O" if oc.exact else "X"
                print(
                    f"  [{arm}] {case.case_id} gold={oc.gold} pred={oc.pred} {mark} "
                    f"(신뢰도 {oc.confidence})",
                    flush=True,
                )
        if not outcomes:
            raise LLMError(f"arm={arm}: 모든 케이스가 실패했다. 백엔드 상태를 확인하라.")
        pairs = [(o.gold, o.pred) for o in outcomes]
        confusion: dict[str, dict[str, int]] = {}
        for g, p in pairs:
            confusion.setdefault(g, Counter())[p] = confusion.setdefault(g, Counter()).get(p, 0) + 1
        scores.append(
            ArmScore(
                arm=arm,
                n=len(outcomes),
                accuracy=round(sum(o.exact for o in outcomes) / len(outcomes), 3),
                risk_accuracy=round(sum(o.risk_correct for o in outcomes) / len(outcomes), 3),
                macro_f1=_macro_f1(pairs),
                # 위반이 실제로 있는 케이스만 평균한다. pass 케이스를 넣으면
                # 깨끗한 케이스에서 위반을 주장할수록 점수가 오른다(결함 15번).
                principle_recall=round(_mean_applicable(outcomes, "principle_recall"), 3),
                principle_recall_union=round(
                    _mean_applicable(outcomes, "principle_recall_union"), 3
                ),
                false_principle_rate=round(
                    statistics.fmean(
                        [o.false_principle_claims for o in outcomes if not o.principle_applicable]
                        or [0.0]
                    ),
                    3,
                ),
                judge_schema_repairs=sum(o.judge_schema_repairs for o in outcomes),
                evidence_empty_rate=round(sum(1 for o in outcomes if not o.evidence) / len(outcomes), 3),
                failed_cases=failed,
                mean_confidence=round(statistics.fmean(o.confidence for o in outcomes), 3),
                total_llm_calls=sum(o.n_llm_calls for o in outcomes),
                total_elapsed_sec=round(sum(o.elapsed_sec for o in outcomes), 1),
                confusion={k: dict(v) for k, v in confusion.items()},
                outcomes=outcomes,
            )
        )

    by_arm = {s.arm: s for s in scores}
    d_acc = d_risk = 0.0
    if "debate" in by_arm and "single" in by_arm:
        d_acc = round(by_arm["debate"].accuracy - by_arm["single"].accuracy, 3)
        d_risk = round(by_arm["debate"].risk_accuracy - by_arm["single"].risk_accuracy, 3)

    return AblationReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        backend=SETTINGS.backend,
        model_small=SETTINGS.model_small,
        model_judge=SETTINGS.model_judge,
        n_seeds=n_seeds,
        screen_situation=(config or DebateConfig()).screen_situation,
        seed_base=seed_base,
        arms=scores,
        delta_accuracy_vs_single=d_acc,
        delta_risk_accuracy_vs_single=d_risk,
        caveats=[
            f"정답셋 {len(cases)}건은 조정결정례 유형을 참조해 재구성한 가공 샘플이다. 실제 원문으로 교체 시 수치는 달라진다.",
            "표본이 작아 신뢰구간이 넓다. 결론은 '방향성'으로만 해석해야 한다.",
            "평가 시 해당 사례 문서를 검색 대상에서 제외해 정답 누출을 막았다.",
        ],
    )


# ------------------------------------------------------------- (B) KOSIS 보유율 대조
class HoldingComparison(BaseModel):
    segment: str
    category: str
    actual_holding_rate: float
    simulated_adoption_rate: float
    abs_error: float


class HoldingReport(BaseModel):
    product_id: str
    product_name: str
    category: str
    mode: str
    n_segments: int
    mae: float
    spearman: float | None
    rows: list[HoldingComparison] = Field(default_factory=list)
    note: str = (
        "'보유율'과 '가입의향률'은 정의가 달라 절대값 차이(MAE)보다 세그먼트 간 순위 일치도(Spearman)가 1차 지표다."
    )


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return round(num / den, 3) if den else None


def compare_holding_rates(report: SimulationReport, category: str) -> HoldingReport:
    payload = json.loads(HOLDING_PATH.read_text(encoding="utf-8"))
    table = {
        (r["segment"], r["category"]): r["rate"] for r in payload["holding_rates"]
    }
    rows = []
    for sr in report.segments:
        actual = table.get((sr.segment, category))
        if actual is None:
            continue
        rows.append(
            HoldingComparison(
                segment=sr.segment,
                category=category,
                actual_holding_rate=actual,
                simulated_adoption_rate=sr.adoption_rate,
                abs_error=round(abs(actual - sr.adoption_rate), 3),
            )
        )
    mae = round(statistics.fmean(r.abs_error for r in rows), 3) if rows else 0.0
    rho = _spearman(
        [r.actual_holding_rate for r in rows], [r.simulated_adoption_rate for r in rows]
    )
    return HoldingReport(
        product_id=report.product_id,
        product_name=report.product_name,
        category=category,
        mode=report.mode,
        n_segments=len(rows),
        mae=mae,
        spearman=rho,
        rows=rows,
    )


class ProductBenchmarkRow(BaseModel):
    product_id: str
    product_name: str
    category: str
    n_segments: int
    persona_pool_size: int
    persona_nemotron_count: int
    persona_synthetic_count: int
    single_adoption_rate: float | None = None
    debate_adoption_rate: float | None = None
    delta_adoption_rate_vs_single: float | None = None
    single_mean_intent: float | None = None
    debate_mean_intent: float | None = None
    delta_intent_vs_single: float | None = None
    single_fail_ratio: float | None = None
    debate_fail_ratio: float | None = None
    single_low_confidence_ratio: float | None = None
    debate_low_confidence_ratio: float | None = None
    single_holding_mae: float | None = None
    debate_holding_mae: float | None = None
    delta_mae_vs_single: float | None = None
    single_holding_spearman: float | None = None
    debate_holding_spearman: float | None = None
    delta_spearman_vs_single: float | None = None
    sim_paths: dict[str, str] = Field(default_factory=dict)


class ProductBenchmarkReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    generated_at: str
    backend: str = ""
    model_small: str = ""
    model_judge: str = ""
    n_products: int
    n_seeds: int
    personas_per_segment: int
    modes: list[Mode]
    rows: list[ProductBenchmarkRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else OUTPUT_DIR / "product_benchmark.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "ProductBenchmarkReport":
        return ProductBenchmarkReport(**json.loads(Path(path).read_text(encoding="utf-8")))


def _weighted_segment_mean(report: SimulationReport | None, attr: str) -> float | None:
    if report is None:
        return None
    total = sum(s.n_personas for s in report.segments)
    if total == 0:
        return None
    return round(sum(getattr(s, attr) * s.n_personas for s in report.segments) / total, 3)


def _weighted_fail_ratio(report: SimulationReport | None) -> float | None:
    if report is None:
        return None
    total = sum(s.n_personas for s in report.segments)
    if total == 0:
        return None
    fail = sum(s.verdict_mix.get("fail", 0.0) * s.n_personas for s in report.segments)
    return round(fail / total, 3)


def _delta(a: float | None, b: float | None) -> float | None:
    return round(a - b, 3) if a is not None and b is not None else None


def run_product_benchmark(
    products: list[Product],
    *,
    modes: tuple[Mode, ...] = ("single", "debate"),
    n_seeds: int = 2,
    k_personas: int = 3,
    persona_limit: int = 2000,
    workers: int = 4,
    segment_names: list[str] | None = None,
    persona_source: PersonaSource = "auto",
    require_real_personas: bool = False,
    config: DebateConfig | None = None,
    progress: bool = True,
    save_simulations: bool = True,
) -> ProductBenchmarkReport:
    from ..config import SETTINGS

    personas = load_personas(
        source=persona_source,
        limit=persona_limit,
        allow_synthetic_fallback=not require_real_personas,
    )
    if require_real_personas:
        require_nemotron_personas(personas)

    rows: list[ProductBenchmarkRow] = []
    for product in products:
        if progress:
            print(f"\n[상품 벤치마크] {product.name} ({product.product_id})", flush=True)
        sims: dict[str, SimulationReport] = {}
        holds: dict[str, HoldingReport] = {}
        sim_paths: dict[str, str] = {}

        for mode in modes:
            if progress:
                print(f"  - mode={mode}", flush=True)
            sim = simulate_product(
                product,
                segment_names=segment_names or product.target_segments or None,
                k_personas=k_personas,
                n_seeds=n_seeds,
                mode=mode,
                workers=workers,
                config=config,
                personas=personas,
                require_real_personas=require_real_personas,
                progress=progress,
            )
            sims[mode] = sim
            holds[mode] = compare_holding_rates(sim, product.category)
            if save_simulations:
                sim_paths[mode] = str(sim.save())

        single = sims.get("single")
        debate = sims.get("debate")
        single_hold = holds.get("single")
        debate_hold = holds.get("debate")
        template = debate or single or next(iter(sims.values()), None)
        rows.append(
            ProductBenchmarkRow(
                product_id=product.product_id,
                product_name=product.name,
                category=product.category,
                n_segments=max((len(s.segments) for s in sims.values()), default=0),
                persona_pool_size=template.persona_pool_size if template else len(personas),
                persona_nemotron_count=template.persona_nemotron_count if template else 0,
                persona_synthetic_count=template.persona_synthetic_count if template else 0,
                single_adoption_rate=_weighted_segment_mean(single, "adoption_rate"),
                debate_adoption_rate=_weighted_segment_mean(debate, "adoption_rate"),
                single_mean_intent=_weighted_segment_mean(single, "mean_intent"),
                debate_mean_intent=_weighted_segment_mean(debate, "mean_intent"),
                single_fail_ratio=_weighted_fail_ratio(single),
                debate_fail_ratio=_weighted_fail_ratio(debate),
                single_low_confidence_ratio=_weighted_segment_mean(single, "low_confidence_ratio"),
                debate_low_confidence_ratio=_weighted_segment_mean(debate, "low_confidence_ratio"),
                single_holding_mae=single_hold.mae if single_hold else None,
                debate_holding_mae=debate_hold.mae if debate_hold else None,
                single_holding_spearman=single_hold.spearman if single_hold else None,
                debate_holding_spearman=debate_hold.spearman if debate_hold else None,
                sim_paths=sim_paths,
            )
        )

    for row in rows:
        row.delta_adoption_rate_vs_single = _delta(row.debate_adoption_rate, row.single_adoption_rate)
        row.delta_intent_vs_single = _delta(row.debate_mean_intent, row.single_mean_intent)
        row.delta_mae_vs_single = _delta(row.debate_holding_mae, row.single_holding_mae)
        row.delta_spearman_vs_single = _delta(
            row.debate_holding_spearman, row.single_holding_spearman
        )

    return ProductBenchmarkReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        backend=SETTINGS.backend,
        model_small=SETTINGS.model_small,
        model_judge=SETTINGS.model_judge,
        n_products=len(products),
        n_seeds=n_seeds,
        personas_per_segment=k_personas,
        modes=list(modes),
        rows=rows,
        notes=[
            "single은 디베이트 없는 단발 RAG 판정 대조군, debate는 3진영 디베이트+RAG 실험군이다.",
            "KOSIS 보유율은 가입의향과 정의가 다르므로 절대값보다 Spearman 순위상관을 우선 확인한다.",
            "벤치마크 데이터는 PoC용 근사/가공 데이터이므로 발표 전 원문·최신 통계로 교체해야 한다.",
        ],
    )


def build_product_benchmark_markdown(report: ProductBenchmarkReport) -> str:
    lines = ["# 여러 상품 벤치마크 리포트"]
    lines.append(
        f"\n- 생성시각: {report.generated_at}"
        f"\n- 실행환경: backend=`{report.backend}`, 토론모델=`{report.model_small}`, 심판모델=`{report.model_judge}`"
        f"\n- 상품 수: {report.n_products} / 모드: {', '.join(report.modes)}"
        f"\n- 세그먼트별 페르소나: {report.personas_per_segment}명 / 멀티시드: {report.n_seeds}회"
    )
    lines.append("\n## 1. 상품별 대조군 비교")
    lines.append(
        "\n| 상품 | 상품군 | 세그먼트 | single 가입률 | debate 가입률 | Δ가입률 | "
        "single 의향 | debate 의향 | Δ의향 | debate fail | debate 저신뢰 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.rows:
        lines.append(
            f"| {row.product_name} | {row.category} | {row.n_segments} | "
            f"{_pct(row.single_adoption_rate)} | {_pct(row.debate_adoption_rate)} | "
            f"{_signed_pct(row.delta_adoption_rate_vs_single)} | "
            f"{_num(row.single_mean_intent)} | {_num(row.debate_mean_intent)} | "
            f"{_signed_num(row.delta_intent_vs_single)} | "
            f"{_pct(row.debate_fail_ratio)} | {_pct(row.debate_low_confidence_ratio)} |"
        )

    lines.append("\n## 2. KOSIS 보유율 대조")
    lines.append("\n| 상품 | single MAE | debate MAE | ΔMAE | single rho | debate rho | Δrho |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report.rows:
        lines.append(
            f"| {row.product_name} | {_num(row.single_holding_mae)} | {_num(row.debate_holding_mae)} | "
            f"{_signed_num(row.delta_mae_vs_single)} | {_num(row.single_holding_spearman)} | "
            f"{_num(row.debate_holding_spearman)} | {_signed_num(row.delta_spearman_vs_single)} |"
        )

    lines.append("\n## 3. 데이터 출처 확인")
    lines.append("\n| 상품 | 페르소나 풀 | Nemotron | 합성 폴백 |")
    lines.append("|---|---:|---:|---:|")
    for row in report.rows:
        lines.append(
            f"| {row.product_name} | {row.persona_pool_size} | {row.persona_nemotron_count} | "
            f"{row.persona_synthetic_count} |"
        )

    lines.append("\n## 4. 해석상 주의")
    lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _signed_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.1%}"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _signed_num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"
