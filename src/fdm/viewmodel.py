"""표시용 뷰모델 — 화면에 무엇을 어떤 순서로 보일지 계산한다.

**여기에는 streamlit이 들어오지 않는다.** UI 파일은 이 모듈이 만든 값을 그리기만 한다.
그래야 (1) 계산을 LLM·브라우저 없이 테스트할 수 있고, (2) 같은 계산을 여러 UI가
공유하며, (3) UI를 갈아엎어도 '무엇을 먼저 보여줄지'에 대한 판단이 남는다.

이 프로젝트에서 특히 중요한 이유: 우려 계층(교차확인 × 심각도)이 측정으로 검증된
유일한 정렬 장치인데(T1 62.5% > T2 53.3% > T3 42.1% > T4 0%, 세 arm 모두 단조),
UI가 바뀔 때마다 그 정렬이 사라지면 산출물이 오탐에 묻힌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents.schema import Concern
from .concerns import TIER_BASIS, TIER_CAVEAT, TIER_LABEL, TIER_MARK, TIER_ORDER, type_label
from .eval.confidence import ConsensusResult
from .eval.simulate import SegmentResult, SimulationReport

# 상위 두 계층은 펼쳐서 보여준다. T3 이하는 접는다 — 먼저 읽혀야 할 것이 위에 와야 한다.
EXPANDED_TIERS = ("T1", "T2")

VERDICT_LABEL = {"pass": "🟢 적합", "warn": "🟡 조건부", "fail": "🔴 부적합"}
FLAG_MARK = {"정상": "🟢", "조건 보완 권고": "🟡", "판매원칙 위험": "🔴", "추가 검증 필요": "⚪"}


# --------------------------------------------------------------------- 우려
@dataclass(frozen=True)
class ConcernView:
    tier: str
    mark: str
    tier_label: str
    severity: str
    cross_checked: bool
    type_label: str
    statement: str
    anchor: str
    verify_with: str
    run_ratio: float | None = None

    @property
    def source_label(self) -> str:
        return "교차확인" if self.cross_checked else "단독"

    @property
    def badge(self) -> str:
        return f"{self.severity}·{self.source_label}"


def concern_view(c: Concern, run_ratio: float | None = None) -> ConcernView:
    return ConcernView(
        tier=c.tier,
        mark=TIER_MARK[c.tier],
        tier_label=c.tier_label,
        severity=c.severity,
        cross_checked=c.cross_checked,
        type_label=type_label(c.type),
        statement=c.statement,
        anchor=c.anchor,
        verify_with=c.verify_with,
        run_ratio=run_ratio,
    )


@dataclass(frozen=True)
class TierGroup:
    tier: str
    mark: str
    label: str
    basis: str
    items: list[ConcernView]
    expanded: bool

    @property
    def heading(self) -> str:
        return f"{self.mark} {self.label} ({len(self.items)}건)"


# ----------------------------------------------------------------- 케이스
@dataclass(frozen=True)
class CaseView:
    persona_id: str
    verdict: str
    verdict_label: str
    intent_mean: float
    intent_range: str
    confidence: float
    confidence_level: str
    needs_review: bool
    label_counts: dict[str, int]
    label_agreement: float
    judge_self_confidence: float
    ungrounded_turns: int
    concerns: list[ConcernView]
    evidence: list[str]
    recommendations: list[str]
    grounding_doc_ids: list[str]

    @property
    def heading(self) -> str:
        tail = "  ⚠ 추가 검증 필요" if self.needs_review else ""
        return (
            f"{self.persona_id[:8]} — {self.verdict_label} / 의향 {self.intent_mean}"
            f" / 신뢰도 {self.confidence:.2f}{tail}"
        )


def case_view(c: ConsensusResult) -> CaseView:
    # 계층 → 심각도 → 교차확인 순으로 정렬해서 넘긴다 (UI가 다시 정렬하지 않게)
    ordered = sorted(c.concerns, key=lambda x: x.sort_key)
    return CaseView(
        persona_id=c.persona_id,
        verdict=c.modal_suitability,
        verdict_label=VERDICT_LABEL.get(c.modal_suitability, c.modal_suitability),
        intent_mean=c.intent_mean,
        intent_range=f"{c.intent_min}~{c.intent_max} (σ={c.intent_std})",
        confidence=c.confidence,
        confidence_level=c.confidence_level,
        needs_review=c.needs_review,
        label_counts=c.label_counts,
        label_agreement=c.label_agreement,
        judge_self_confidence=c.judge_self_confidence,
        ungrounded_turns=c.ungrounded_turns,
        concerns=[concern_view(x, c.concern_run_ratio.get(x.type)) for x in ordered],
        evidence=list(c.evidence),
        recommendations=list(c.recommendations),
        grounding_doc_ids=list(c.grounding_doc_ids),
    )


# --------------------------------------------------------------- 세그먼트
@dataclass(frozen=True)
class SegmentView:
    name: str
    n_personas: int
    mean_intent: float
    adoption_rate: float
    verdict_mix: dict[str, float]
    confidence: float
    flag: str
    flag_mark: str
    tiers: list[TierGroup]
    cases: list[CaseView]

    @property
    def verdict_mix_text(self) -> str:
        return "/".join(f"{self.verdict_mix.get(k, 0):.0%}" for k in ("pass", "warn", "fail"))

    @property
    def tier_counts(self) -> dict[str, int]:
        return {g.tier: len(g.items) for g in self.tiers}

    @property
    def action_needed(self) -> int:
        """T1+T2 — 지금 손대야 하는 건수."""
        return sum(len(g.items) for g in self.tiers if g.tier in EXPANDED_TIERS)

    @property
    def counts_badge(self) -> str:
        return " · ".join(f"{g.mark}{len(g.items)}" for g in self.tiers)

    @property
    def has_concerns(self) -> bool:
        return any(g.items for g in self.tiers)


def segment_view(s: SegmentResult) -> SegmentView:
    grouped = s.tiers
    return SegmentView(
        name=s.segment,
        n_personas=s.n_personas,
        mean_intent=s.mean_intent,
        adoption_rate=s.adoption_rate,
        verdict_mix=dict(s.verdict_mix),
        confidence=s.mean_confidence,
        flag=s.flag,
        flag_mark=FLAG_MARK.get(s.flag, ""),
        tiers=[
            TierGroup(
                tier=t,
                mark=TIER_MARK[t],
                label=TIER_LABEL[t],
                basis=TIER_BASIS[t],
                items=[concern_view(c) for c in grouped[t]],
                expanded=t in EXPANDED_TIERS,
            )
            for t in TIER_ORDER
        ],
        cases=[case_view(c) for c in s.cases],
    )


# ------------------------------------------------------------------ 히트맵
@dataclass(frozen=True)
class HeatCell:
    text: str
    ratio: float  # 0(나쁨)~1(좋음). 색은 UI가 정한다


@dataclass(frozen=True)
class Heatmap:
    columns: list[str]
    rows: list[str]
    cells: list[list[HeatCell]]


def heatmap(segments: list[SegmentView]) -> Heatmap:
    """지표마다 스케일이 달라 지표별로 정규화한다. fail비율은 낮을수록 좋으므로 뒤집는다."""
    specs: list[tuple[str, list[float], list[str]]] = [
        ("가입의향", [s.mean_intent for s in segments], [f"{s.mean_intent:.0f}" for s in segments]),
        ("가입률", [s.adoption_rate for s in segments], [f"{s.adoption_rate:.0%}" for s in segments]),
        ("신뢰도", [s.confidence for s in segments], [f"{s.confidence:.2f}" for s in segments]),
        (
            "fail비율",
            [-s.verdict_mix.get("fail", 0) for s in segments],
            [f"{s.verdict_mix.get('fail', 0):.0%}" for s in segments],
        ),
    ]
    cells: list[list[HeatCell]] = []
    for i, _ in enumerate(segments):
        row = []
        for _, vals, texts in specs:
            lo, hi = min(vals), max(vals)
            ratio = 0.5 if hi <= lo else (vals[i] - lo) / (hi - lo)
            row.append(HeatCell(text=texts[i], ratio=ratio))
        cells.append(row)
    return Heatmap(
        columns=[name for name, _, _ in specs], rows=[s.name for s in segments], cells=cells
    )


# ------------------------------------------------------------------- 전체
@dataclass(frozen=True)
class TierSummary:
    """계층 하나의 표시 정보 + 전체 건수. UI가 지표 카드와 범례에 그대로 쓴다."""

    tier: str
    mark: str
    label: str
    basis: str
    count: int


@dataclass(frozen=True)
class SimulationView:
    product_id: str
    product_name: str
    mode: str
    subtitle: str
    segments: list[SegmentView]
    tier_summary: list[TierSummary]
    caveat: str
    mode_warning: str = ""
    heat: Heatmap = field(default=None)  # type: ignore[assignment]

    @property
    def tier_totals(self) -> dict[str, int]:
        return {t.tier: t.count for t in self.tier_summary}

    @property
    def action_needed(self) -> int:
        return sum(s.action_needed for s in self.segments)


def build_view(sim: SimulationReport) -> SimulationView:
    segs = [segment_view(s) for s in sim.segments]
    totals = {t: 0 for t in TIER_ORDER}
    for s in segs:
        for t, n in s.tier_counts.items():
            totals[t] += n

    warning = ""
    if sim.mode != "ensemble":
        # 교차확인은 단발·디베이트가 둘 다 제기해야 성립한다 → T1이 구조적으로 불가능
        warning = (
            f"`{sim.mode}` 단독 실행이라 **교차확인이 성립하지 않습니다**. "
            "모든 우려가 '단독'으로 계층화되고 T1(즉시 조치)은 나올 수 없습니다. "
            "모드를 `ensemble`로 두고 다시 실행하세요."
        )

    return SimulationView(
        product_id=sim.product_id,
        product_name=sim.product_name,
        mode=sim.mode,
        subtitle=(
            f"`{sim.product_id}` · {sim.mode} · 시드 {sim.n_seeds}회 · "
            f"{sim.backend}/{sim.model_judge} · {sim.generated_at}"
        ),
        segments=segs,
        tier_summary=[
            TierSummary(
                tier=t, mark=TIER_MARK[t], label=TIER_LABEL[t],
                basis=TIER_BASIS[t], count=totals[t],
            )
            for t in TIER_ORDER
        ],
        caveat=TIER_CAVEAT,
        mode_warning=warning,
        heat=heatmap(segs),
    )


# --------------------------------------------------------- 텍스트 포맷 도우미
def md_table(headers: list[str], rows: list[list[object]]) -> str:
    """마크다운 표. pandas가 없는 환경에서도 표를 그리기 위한 것이다.

    (이 개발 PC는 앱 제어 정책이 pandas의 확장 모듈을 차단해
     st.dataframe / st.table / altair 를 쓸 수 없다.)
    """
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def segment_table(segments: list[SegmentView]) -> str:
    return md_table(
        ["세그먼트", "n", "가입의향", "가입률", "pass/warn/fail", "신뢰도", "상태", "조치필요"],
        [
            [
                s.name, s.n_personas, s.mean_intent, f"{s.adoption_rate:.0%}",
                s.verdict_mix_text, f"{s.confidence:.2f}",
                f"{s.flag_mark} {s.flag}", s.action_needed,
            ]
            for s in segments
        ],
    )


def tier_legend_table(view: SimulationView) -> str:
    return md_table(
        ["계층", "건수", "판단 근거 (실측)"],
        [[f"{t.mark} **{t.label}**", t.count, t.basis] for t in view.tier_summary],
    )


def case_concern_table(case: CaseView) -> str:
    return md_table(
        ["계층", "심각도", "교차확인", "재현", "유형", "내용"],
        [
            [
                f"{c.mark} {c.tier_label}", c.severity,
                "O" if c.cross_checked else "—",
                f"{c.run_ratio:.0%}" if c.run_ratio is not None else "—",
                c.type_label, c.statement,
            ]
            for c in case.concerns
        ],
    )
