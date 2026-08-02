"""Streamlit 프로토타입: FDM Product Workbench.

실행: uv run streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdm.config import OUTPUT_DIR, SETTINGS  # noqa: E402
from fdm.eval.benchmark import AblationReport, compare_holding_rates  # noqa: E402
from fdm.eval.simulate import (  # noqa: E402
    SimulationReport,
    default_variants,
    sensitivity_analysis,
    simulate_product,
)
from fdm.personas.loader import load_personas  # noqa: E402
from fdm.personas.schema import Persona, Segment  # noqa: E402
from fdm.products.schema import Clause, Fee, Preferential, Product  # noqa: E402
from fdm.report import build_report, save_report  # noqa: E402
from fdm.services.workbench import (  # noqa: E402
    PRESET_CONFIGS,
    WorkbenchDB,
    estimate_run_cost,
    load_sensitivity_rows,
    safe_slug,
    segment_profile,
    validate_product_for_workbench,
)

st.set_page_config(page_title="FDM Product Workbench", layout="wide")

CATEGORY_LABELS = {
    "saving": "적금",
    "deposit": "예금",
    "loan": "대출",
    "card": "카드",
    "pension": "연금",
    "fund": "펀드",
}
LABEL_TO_CATEGORY = {v: k for k, v in CATEGORY_LABELS.items()}
MODE_OPTIONS = ["single", "debate"]
DEFAULT_PERSONA_SOURCE = "synthetic"
DEFAULT_PERSONA_LIMIT = 400
RISK_COLORS = {
    "정상": "#2f6f4e",
    "조건 보완 권고": "#b7791f",
    "판매원칙 위험": "#b91c1c",
    "추가 검증 필요": "#4b5563",
}
DIAGNOSTIC_ITEM_PREFIXES = (
    "LLM JSON 파싱/호출 실패",
    "일부 시드에서 LLM JSON 파싱/호출 실패",
    "seed=",
)
DIAGNOSTIC_ITEM_MARKERS = (
    "FDM_LLM_",
    "FDM_OPENAI_",
    "FDM_GEMINI_",
    "generativelanguage.googleapis.com",
    "openai/chat/completions",
    "JSON 파싱 실패",
    "연결 실패:",
    "호출 실패:",
)
CATEGORY_FORM_COPY = {
    "saving": {
        "finance_title": "적금 금리·기간·납입",
        "rate1": "기본금리(연 %)",
        "rate2": "최고우대금리(연 %)",
        "term": "저축기간(개월)",
        "amount_min": "월 납입 최소(만원)",
        "amount_max": "월 납입 최대(만원)",
        "limit": None,
        "condition_title": "우대조건",
        "condition_count": "우대조건 수",
        "condition_name": "조건명",
        "condition_bonus": "우대금리(%p)",
        "condition_req": "달성 요건",
        "condition_default": "우대조건",
        "fee_title": "수수료·과세·중도해지",
        "tax": "과세",
        "early": "중도해지",
        "risk": "유의사항(줄바꿈으로 구분)",
    },
    "deposit": {
        "finance_title": "예금 금리·기간·가입금액",
        "rate1": "기본금리(연 %)",
        "rate2": "최고금리(연 %)",
        "term": "예치기간(개월)",
        "amount_min": "가입금액 최소(만원)",
        "amount_max": "가입금액 최대(만원)",
        "limit": None,
        "condition_title": "우대조건",
        "condition_count": "우대조건 수",
        "condition_name": "조건명",
        "condition_bonus": "우대금리(%p)",
        "condition_req": "달성 요건",
        "condition_default": "우대조건",
        "fee_title": "수수료·과세·중도해지",
        "tax": "과세",
        "early": "중도해지",
        "risk": "유의사항(줄바꿈으로 구분)",
    },
    "loan": {
        "finance_title": "대출 금리·기간·한도",
        "rate1": "최저금리(연 %)",
        "rate2": "최고금리(연 %)",
        "term": "상환기간(개월)",
        "amount_min": None,
        "amount_max": None,
        "limit": "대출 한도(만원)",
        "condition_title": "금리 감면·한도 우대 조건",
        "condition_count": "감면/우대 조건 수",
        "condition_name": "조건명",
        "condition_bonus": "금리 인하폭(%p)",
        "condition_req": "적용 요건",
        "condition_default": "금리 감면 조건",
        "fee_title": "수수료·상환·담보/보증",
        "tax": "인지세/세금",
        "early": "중도상환 조건",
        "risk": "상환·담보·연체 유의사항(줄바꿈으로 구분)",
    },
    "card": {
        "finance_title": "카드 한도·실적 기간",
        "rate1": None,
        "rate2": None,
        "term": "실적 산정기간(개월)",
        "amount_min": None,
        "amount_max": None,
        "limit": "카드 이용 한도(만원)",
        "condition_title": "혜택·실적 조건",
        "condition_count": "혜택/실적 조건 수",
        "condition_name": "혜택명",
        "condition_bonus": "혜택률/적립률(%)",
        "condition_req": "실적/적용 조건",
        "condition_default": "혜택 조건",
        "fee_title": "연회비·수수료·이용 조건",
        "tax": "연회비/부가서비스 설명",
        "early": "해지·혜택 회수 조건",
        "risk": "실적 제외·혜택 제한 유의사항(줄바꿈으로 구분)",
    },
    "pension": {
        "finance_title": "연금 수익률·납입·기간",
        "rate1": "공시/기준수익률(연 %)",
        "rate2": "예상/최고수익률(연 %)",
        "term": "납입/거치기간(개월)",
        "amount_min": "월 납입 최소(만원)",
        "amount_max": "월 납입 최대(만원)",
        "limit": None,
        "condition_title": "세제·납입 우대조건",
        "condition_count": "우대조건 수",
        "condition_name": "조건명",
        "condition_bonus": "혜택/가산폭",
        "condition_req": "적용 요건",
        "condition_default": "연금 우대조건",
        "fee_title": "수수료·세제·중도인출",
        "tax": "세제 혜택/과세",
        "early": "중도인출/해지 조건",
        "risk": "원금손실·세제 추징 유의사항(줄바꿈으로 구분)",
    },
    "fund": {
        "finance_title": "펀드 수익률·투자금액·기간",
        "rate1": "기준/목표수익률(연 %)",
        "rate2": "예상/최고수익률(연 %)",
        "term": "권장 투자기간(개월)",
        "amount_min": "투자금액 최소(만원)",
        "amount_max": "투자금액 최대(만원)",
        "limit": None,
        "condition_title": "수수료·운용 조건",
        "condition_count": "운용/우대 조건 수",
        "condition_name": "조건명",
        "condition_bonus": "수수료 인하/혜택폭",
        "condition_req": "적용 요건",
        "condition_default": "펀드 우대조건",
        "fee_title": "보수·수수료·환매",
        "tax": "과세",
        "early": "환매 조건",
        "risk": "투자위험·환매 유의사항(줄바꿈으로 구분)",
    },
}
@st.cache_resource(show_spinner=False)
def db() -> WorkbenchDB:
    store = WorkbenchDB()
    store.initialize()
    return store


@st.cache_data(show_spinner=False)
def cached_personas(source: str, limit: int) -> list[Persona]:
    return load_personas(
        source=source,  # type: ignore[arg-type]
        limit=limit,
        allow_synthetic_fallback=True,
    )


def selected_version(store: WorkbenchDB):
    vid = st.session_state.get("active_version_id")
    if vid is None:
        return None
    try:
        return store.get_product_version(int(vid))
    except Exception:
        st.session_state.pop("active_version_id", None)
        return None


def selected_run(store: WorkbenchDB):
    rid = st.session_state.get("active_run_id")
    if rid is None:
        return None
    try:
        return store.get_simulation_run(int(rid))
    except Exception:
        st.session_state.pop("active_run_id", None)
        return None


def product_version_label(record) -> str:
    return f"{record.product.name} · v{record.version_number} · {record.created_at}"


def run_label(run) -> str:
    suffix = run.finished_at or run.started_at
    return f"#{run.id} {run.product_name} v{run.version_number} · {run.preset} · {run.status} · {suffix}"


def split_lines(text: str) -> list[str]:
    return [x.strip() for x in text.splitlines() if x.strip()]


def split_csv(text: str) -> list[str] | None:
    items = [x.strip() for x in text.split(",") if x.strip()]
    return items or None


def issues_panel(product: Product) -> list[str]:
    issues = validate_product_for_workbench(product)
    errors = [i.message for i in issues if i.severity == "error"]
    warnings = [i.message for i in issues if i.severity == "warning"]
    if errors:
        for msg in errors:
            st.error(msg)
    if warnings:
        for msg in warnings:
            st.warning(msg)
    if not issues:
        st.success("상품 입력값이 현재 Product 스키마와 업무 검증 기준을 통과했습니다.")
    return errors


def simulation_dataframe(sim: SimulationReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "세그먼트": s.segment,
                "페르소나수": s.n_personas,
                "가입의향": s.mean_intent,
                "가입률": s.adoption_rate,
                "pass": s.verdict_mix.get("pass", 0.0),
                "warn": s.verdict_mix.get("warn", 0.0),
                "fail": s.verdict_mix.get("fail", 0.0),
                "신뢰도": s.mean_confidence,
                "저신뢰비율": s.low_confidence_ratio,
                "상태": s.flag,
            }
            for s in sim.segments
        ]
    )


def is_diagnostic_item(item: str) -> bool:
    text = item.strip()
    if not text:
        return False
    return text.startswith(DIAGNOSTIC_ITEM_PREFIXES) or any(
        marker in text for marker in DIAGNOSTIC_ITEM_MARKERS
    )


def top_items(items: list[str], n: int = 5) -> list[tuple[str, int]]:
    return Counter(
        x.strip()
        for x in items
        if x and x.strip() and not is_diagnostic_item(x)
    ).most_common(n)


def scenario_delta_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = df[df["label"] == "기준안"][
        ["segment", "adoption_rate", "mean_intent", "mean_confidence", "fail_ratio"]
    ].rename(
        columns={
            "adoption_rate": "base_adoption_rate",
            "mean_intent": "base_mean_intent",
            "mean_confidence": "base_mean_confidence",
            "fail_ratio": "base_fail_ratio",
        }
    )
    merged = df.merge(base, on="segment", how="left")
    merged["가입의향 변화"] = (merged["mean_intent"] - merged["base_mean_intent"]).round(1)
    merged["가입률 변화"] = (merged["adoption_rate"] - merged["base_adoption_rate"]).round(3)
    merged["fail 비율 변화"] = (merged["fail_ratio"] - merged["base_fail_ratio"]).round(3)
    merged["신뢰도 변화"] = (merged["mean_confidence"] - merged["base_mean_confidence"]).round(3)
    merged["주의 플래그"] = merged.apply(
        lambda r: "가입의향 상승과 위험 증가가 동시에 발생"
        if r["label"] != "기준안" and r["가입의향 변화"] > 0 and r["fail 비율 변화"] > 0
        else "",
        axis=1,
    )
    return merged


def ensure_personas(source: str, limit: int) -> list[Persona]:
    try:
        return cached_personas(source, limit)
    except Exception as exc:
        st.error(f"페르소나 로딩 실패: {exc}")
        st.stop()


def product_from_form(template: Product, segment_names: list[str]) -> Product | None:
    st.subheader("상품 설계 캔버스")

    c1, c2, c3, c4 = st.columns([1.1, 1.4, 1.2, 1.0])
    product_id = c1.text_input("상품 ID", value=template.product_id, key="product_id")
    name = c2.text_input("상품명", value=template.name, key="product_name")
    issuer = c3.text_input("발행기관", value=template.issuer, key="issuer")
    category_label = c4.selectbox(
        "상품군",
        list(LABEL_TO_CATEGORY.keys()),
        index=list(LABEL_TO_CATEGORY).index(CATEGORY_LABELS.get(template.category, "적금")),
        key="category",
    )
    category = LABEL_TO_CATEGORY[category_label]
    copy = CATEGORY_FORM_COPY[category]
    template_preferentials = template.preferentials or []
    template_fees = template.fees or []

    summary = st.text_area("상품 요약", value=template.summary, height=90, key="summary")

    st.markdown(f"**{copy['finance_title']}**")
    intr_rate = None
    intr_rate2 = None
    save_trm_months = None
    min_monthly = None
    max_monthly = None
    limit_manwon = None
    rate_basis = template.rate_basis
    intr_rate_type = template.intr_rate_type

    if copy["rate1"]:
        r1, r2, r3, r4, r5 = st.columns(5)
        intr_rate = r1.number_input(
            copy["rate1"],
            min_value=0.0,
            max_value=40.0,
            value=float(template.intr_rate if template.intr_rate is not None else 3.5),
            step=0.1,
            key="intr_rate",
        )
        intr_rate2 = r2.number_input(
            copy["rate2"],
            min_value=0.0,
            max_value=40.0,
            value=float(template.intr_rate2 if template.intr_rate2 is not None else intr_rate),
            step=0.1,
            key="intr_rate2",
        )
        save_trm_months = r3.number_input(
            copy["term"],
            min_value=0,
            max_value=600,
            value=int(template.save_trm_months or 12),
            step=1,
            key="save_trm_months",
        )
        rate_basis = r4.selectbox(
            "고정/변동",
            ["고정", "변동"],
            index=0 if template.rate_basis == "고정" else 1,
            key="rate_basis",
        )
        intr_rate_type = r5.text_input("금리/수익률 유형", value=template.intr_rate_type, key="intr_rate_type")
    else:
        c1, c2 = st.columns(2)
        limit_manwon = c1.number_input(
            copy["limit"],
            min_value=0,
            max_value=100000,
            value=int(template.limit_manwon or 300),
            step=50,
            key="limit_manwon",
        )
        save_trm_months = c2.number_input(
            copy["term"],
            min_value=0,
            max_value=60,
            value=int(template.save_trm_months or 1),
            step=1,
            key="save_trm_months",
        )
        rate_basis = "고정"
        intr_rate_type = "혜택형"
        st.caption("카드 상품은 금리 입력 대신 이용 한도, 실적 기간, 혜택/수수료 조건을 중심으로 검증합니다.")

    amount_cols = []
    if copy["amount_min"] and copy["amount_max"]:
        amount_cols.extend(["min", "max"])
    if copy["limit"] and copy["rate1"]:
        amount_cols.append("limit")
    if amount_cols:
        cols = st.columns(len(amount_cols))
        idx = 0
        if "min" in amount_cols:
            min_monthly = cols[idx].number_input(
                copy["amount_min"],
                min_value=0,
                max_value=50000,
                value=int(template.min_monthly_manwon or 0),
                step=5,
                key="min_monthly",
            )
            idx += 1
        if "max" in amount_cols:
            max_monthly = cols[idx].number_input(
                copy["amount_max"],
                min_value=0,
                max_value=50000,
                value=int(template.max_monthly_manwon or 50),
                step=5,
                key="max_monthly",
            )
            idx += 1
        if "limit" in amount_cols:
            limit_manwon = cols[idx].number_input(
                copy["limit"],
                min_value=0,
                max_value=100000,
                value=int(template.limit_manwon or 10000),
                step=100,
                key="limit_manwon",
            )

    st.markdown(f"**{copy['condition_title']}**")
    pref_count = st.number_input(
        copy["condition_count"],
        min_value=0,
        max_value=8,
        value=max(1, len(template_preferentials)),
        step=1,
        key="pref_count",
    )
    preferentials: list[Preferential] = []
    for i in range(int(pref_count)):
        base = template_preferentials[i] if i < len(template_preferentials) else None
        p1, p2, p3, p4 = st.columns([1.1, 0.7, 2.4, 0.8])
        pname = p1.text_input(copy["condition_name"], value=base.name if base else "", key=f"pref_name_{i}")
        bonus = p2.number_input(
            copy["condition_bonus"],
            min_value=0.0,
            max_value=10.0,
            value=float(base.rate_bonus_pct if base else 0.0),
            step=0.1,
            key=f"pref_bonus_{i}",
        )
        requirement = p3.text_input(copy["condition_req"], value=base.requirement if base else "", key=f"pref_req_{i}")
        attain = p4.number_input(
            "추정 달성률",
            min_value=0.0,
            max_value=1.0,
            value=float(base.est_attainment_rate if base and base.est_attainment_rate is not None else 0.4),
            step=0.05,
            key=f"pref_attain_{i}",
        )
        if pname or requirement:
            preferentials.append(
                Preferential(
                    name=pname or f"{copy['condition_default']} {i + 1}",
                    rate_bonus_pct=bonus,
                    requirement=requirement,
                    est_attainment_rate=attain,
                )
            )

    st.markdown(f"**{copy['fee_title']}**")
    fee_count = st.number_input(
        "수수료 항목 수",
        min_value=0,
        max_value=12,
        value=len(template_fees),
        step=1,
        key="fee_count",
    )
    fees: list[Fee] = []
    for i in range(int(fee_count)):
        base = template_fees[i] if i < len(template_fees) else None
        f1, f2, f3 = st.columns([1.0, 1.0, 2.0])
        fname = f1.text_input("수수료명", value=base.name if base else "", key=f"fee_name_{i}")
        amount = f2.text_input("금액", value=base.amount if base else "", key=f"fee_amount_{i}")
        condition = f3.text_area("조건", value=base.condition if base else "", height=90, key=f"fee_condition_{i}")
        if fname or amount or condition:
            fees.append(Fee(name=fname or f"수수료 {i + 1}", amount=amount or "별도 고지", condition=condition))

    taxation = st.text_input(copy["tax"], value=template.taxation, key="taxation")
    early_termination = st.text_area(
        copy["early"],
        value=template.early_termination,
        height=70,
        key="early_termination",
    )
    risk_notes = split_lines(
        st.text_area(
            copy["risk"],
            value="\n".join(template.risk_notes),
            height=100,
            key="risk_notes",
        )
    )

    st.markdown("**타깃과 근거**")
    target_description = st.text_area("타깃 설명", value=template.target_description, height=110, key="target_description")
    target_segments = st.multiselect(
        "타깃 세그먼트",
        segment_names,
        default=[s for s in template.target_segments if s in segment_names],
        key="target_segments",
    )
    clause_count = st.number_input(
        "약관/설명서 조항 수",
        min_value=0,
        max_value=15,
        value=max(1, len(template.clauses)),
        step=1,
        key="clause_count",
    )
    clauses: list[Clause] = []
    for i in range(int(clause_count)):
        base = template.clauses[i] if i < len(template.clauses) else None
        c1, c2 = st.columns([0.8, 1.2])
        cid = c1.text_input("조항 ID", value=base.id if base else "", key=f"clause_id_{i}")
        title = c2.text_input("조항 제목", value=base.title if base else "", key=f"clause_title_{i}")
        text = st.text_area("조항 본문", value=base.text if base else "", height=70, key=f"clause_text_{i}")
        if cid or title or text:
            clauses.append(Clause(id=cid or f"조항 {i + 1}", title=title or "제목 미입력", text=text))

    try:
        return Product(
            product_id=product_id.strip(),
            name=name.strip(),
            category=category,  # type: ignore[arg-type]
            issuer=issuer.strip(),
            summary=summary.strip(),
            intr_rate=intr_rate if intr_rate is not None and intr_rate > 0 else None,
            intr_rate2=intr_rate2 if intr_rate2 is not None and intr_rate2 > 0 else None,
            intr_rate_type=intr_rate_type.strip() or "단리",
            rate_basis=rate_basis,  # type: ignore[arg-type]
            save_trm_months=int(save_trm_months) if save_trm_months else None,
            min_monthly_manwon=int(min_monthly) if min_monthly else None,
            max_monthly_manwon=int(max_monthly) if max_monthly else None,
            limit_manwon=int(limit_manwon) if limit_manwon else None,
            preferentials=preferentials,
            fees=fees,
            taxation=taxation.strip() or "이자소득세 15.4% 원천징수",
            early_termination=early_termination.strip(),
            risk_notes=risk_notes,
            target_description=target_description.strip(),
            target_segments=target_segments,
            clauses=clauses,
        )
    except ValidationError as exc:
        st.error("Product 스키마 검증에 실패했습니다.")
        st.code(str(exc), language="text")
        return None


def segment_builder(store: WorkbenchDB, personas: list[Persona]) -> list[Segment]:
    records = store.list_segment_definitions()
    names = [r.name for r in records]
    active_version = selected_version(store)
    default_names = active_version.product.target_segments if active_version else names[:2]
    default_names = [n for n in default_names if n in names] or names[:2]

    st.subheader("세그먼트 빌더")
    selected_names = st.multiselect(
        "검증 대상 세그먼트",
        names,
        default=st.session_state.get("selected_segment_names", default_names),
        key="segment_multiselect",
    )
    st.session_state["selected_segment_names"] = selected_names
    selected = [r.segment for r in records if r.name in selected_names]

    if selected:
        profile_rows = [segment_profile(seg, personas) for seg in selected]
        st.dataframe(
            pd.DataFrame(profile_rows).rename(
                columns={
                    "name": "세그먼트",
                    "n_personas": "예상 표본",
                    "avg_income_manwon": "평균 연소득(만원)",
                    "avg_dsr_pct": "평균 DSR(%)",
                    "avg_surplus_manwon": "평균 월여유(만원)",
                    "synthetic_ratio": "합성 폴백 비율",
                    "status": "상태",
                }
            )[
                [
                    "세그먼트",
                    "예상 표본",
                    "평균 연소득(만원)",
                    "평균 DSR(%)",
                    "평균 월여유(만원)",
                    "합성 폴백 비율",
                    "상태",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        if any(r["n_personas"] == 0 for r in profile_rows):
            st.error("예상 표본이 0명인 세그먼트는 실행할 수 없습니다.")
        if any(0 < r["n_personas"] < 5 for r in profile_rows):
            st.warning("표본이 5명 미만인 세그먼트가 있습니다. 결과는 방향성으로만 해석하세요.")
        if any(r["synthetic_ratio"] >= 0.5 and r["n_personas"] > 0 for r in profile_rows):
            st.info("합성 폴백 페르소나 비율이 높은 세그먼트가 있습니다. 실제 Nemotron 캐시 기반 재검증을 권장합니다.")
    else:
        st.info("검증 대상 세그먼트를 하나 이상 선택하세요.")

    st.divider()
    st.markdown("**커스텀 세그먼트 저장**")
    base_name = st.selectbox("기준 프리셋", ["빈 세그먼트"] + names, key="segment_base")
    base = next((r.segment for r in records if r.name == base_name), Segment(name="커스텀_세그먼트"))
    key_suffix = base_name.replace(" ", "_")

    with st.form("custom_segment_form"):
        name = st.text_input(
            "세그먼트명",
            value=f"{base.name}_커스텀" if base_name != "빈 세그먼트" else "커스텀_세그먼트",
        )
        c1, c2, c3, c4 = st.columns(4)
        age_min = c1.number_input("나이 최소", 19, 95, int(base.age_min or 19), key=f"age_min_{key_suffix}")
        age_max = c2.number_input("나이 최대", 19, 95, int(base.age_max or 75), key=f"age_max_{key_suffix}")
        income_min = c3.number_input(
            "연소득 최소(만원)",
            0,
            50000,
            int(base.income_min_manwon or 0),
            step=100,
            key=f"income_min_{key_suffix}",
        )
        income_max = c4.number_input(
            "연소득 최대(만원)",
            0,
            50000,
            int(base.income_max_manwon or 0),
            step=100,
            key=f"income_max_{key_suffix}",
        )
        d1, d2, d3 = st.columns(3)
        quintiles = d1.multiselect(
            "소득 분위",
            [1, 2, 3, 4, 5],
            default=base.income_quintiles or [],
            key=f"quintiles_{key_suffix}",
        )
        dsr_min = d2.number_input(
            "DSR 하한(%)",
            0.0,
            100.0,
            float(base.dsr_min_pct or 0.0),
            step=1.0,
            key=f"dsr_{key_suffix}",
        )
        surplus_max = d3.number_input(
            "월 여유자금 상한(만원)",
            0,
            5000,
            int(base.monthly_surplus_max_manwon or 0),
            step=10,
            key=f"surplus_{key_suffix}",
        )
        e1, e2, e3 = st.columns(3)
        occupations = e1.text_input(
            "직업 키워드(쉼표 구분)",
            value=", ".join(base.occupations_include or []),
            key=f"occ_{key_suffix}",
        )
        regions = e2.text_input(
            "지역 키워드(쉼표 구분)",
            value=", ".join(base.regions or []),
            key=f"region_{key_suffix}",
        )
        sex = e3.selectbox(
            "성별",
            ["전체", "남성", "여성"],
            index=["전체", "남성", "여성"].index(base.sex) if base.sex in {"남성", "여성"} else 0,
            key=f"sex_{key_suffix}",
        )
        submitted = st.form_submit_button("커스텀 세그먼트 저장")
    if submitted:
        seg = Segment(
            name=name.strip() or "커스텀_세그먼트",
            age_min=int(age_min) if age_min else None,
            age_max=int(age_max) if age_max else None,
            income_min_manwon=int(income_min) if income_min else None,
            income_max_manwon=int(income_max) if income_max else None,
            income_quintiles=quintiles or None,
            dsr_min_pct=float(dsr_min) if dsr_min else None,
            monthly_surplus_max_manwon=int(surplus_max) if surplus_max else None,
            occupations_include=split_csv(occupations),
            regions=split_csv(regions),
            sex=None if sex == "전체" else sex,
        )
        store.save_segment_definition(seg, is_preset=False)
        st.success(f"저장했습니다: {seg.name}")
        st.cache_data.clear()
        st.rerun()

    return selected


def sidebar(store: WorkbenchDB) -> tuple[str, int]:
    with st.sidebar:
        st.subheader("실행 환경")
        persona_source = DEFAULT_PERSONA_SOURCE
        persona_limit = DEFAULT_PERSONA_LIMIT
        for label, value in (
            ("LLM 백엔드", SETTINGS.backend),
            ("토론모델", SETTINGS.model_small),
            ("심판모델", SETTINGS.model_judge),
            ("페르소나 출처", persona_source),
            ("페르소나 로드 수", f"{persona_limit}"),
        ):
            st.markdown(f"**{label}**  \n`{value}`")
        if SETTINGS.backend == "gemini" and (
            not SETTINGS.model_small.startswith("gemini-") or not SETTINGS.model_judge.startswith("gemini-")
        ):
            st.warning("Gemini 백엔드는 Gemini 모델명을 사용해야 합니다.")
        st.caption("LLM 설정은 서버 환경변수 기준이며, 페르소나는 UI 기본값으로 고정됩니다.")

        st.divider()
        st.subheader("현재 작업")
        versions = store.list_product_versions()
        if versions:
            labels = [product_version_label(v) for v in versions]
            current = st.session_state.get("active_version_id", versions[0].id)
            index = next((i for i, v in enumerate(versions) if v.id == current), 0)
            picked = st.selectbox("상품 버전", labels, index=index)
            st.session_state["active_version_id"] = versions[labels.index(picked)].id
        else:
            st.info("저장된 상품 버전이 없습니다.")

        runs = store.list_simulation_runs(limit=50)
        if runs:
            run_labels = [run_label(r) for r in runs]
            current_run = st.session_state.get("active_run_id", runs[0].id)
            run_index = next((i for i, r in enumerate(runs) if r.id == current_run), 0)
            picked_run = st.selectbox("실행 이력", run_labels, index=run_index)
            st.session_state["active_run_id"] = runs[run_labels.index(picked_run)].id
        else:
            st.caption("아직 실행 이력이 없습니다.")

    return persona_source, persona_limit


store = db()
persona_source, persona_limit = sidebar(store)

st.title("FDM Product Workbench")
st.caption(
    "금융상품 설계자가 상품 초안을 입력하고, 합성 페르소나 세그먼트에서 출시 전 위험과 개선안을 검토하는 업무형 프로토타입입니다. "
    "결과는 합성 페르소나 기반 탐색·경보용이며 개별 소비자 추천이나 확정 승인 근거가 아닙니다."
)

tab_portfolio, tab_product, tab_segment, tab_run, tab_result, tab_scenario, tab_report = st.tabs(
    ["포트폴리오", "상품 설계", "세그먼트", "검증 실행", "결과 진단", "시나리오", "보고서"]
)

with tab_portfolio:
    st.subheader("상품 포트폴리오 홈")
    rows = store.portfolio_rows()
    if not rows:
        st.info("저장된 상품 프로젝트가 없습니다. 상품 설계 탭에서 신규 초안을 저장하세요.")
    else:
        df = pd.DataFrame([r.__dict__ for r in rows]).rename(
            columns={
                "project_id": "상품 ID",
                "name": "상품명",
                "category": "상품군",
                "status": "상태",
                "version_number": "최신 버전",
                "average_intent": "평균 가입의향",
                "risk_segments": "위험 세그먼트 수",
                "low_confidence_segments": "저신뢰 세그먼트 수",
                "last_run_at": "마지막 실행",
                "next_action": "다음 액션",
            }
        )
        df["상품군"] = df["상품군"].map(lambda x: CATEGORY_LABELS.get(x, x))
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("상품 프로젝트", len(rows))
        c2.metric("보완 필요", sum(1 for r in rows if r.status == "보완 필요"))
        c3.metric("출시 검토 가능", sum(1 for r in rows if r.status == "출시 검토 가능"))
        c4.metric("실행 이력", len(store.list_simulation_runs(limit=200)))

    with st.expander("최근 감사 로그", expanded=False):
        audit = store.audit_rows(limit=30)
        if audit:
            st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)
        else:
            st.caption("아직 감사 로그가 없습니다.")

with tab_product:
    segment_names = [r.name for r in store.list_segment_definitions()]
    active = selected_version(store)
    if active:
        template = active.product
    else:
        template = Product(
            product_id="NEW-2026-001",
            name="신규 금융상품 초안",
            category="saving",
            issuer="KB국민은행(가상 신상품)",
            summary="상품 관련 설명을 입력해주세요",
            intr_rate=3.0,
            intr_rate2=4.5,
            save_trm_months=12,
            min_monthly_manwon=10,
            max_monthly_manwon=50,
            preferentials=[
                Preferential(name="급여이체 우대", rate_bonus_pct=0.5, requirement="당행 계좌로 급여 입금")
            ],
            early_termination="중도해지 시 우대금리 미적용, 가입기간별 중도해지이율 적용 등",
            risk_notes=["유의사항을 작성해주세요"],
            target_description="상품 설계자가 지정한 타깃 세그먼트",
            target_segments=segment_names[:2],
            clauses=[
                Clause(
                    id="설명서 제1항",
                    title="최고금리 표시",
                    text="최고금리는 모든 우대조건 충족을 전제로 하며 실제 적용금리는 고객별로 달라질 수 있습니다.",
                )
            ],
        )

    product = product_from_form(template, segment_names)
    if product:
        st.markdown("**사전 검증 경고**")
        errors = issues_panel(product)
        change_note = st.text_input("변경 메모", value="UI 상품 설계 캔버스 저장", key="change_note")
        save_disabled = bool(errors)
        if st.button("상품 버전 저장", type="primary", disabled=save_disabled):
            rec = store.save_product_version(product, change_note=change_note)
            st.session_state["active_version_id"] = rec.id
            st.success(f"상품 버전 v{rec.version_number}을 저장했습니다. artifact: {rec.artifact_path}")
            st.rerun()

with tab_segment:
    personas = ensure_personas(persona_source, persona_limit)
    selected_segments = segment_builder(store, personas)

with tab_run:
    active = selected_version(store)
    if active is None:
        st.info("먼저 상품 설계 탭에서 상품 버전을 저장하세요.")
    else:
        personas = ensure_personas(persona_source, persona_limit)
        all_segments = store.list_segment_definitions()
        segment_names = [r.name for r in all_segments]
        default_names = st.session_state.get("selected_segment_names") or active.product.target_segments or segment_names[:2]
        selected_names = st.multiselect(
            "검증 대상 세그먼트",
            segment_names,
            default=[n for n in default_names if n in segment_names],
            key="run_segment_names",
        )
        selected_segments = [r.segment for r in all_segments if r.name in selected_names]
        profiles = [segment_profile(seg, personas) for seg in selected_segments]
        if profiles:
            st.dataframe(
                pd.DataFrame(profiles).rename(
                    columns={
                        "name": "세그먼트",
                        "n_personas": "예상 표본",
                        "avg_income_manwon": "평균 연소득(만원)",
                        "avg_dsr_pct": "평균 DSR(%)",
                        "avg_surplus_manwon": "평균 월여유(만원)",
                        "status": "상태",
                    }
                )[["세그먼트", "예상 표본", "평균 연소득(만원)", "평균 DSR(%)", "평균 월여유(만원)", "상태"]],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("검증 실행 설정")
        preset = st.radio("실행 프리셋", list(PRESET_CONFIGS.keys()), horizontal=True)
        cfg = PRESET_CONFIGS[preset]
        st.caption(cfg["purpose"])
        advanced = st.checkbox("고급 설정 조정", value=False)
        if advanced:
            c1, c2, c3, c4 = st.columns(4)
            mode = c1.selectbox("실행 모드", MODE_OPTIONS, index=MODE_OPTIONS.index(cfg["mode"]))
            n_seeds = c2.slider("멀티시드 수", 1, 7, int(cfg["n_seeds"]))
            personas_per_segment = c3.slider("세그먼트별 페르소나 수", 1, 10, int(cfg["personas_per_segment"]))
            workers = c4.slider("workers", 1, 8, int(cfg["workers"]))
            include_sensitivity = st.checkbox("민감도 분석 포함", value=bool(cfg["include_sensitivity"]))
        else:
            mode = str(cfg["mode"])
            n_seeds = int(cfg["n_seeds"])
            personas_per_segment = int(cfg["personas_per_segment"])
            workers = int(cfg["workers"])
            include_sensitivity = bool(cfg["include_sensitivity"])

        estimate = estimate_run_cost(
            n_segments=len(selected_segments),
            mode=mode,
            n_seeds=n_seeds,
            personas_per_segment=personas_per_segment,
            workers=workers,
            include_sensitivity=include_sensitivity,
            n_variants=len(default_variants(active.product)),
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("예상 케이스", estimate["cases"])
        m2.metric("예상 LLM 호출", estimate["llm_calls"])
        m3.metric("예상 시간", f"{estimate['estimated_seconds']}초")
        m4.metric("백엔드", SETTINGS.backend)
        st.caption(
            f"페르소나 출처 `{persona_source}`, 모델 `{SETTINGS.model_small}` / `{SETTINGS.model_judge}`. "
            "mock 백엔드는 결정론적 스텁으로 데모 흐름 확인에 사용합니다."
        )

        blocked = (
            not selected_segments
            or any(p["n_personas"] == 0 for p in profiles)
            or estimate["cases"] == 0
        )
        if blocked:
            st.warning("실행 가능한 세그먼트와 표본 수를 먼저 확보하세요.")

        if st.button("검증 실행", type="primary", disabled=blocked):
            run_id: int | None = None
            run_settings = {
                "segment_names": selected_names,
                "persona_limit": persona_limit,
                "include_sensitivity": include_sensitivity,
                "estimated": estimate,
            }
            try:
                run_id = store.start_simulation_run(
                    product_version_id=active.id,
                    preset=preset,
                    mode=mode,
                    n_seeds=n_seeds,
                    personas_per_segment=personas_per_segment,
                    workers=workers,
                    persona_source=persona_source,
                    settings=run_settings,
                )
                product_for_run = active.product.model_copy(update={"target_segments": selected_names})
                with st.spinner("검증을 실행하는 중입니다. mock 외 백엔드는 시간이 걸릴 수 있습니다."):
                    sim = simulate_product(
                        product_for_run,
                        segments=selected_segments,
                        k_personas=personas_per_segment,
                        n_seeds=n_seeds,
                        mode=mode,  # type: ignore[arg-type]
                        workers=workers,
                        personas=personas,
                        persona_source=persona_source,  # type: ignore[arg-type]
                        progress=False,
                    )
                    sim_path = OUTPUT_DIR / f"sim_{safe_slug(product_for_run.product_id)}_run_{run_id}.json"
                    sim.save(sim_path)
                    sensitivity_path = None
                    if include_sensitivity:
                        sens_rows = sensitivity_analysis(
                            product_for_run,
                            default_variants(product_for_run),
                            segments=selected_segments,
                            k_personas=max(1, personas_per_segment // 2),
                            n_seeds=max(1, n_seeds - 1),
                            mode=mode,  # type: ignore[arg-type]
                            workers=workers,
                            personas=personas,
                            persona_source=persona_source,  # type: ignore[arg-type]
                        )
                        sensitivity_path = OUTPUT_DIR / f"sensitivity_{safe_slug(product_for_run.product_id)}_run_{run_id}.json"
                        store.save_sensitivity_artifact(run_id, sens_rows, path=sensitivity_path)
                store.complete_simulation_run(run_id, sim, artifact_path=sim_path, sensitivity_path=sensitivity_path)
                st.session_state["active_run_id"] = run_id
                st.success(f"검증이 완료되었습니다. 실행 #{run_id}을 결과 진단 탭에서 확인하세요.")
                st.rerun()
            except Exception as exc:
                if run_id is not None:
                    store.fail_simulation_run(run_id, str(exc))
                st.error("검증 실행 중 오류가 발생했습니다.")
                st.exception(exc)

with tab_result:
    run = selected_run(store)
    if run is None:
        st.info("검증 실행 이력이 없습니다.")
    elif run.status != "완료":
        st.warning(f"선택한 실행은 현재 `{run.status}` 상태입니다.")
        if run.error_summary:
            st.code(run.error_summary, language="text")
    else:
        sim = store.load_simulation_report(run.id)
        df = simulation_dataframe(sim)
        st.subheader("결과 진단 대시보드")
        risk_segments = [s for s in sim.segments if s.verdict_mix.get("fail", 0) or s.verdict_mix.get("warn", 0)]
        low_segments = [s for s in sim.segments if s.low_confidence_ratio >= 0.5]
        best = max(sim.segments, key=lambda s: s.mean_intent, default=None)
        worst = min(sim.segments, key=lambda s: s.mean_intent, default=None)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("평균 가입의향", f"{df['가입의향'].mean():.1f}" if not df.empty else "-")
        c2.metric("위험 세그먼트", len(risk_segments))
        c3.metric("저신뢰 세그먼트", len(low_segments))
        c4.metric("가입의향 최고", best.segment if best else "-")
        if worst:
            st.caption(f"가입의향 최저 세그먼트: {worst.segment} ({worst.mean_intent}점)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("세그먼트:N", sort="-y"),
                    y=alt.Y("가입의향:Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "상태:N",
                        scale=alt.Scale(domain=list(RISK_COLORS.keys()), range=list(RISK_COLORS.values())),
                    ),
                    tooltip=["세그먼트", "가입의향", "가입률", "신뢰도", "상태"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)

            mix = df.melt(id_vars=["세그먼트"], value_vars=["pass", "warn", "fail"], var_name="판정", value_name="비율")
            mix_chart = (
                alt.Chart(mix)
                .mark_bar()
                .encode(
                    x=alt.X("세그먼트:N"),
                    y=alt.Y("비율:Q", stack="normalize"),
                    color=alt.Color("판정:N", scale=alt.Scale(domain=["pass", "warn", "fail"], range=["#2f6f4e", "#b7791f", "#b91c1c"])),
                    tooltip=["세그먼트", "판정", "비율"],
                )
                .properties(height=260)
            )
            st.altair_chart(mix_chart, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**주요 위험요인 Top 5**")
            risks = top_items([r for s in sim.segments for r in s.top_risks])
            if risks:
                for item, count in risks:
                    st.write(f"- {item} ({count}회)")
            else:
                st.caption("집계된 위험요인이 없습니다.")
        with right:
            st.markdown("**주요 개선권고 Top 5**")
            recs = top_items([r for s in sim.segments for r in s.top_recommendations])
            if recs:
                for item, count in recs:
                    st.write(f"- {item} ({count}회)")
            else:
                st.caption("집계된 개선권고가 없습니다.")

        st.divider()
        st.subheader("케이스 상세 및 근거 트레이스")
        seg_pick = st.selectbox("세그먼트", [s.segment for s in sim.segments], key="case_segment")
        seg = next(s for s in sim.segments if s.segment == seg_pick)
        persona_lookup = {p.persona_id: p for p in ensure_personas(run.persona_source or persona_source, run.settings.get("persona_limit", persona_limit))}
        for case in seg.cases:
            with st.expander(
                f"{case.persona_id} · {case.modal_suitability.upper()} · 의향 {case.intent_mean} · 신뢰도 {case.confidence:.2f}"
            ):
                persona = persona_lookup.get(case.persona_id)
                c1, c2 = st.columns(2)
                if persona:
                    c1.markdown("**페르소나 요약**")
                    c1.write(
                        f"{persona.age}세 {persona.sex}, {persona.region}, {persona.occupation}, "
                        f"가구원 {persona.household_size}명"
                    )
                    if persona.finance:
                        c1.caption(persona.finance.summary())
                c2.markdown("**판정 안정성**")
                c2.write(
                    f"라벨 분포: {case.label_counts}  \n"
                    f"라벨 합의도: {case.label_agreement:.0%}  \n"
                    f"의향 범위: {case.intent_min}~{case.intent_max} (표준편차 {case.intent_std})  \n"
                    f"무근거 발화: {case.ungrounded_turns}건"
                )
                st.markdown("**근거**")
                st.write(case.evidence or "근거 없음")
                st.markdown("**위험요인**")
                st.write(case.risks or "위험요인 없음")
                st.markdown("**개선권고**")
                st.write(case.recommendations or "개선권고 없음")
                st.caption("인용 문서 ID: " + (", ".join(case.grounding_doc_ids) or "없음"))
                st.markdown("**요약 로그 JSON**")
                st.json(case.model_dump(mode="json"))

with tab_scenario:
    run = selected_run(store)
    if run is None or run.status != "완료":
        st.info("완료된 검증 실행을 먼저 선택하세요.")
    else:
        active_product = store.get_product_version(run.product_version_id).product
        st.subheader("시나리오 랩")
        st.caption("기준안과 금리·조건 변형안의 가입의향, 가입률, fail 비율, 신뢰도 변화를 세그먼트별로 비교합니다.")
        sensitivity_path = run.sensitivity_path
        rows = load_sensitivity_rows(sensitivity_path) if sensitivity_path else []
        if not rows:
            st.info("저장된 민감도 분석 결과가 없습니다.")
            if st.button("기본 시나리오 실행"):
                personas = ensure_personas(run.persona_source or persona_source, run.settings.get("persona_limit", persona_limit))
                sim = store.load_simulation_report(run.id)
                segment_names = run.settings.get("segment_names") or [s.segment for s in sim.segments]
                segments = [
                    record.segment
                    for name in segment_names
                    if (record := store.get_segment_by_name(name)) is not None
                ]
                with st.spinner("기본 시나리오를 실행하는 중입니다."):
                    rows = sensitivity_analysis(
                        active_product.model_copy(update={"target_segments": segment_names}),
                        default_variants(active_product),
                        segments=segments,
                        k_personas=max(1, run.personas_per_segment // 2),
                        n_seeds=max(1, run.n_seeds - 1),
                        mode=run.mode,  # type: ignore[arg-type]
                        workers=run.workers,
                        personas=personas,
                        persona_source=run.persona_source,  # type: ignore[arg-type]
                    )
                    path = OUTPUT_DIR / f"sensitivity_{safe_slug(active_product.product_id)}_run_{run.id}.json"
                    store.save_sensitivity_artifact(run.id, rows, path=path)
                st.success(f"시나리오 결과를 저장했습니다: {path}")
                st.rerun()
        if rows:
            raw = [r.model_dump(mode="json") for r in rows]
            delta = scenario_delta_frame(raw)
            show = delta[
                [
                    "label",
                    "segment",
                    "adoption_rate",
                    "mean_intent",
                    "fail_ratio",
                    "mean_confidence",
                    "가입의향 변화",
                    "가입률 변화",
                    "fail 비율 변화",
                    "신뢰도 변화",
                    "주의 플래그",
                ]
            ].rename(
                columns={
                    "label": "시나리오",
                    "segment": "세그먼트",
                    "adoption_rate": "가입률",
                    "mean_intent": "가입의향",
                    "fail_ratio": "fail 비율",
                    "mean_confidence": "신뢰도",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)
            chart_df = delta[delta["label"] != "기준안"].copy()
            if not chart_df.empty:
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", title="시나리오"),
                        y=alt.Y("가입의향 변화:Q", title="기준안 대비 가입의향 변화"),
                        color=alt.Color("segment:N", title="세그먼트"),
                        column=alt.Column("segment:N", title=None),
                        tooltip=["label", "segment", "가입의향 변화", "가입률 변화", "fail 비율 변화", "신뢰도 변화"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart, use_container_width=True)
            flagged = show[show["주의 플래그"] != ""]
            if not flagged.empty:
                st.warning("가입의향은 올랐지만 판매원칙 위험도 증가한 변형안이 있습니다.")
                st.dataframe(flagged, use_container_width=True, hide_index=True)

with tab_report:
    run = selected_run(store)
    if run is None or run.status != "완료":
        st.info("완료된 검증 실행을 먼저 선택하세요.")
    else:
        product = store.get_product_version(run.product_version_id).product
        sim = store.load_simulation_report(run.id)
        sensitivity_rows = load_sensitivity_rows(run.sensitivity_path) if run.sensitivity_path else None
        holding = compare_holding_rates(sim, product.category)
        ablation = (
            AblationReport.load(OUTPUT_DIR / "ablation.json")
            if (OUTPUT_DIR / "ablation.json").exists()
            else None
        )
        text = build_report(product, sim, ablation=ablation, holding=holding, sensitivity=sensitivity_rows)
        st.subheader("보고서 및 승인 패키지")
        st.caption("Markdown 보고서는 모델/데이터/실행 설정과 한계 고지를 포함합니다. PDF/PPT는 프로토타입 필수 범위에서 제외되어 있습니다.")
        c1, c2, c3 = st.columns(3)
        c1.metric("실행", f"#{run.id}")
        c2.metric("세그먼트", len(sim.segments))
        c3.metric("인용 문서", len({d for s in sim.segments for c in s.cases for d in c.grounding_doc_ids}))
        with st.expander("Markdown 미리보기", expanded=True):
            st.markdown(text)
        if st.button("Markdown 보고서 저장", type="primary"):
            path = save_report(text, product.product_id, OUTPUT_DIR / f"report_{safe_slug(product.product_id)}_run_{run.id}.md")
            store.save_report_artifact(run.id, path=path, metadata={"product_id": product.product_id})
            st.success(f"보고서를 저장했습니다: {path}")
            st.rerun()
        st.download_button(
            "Markdown 다운로드",
            text,
            file_name=f"report_{safe_slug(product.product_id)}_run_{run.id}.md",
            mime="text/markdown",
        )
        if run.report_path:
            st.caption(f"최근 저장된 보고서: {run.report_path}")
