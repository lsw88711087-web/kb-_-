"""로컬 확인용 대시보드 — 그리기만 한다.

계산은 전부 `fdm.viewmodel`에 있다. 이 파일에는 판단이 들어가지 않는다:
무엇을 먼저 보여줄지(계층 순서), 무엇이 조치 필요인지(T1+T2), 색을 어떻게 매길지
(정규화 비율)는 모두 뷰모델이 정하고 여기서는 받아서 렌더링만 한다.

`ui/app.py`(워크벤치)와 별개다. 이 개발 PC는 Windows 앱 제어 정책이
`pandas/_libs/json*.pyd`를 차단해 `import pandas`가 실패하고, 따라서
st.dataframe / st.table / st.vega_lite_chart / altair 를 쓸 수 없다.
여기서는 마크다운 표와 st.metric 만으로 그린다.

실행: uv run --extra ui streamlit run ui/local_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdm.config import OUTPUT_DIR, SETTINGS  # noqa: E402
from fdm.eval.simulate import SimulationReport  # noqa: E402
from fdm.products.schema import load_all_products, load_product  # noqa: E402
from fdm.report import build_report, save_report  # noqa: E402
from fdm.viewmodel import (  # noqa: E402
    build_view,
    case_concern_table,
    segment_table,
    tier_legend_table,
)

st.set_page_config(page_title="상품 사전검증 — 로컬", layout="wide")


@st.cache_data(show_spinner=False)
def load_sims() -> dict[str, dict]:
    out = {}
    for p in sorted(OUTPUT_DIR.glob("sim_*.json"), reverse=True):
        try:
            out[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def heat_html(view) -> str:
    """뷰모델이 준 0~1 비율을 색으로만 바꾼다. 정규화는 뷰모델이 이미 했다."""
    h = view.heat
    out = ["<table style='border-collapse:collapse'><tr><th style='padding:6px 10px'></th>"]
    out += [f"<th style='padding:6px 10px'>{c}</th>" for c in h.columns]
    out.append("</tr>")
    for name, row in zip(h.rows, h.cells):
        out.append(f"<tr><td style='padding:6px 10px'><b>{name}</b></td>")
        for cell in row:
            r = int(255 * min(1.0, 2 * (1 - cell.ratio)))
            g = int(255 * min(1.0, 2 * cell.ratio))
            out.append(
                f"<td style='background:rgb({r},{g},120);color:#111;"
                f"text-align:center;padding:6px 10px'>{cell.text}</td>"
            )
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# ----------------------------------------------------------------- 사이드바
with st.sidebar:
    st.subheader("실행 환경")
    st.write(f"backend `{SETTINGS.backend}`")
    st.write(f"토론 `{SETTINGS.model_small}`")
    st.write(f"심판 `{SETTINGS.model_judge}`")
    st.divider()

    st.subheader("새 시뮬레이션")
    products = {p.stem: p.stem for p in sorted((ROOT / "data" / "products").glob("*.json"))}
    sel = st.selectbox("상품", list(products))
    mode = st.radio("모드", ["ensemble", "single", "debate"], horizontal=True)
    st.caption(
        "**ensemble에서만 교차확인이 성립**해 T1(즉시 조치)이 나옵니다. "
        "대신 호출이 6배입니다. 라벨은 단발에서 취하므로 판정 정확도는 single과 같습니다."
    )
    n_seeds = st.slider("멀티시드", 1, 5, 1)
    k_personas = st.slider("세그먼트별 페르소나", 1, 5, 1)

    if st.button("실행", type="primary"):
        from fdm.eval.simulate import simulate_product

        with st.spinner("LLM 호출 중…"):
            new_sim = simulate_product(
                load_product(sel), n_seeds=n_seeds, k_personas=k_personas, mode=mode
            )
            path = new_sim.save()
        st.cache_data.clear()
        st.success(f"완료 · {path.name}")

sims = load_sims()
if not sims:
    st.info("결과가 없습니다. 사이드바에서 시뮬레이션을 실행하세요.")
    st.stop()

picked = st.selectbox("결과 파일", list(sims))
sim = SimulationReport(**sims[picked])
view = build_view(sim)

# --------------------------------------------------------------------- 헤더
st.title(view.product_name)
st.caption(view.subtitle)

for col, tier in zip(st.columns(len(view.tier_summary)), view.tier_summary):
    col.metric(f"{tier.mark} {tier.label}", tier.count)

if view.mode_warning:
    st.warning(view.mode_warning)

tab_tier, tab_seg, tab_case, tab_report = st.tabs(
    ["우려 계층", "세그먼트", "케이스 상세", "리포트"]
)

# ---------------------------------------------------------------- 우려 계층
with tab_tier:
    st.caption(
        "교차확인(단발·디베이트가 모두 제기) × 심각도를 곱해 읽을 순서를 만듭니다. "
        "**지우지 않고 정렬합니다** — 놓치면 책임이고, 많으면 무시되기 때문입니다."
    )
    st.markdown(tier_legend_table(view))
    st.caption(view.caveat)
    st.divider()

    for seg in view.segments:
        st.markdown(
            f"### {seg.name}  <small>{seg.counts_badge}</small>", unsafe_allow_html=True
        )
        if not seg.has_concerns:
            st.caption("구조화된 우려 없음")
            continue
        for group in seg.tiers:
            if not group.items:
                continue
            with st.expander(group.heading, expanded=group.expanded):
                for c in group.items:
                    st.markdown(f"**{c.type_label}** `{c.badge}`  \n{c.statement}")
                    if c.anchor:
                        st.caption(f"근거 · {c.anchor}")
                    if c.verify_with:
                        st.caption(f"확인방법 · {c.verify_with}")
                    st.divider()

# ----------------------------------------------------------------- 세그먼트
with tab_seg:
    st.subheader("세그먼트 요약")
    st.markdown(segment_table(view.segments))
    st.caption("조치필요 = T1+T2 건수")
    st.subheader("히트맵")
    st.caption("지표마다 스케일이 달라 지표별로 정규화했습니다. fail비율은 낮을수록 좋습니다.")
    st.markdown(heat_html(view), unsafe_allow_html=True)

# -------------------------------------------------------------- 케이스 상세
with tab_case:
    seg = next(
        s for s in view.segments
        if s.name == st.selectbox("세그먼트", [s.name for s in view.segments])
    )
    for case in seg.cases:
        with st.expander(case.heading):
            a, b = st.columns(2)
            a.markdown(
                f"**라벨 분포** {case.label_counts}  \n"
                f"**합의도** {case.label_agreement:.0%}  \n"
                f"**의향 범위** {case.intent_range}"
            )
            b.markdown(
                f"**신뢰도** {case.confidence:.2f} ({case.confidence_level})  \n"
                f"**심판 자기신뢰** {case.judge_self_confidence:.2f}  \n"
                f"**무근거 발화** {case.ungrounded_turns}건"
            )
            if case.concerns:
                st.markdown("**우려 (계층순)**")
                st.markdown(case_concern_table(case))
                st.caption(
                    "재현 = 여러 시드 중 이 우려가 나온 비율. "
                    "**재현성이지 정확성이 아닙니다** (실측: 저신뢰 오답률 40% vs 전체 42%)."
                )
            st.markdown("**근거**")
            for e in case.evidence or ["—"]:
                st.markdown(f"- {e}")
            if case.recommendations:
                st.markdown("**개선권고**")
                for r in case.recommendations:
                    st.markdown(f"- {r}")
            st.caption("인용 문서 · " + (", ".join(case.grounding_doc_ids) or "없음"))

# -------------------------------------------------------------------- 리포트
with tab_report:
    product = next(
        (p for p in load_all_products() if p.product_id == view.product_id), None
    )
    if product is None:
        st.info("상품 정의를 찾을 수 없습니다(민감도 변형 상품일 수 있음).")
    else:
        text = build_report(product, sim)
        if st.button("outputs/ 에 저장"):
            st.success(f"저장 · {save_report(text, product.product_id)}")
        st.markdown(text)
