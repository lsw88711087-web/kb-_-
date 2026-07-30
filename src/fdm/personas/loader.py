"""Nemotron-Personas-Korea 로더 + 세그먼트 필터.

우선순위:
  1. data/personas/*.jsonl  (로컬 캐시 / 오프라인)
  2. Hugging Face `nvidia/Nemotron-Personas-Korea` (datasets 설치 + 네트워크 필요)
  3. 합성 폴백 (KOSIS 연령·지역 분포로 생성) — 네트워크 없이 파이프라인 확인용
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from ..config import PERSONA_DIR
from .finance import attach_finance, load_kosis_params
from .schema import Persona, Segment

HF_DATASET = "nvidia/Nemotron-Personas-Korea"
PersonaSource = Literal["auto", "jsonl", "hf", "synthetic"]
SYNTHETIC_SOURCE = "synthetic-fallback"

# Nemotron 컬럼명 → Persona 필드 (데이터셋 버전차를 흡수하기 위해 후보 리스트로 둔다)
FIELD_CANDIDATES = {
    "age": ["age"],
    "sex": ["sex", "gender"],
    "region": ["region", "province", "administrative_division", "city"],
    "occupation": ["occupation", "professional_role", "job"],
    "education": ["education_level", "education"],
    "marital_status": ["marital_status"],
    "household_size": ["household_size"],
}
TEXT_CANDIDATES = ["persona", "professional_persona", "personality", "description"]
TRAIT_CANDIDATES = ["skills_and_expertise", "hobbies_and_interests", "career_goals_and_ambitions"]

# Nemotron-Personas-Korea는 성별을 '남자'/'여자'로 표기한다. 영문 표기도 함께 받는다.
MALE_TOKENS = {"male", "m", "남", "남성", "남자"}
FEMALE_TOKENS = {"female", "f", "여", "여성", "여자"}

# 이 데이터셋에는 가구원 수 컬럼이 없고 family_type(가구 형태)만 있다.
# 형태별 대표 가구원 수로 환산한다. 부분 문자열 매칭이므로 순서가 우선순위다.
FAMILY_TYPE_SIZE = [
    ("배우자·자녀와 거주", 4),
    ("배우자와 자녀", 4),
    ("자녀와 거주", 2),  # 한부모
    ("배우자와 거주", 2),
    ("부모와 동거", 3),
    ("어머니와 동거", 2),
    ("아버지와 동거", 2),
    ("혼자 거주", 1),
    ("3세대", 5),
    ("기타2세대", 3),
]


def household_size_from(row: dict) -> int | None:
    """family_type → 대표 가구원 수. 매칭 실패 시 None."""
    ft = str(row.get("family_type") or "")
    if not ft:
        return None
    for kw, size in FAMILY_TYPE_SIZE:
        if kw in ft:
            return size
    return None


def _first(row: dict, keys: list[str]) -> object | None:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", []):
            return v
    return None


def _to_int(v: object, default: int) -> int:
    try:
        return int(float(str(v).strip().rstrip("세")))
    except (TypeError, ValueError):
        return default


def row_to_persona(row: dict, idx: int, *, source: str | None = None) -> Persona | None:
    age = _to_int(_first(row, FIELD_CANDIDATES["age"]), -1)
    if age < 19 or age > 95:  # 성인 대상 상품 검증이므로 미성년 제외
        return None
    sex_raw = str(_first(row, FIELD_CANDIDATES["sex"]) or "").strip().lower()
    sex = "남성" if sex_raw in MALE_TOKENS else ("여성" if sex_raw in FEMALE_TOKENS else "미상")
    traits: list[str] = []
    for key in TRAIT_CANDIDATES:
        v = row.get(key)
        if isinstance(v, str) and v:
            traits.extend([t.strip() for t in v.split(",")[:3] if t.strip()])
        elif isinstance(v, list):
            traits.extend([str(t) for t in v[:3]])

    return Persona(
        persona_id=str(row.get("uuid") or row.get("id") or f"nemo-{idx:06d}"),
        age=age,
        sex=sex,
        region=str(_first(row, FIELD_CANDIDATES["region"]) or "미상"),
        occupation=str(_first(row, FIELD_CANDIDATES["occupation"]) or "미상"),
        education=str(_first(row, FIELD_CANDIDATES["education"]) or "미상"),
        marital_status=str(_first(row, FIELD_CANDIDATES["marital_status"]) or "미상"),
        household_size=_to_int(
            _first(row, FIELD_CANDIDATES["household_size"]),
            household_size_from(row) or 1,
        ),
        persona_text=str(_first(row, TEXT_CANDIDATES) or ""),
        traits=traits[:6],
        source=source or str(row.get("source") or HF_DATASET),
    )


# ------------------------------------------------------------------- 소스별 로드
def load_from_jsonl(path: Path) -> list[Persona]:
    out: list[Persona] = []
    cache_source = f"{HF_DATASET} (local-cache:{path.name})"
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "persona_id" in row:
                p = Persona(**row)
                if p.source.startswith(HF_DATASET) and "local-cache:" not in p.source:
                    p = p.model_copy(update={"source": f"{p.source}; local-cache:{path.name}"})
            else:
                p = row_to_persona(row, i, source=cache_source)
            if p:
                out.append(p)
    return out


def load_from_hf(limit: int = 5000) -> list[Persona]:
    from datasets import load_dataset  # 선택 의존성

    ds = load_dataset(HF_DATASET, split=f"train[:{limit}]")
    out = []
    for i, row in enumerate(ds):
        p = row_to_persona(dict(row), i, source=f"{HF_DATASET} (huggingface)")
        if p:
            out.append(p)
    return out


def synthesize(n: int = 400, seed: int = 42) -> list[Persona]:
    """네트워크 없이 쓰는 폴백. 연령·지역 분포만 KOSIS 공표치에 맞춘다."""
    params = load_kosis_params()
    rng = random.Random(seed)
    bands = list(params["age_band_share_adult"].items())
    regions = list(params["region_population_share"].items())
    occupations = [
        "사무직 회사원", "생산직 근로자", "서비스직 종사자", "판매직 종사자", "자영업자",
        "프리랜서 디자이너", "소프트웨어 개발자", "간호사", "초등학교 교사", "공무원",
        "중소기업 관리자", "농업 종사자", "무직", "대학생", "전업주부", "아르바이트",
    ]
    edu = ["고졸", "전문대졸", "대졸", "대학원졸"]
    marital = ["미혼", "기혼", "이혼", "사별"]
    band_range = {"20대": (20, 29), "30대": (30, 39), "40대": (40, 49), "50대": (50, 59), "60대이상": (60, 82)}

    def pick(pairs):
        r, acc = rng.random(), 0.0
        total = sum(w for _, w in pairs)
        for k, w in pairs:
            acc += w / total
            if r <= acc:
                return k
        return pairs[-1][0]

    out = []
    for i in range(n):
        band = pick(bands)
        lo, hi = band_range[band]
        age = rng.randint(lo, hi)
        occ = rng.choice(occupations)
        if age >= 65 and rng.random() < 0.6:
            occ = "무직"
        p = Persona(
            persona_id=f"synth-{i:05d}",
            age=age,
            sex=rng.choice(["남성", "여성"]),
            region=pick(regions),
            occupation=occ,
            education=rng.choice(edu),
            marital_status=rng.choice(marital) if age >= 30 else "미혼",
            household_size=rng.choice([1, 1, 2, 2, 3, 3, 4]),
            persona_text=f"{age}세 {occ}. 금융상품 선택 시 {rng.choice(['안정성', '수익률', '편의성', '수수료'])}을 가장 중시한다.",
            traits=[rng.choice(["보수적 투자성향", "적극적 투자성향", "디지털 친화", "대면 채널 선호", "가격 민감"])],
            source=f"{SYNTHETIC_SOURCE} (Nemotron 미설치)",
        )
        out.append(p)
    return out


def is_nemotron_persona(p: Persona) -> bool:
    """공개 Nemotron-Personas-Korea 레코드에서 온 페르소나인지 확인."""
    return p.source.startswith(HF_DATASET)


def persona_source_counts(personas: Iterable[Persona]) -> dict[str, int]:
    """리포트/doctor에서 쓸 출처별 건수."""
    return dict(Counter(p.source for p in personas))


def require_nemotron_personas(personas: Iterable[Persona]) -> None:
    """합성 폴백이 섞였으면 명시적으로 실패시킨다."""
    items = list(personas)
    if not items:
        raise RuntimeError("페르소나가 0명이다. 데이터 로딩 또는 세그먼트 조건을 확인해야 한다.")
    bad = [p.source for p in items if not is_nemotron_persona(p)]
    if bad:
        sample = ", ".join(sorted(set(bad))[:3])
        raise RuntimeError(
            "Nemotron-Personas-Korea 데이터가 아닌 페르소나가 포함되어 있다. "
            f"출처 예시: {sample}. Colab에서는 `python scripts/fetch_personas.py --limit 5000` "
            "또는 `--persona-source hf --require-real-personas`로 다시 실행하라."
        )


def load_personas(
    *,
    source: PersonaSource = "auto",
    limit: int = 2000,
    with_finance: bool = True,
    allow_synthetic_fallback: bool | None = None,
) -> list[Persona]:
    if allow_synthetic_fallback is None:
        allow_synthetic_fallback = source == "auto"

    personas: list[Persona] = []
    files = sorted(PERSONA_DIR.glob("*.jsonl")) if PERSONA_DIR.exists() else []
    failures: list[str] = []

    if source == "synthetic":
        personas = synthesize(n=min(limit, 400))
    elif source in {"auto", "jsonl"} and files:
        for f in files:
            personas.extend(load_from_jsonl(f))
    elif source == "jsonl" and not allow_synthetic_fallback:
        raise RuntimeError(
            f"{PERSONA_DIR} 아래에 persona jsonl 캐시가 없다. "
            "`python scripts/fetch_personas.py --limit 5000`로 먼저 저장하라."
        )

    if not personas and source in {"auto", "hf"}:
        try:
            personas = load_from_hf(limit=limit)
        except Exception as e:
            failures.append(f"Hugging Face 로딩 실패: {e}")
            if source == "hf" and not allow_synthetic_fallback:
                raise RuntimeError(
                    "Nemotron-Personas-Korea를 Hugging Face에서 불러오지 못했다. "
                    "`uv sync --extra personas` 또는 Colab의 `pip install datasets pyarrow`와 "
                    "네트워크 권한을 확인하라."
                ) from e
    if not personas:
        if not allow_synthetic_fallback:
            detail = " / ".join(failures) if failures else "사용 가능한 페르소나 소스가 없다."
            raise RuntimeError(detail)
        personas = synthesize(n=min(limit, 400))

    personas = personas[:limit]
    if with_finance:
        personas = [attach_finance(p) for p in personas]
    return personas


def filter_segment(personas: Iterable[Persona], segment: Segment) -> list[Persona]:
    return [p for p in personas if segment.matches(p)]


def sample_cohort(
    personas: Iterable[Persona], segment: Segment, k: int, seed: int = 0
) -> list[Persona]:
    """세그먼트 내에서 k명 코호트를 재현 가능하게 추출."""
    pool = filter_segment(personas, segment)
    if len(pool) <= k:
        return pool
    rng = random.Random(f"{segment.name}:{seed}")
    return rng.sample(pool, k)
