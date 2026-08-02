"""전역 설정. 환경변수(.env) 기반."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv은 선택 의존성처럼 다룬다
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PERSONA_DIR = DATA_DIR / "personas"
PRODUCT_DIR = DATA_DIR / "products"
RAG_DIR = DATA_DIR / "rag"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUT_DIR = Path(os.environ.get("FDM_OUTPUT_DIR", str(ROOT / "outputs")))


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _env_any(keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        v = os.environ.get(key)
        if v not in (None, ""):
            return v
    return default


def _default_model(role: str) -> str:
    backend = os.environ.get("FDM_BACKEND", "mock")
    if backend == "gemini":
        return "gemini-3.6-flash"
    if backend == "openai":
        return "gpt-4.1-mini" if role == "small" else "gpt-4.1"
    return "qwen3:8b"


def _default_api_key() -> str:
    backend = os.environ.get("FDM_BACKEND", "mock")
    if backend == "gemini":
        return _env_any(("FDM_GEMINI_API_KEY", "GEMINI_API_KEY", "FDM_LLM_API_KEY"), "")
    if backend == "openai":
        return _env_any(("FDM_OPENAI_API_KEY", "OPENAI_API_KEY", "FDM_LLM_API_KEY"), "")
    if backend == "vllm":
        return _env_any(("FDM_VLLM_API_KEY", "FDM_LLM_API_KEY"), "")
    return _env_any(
        (
            "FDM_LLM_API_KEY",
            "FDM_VLLM_API_KEY",
            "FDM_OPENAI_API_KEY",
            "FDM_GEMINI_API_KEY",
            "GEMINI_API_KEY",
        ),
        "",
    )


@dataclass
class Settings:
    backend: str = field(default_factory=lambda: _env("FDM_BACKEND", "mock"))
    ollama_base_url: str = field(
        default_factory=lambda: _env("FDM_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    vllm_base_url: str = field(
        default_factory=lambda: _env(
            "FDM_VLLM_BASE_URL",
            _env("FDM_LLM_BASE_URL", "http://localhost:8000/v1"),
        )
    )
    openai_base_url: str = field(
        default_factory=lambda: _env("FDM_OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    gemini_base_url: str = field(
        default_factory=lambda: _env(
            "FDM_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
    )
    llm_api_key: str = field(default_factory=_default_api_key)
    model_small: str = field(default_factory=lambda: _env("FDM_MODEL_SMALL", _default_model("small")))
    model_judge: str = field(default_factory=lambda: _env("FDM_MODEL_JUDGE", _default_model("judge")))
    timeout: float = field(default_factory=lambda: float(_env("FDM_TIMEOUT", "180")))
    max_tokens: int = field(default_factory=lambda: int(_env("FDM_MAX_TOKENS", "1200")))
    gemini_reasoning_effort: str = field(
        default_factory=lambda: _env("FDM_GEMINI_REASONING_EFFORT", "low")
    )
    # Qwen3·EXAONE 등 하이브리드 추론 모델의 사고 모드. 켜면 2~3배 느리고
    # 토큰 예산을 사고가 다 써서 본문이 잘리는 일이 생긴다. 기본은 끔.
    think: bool = field(default_factory=lambda: _env("FDM_THINK", "0") not in ("0", "false", "False"))
    keep_alive: str = field(default_factory=lambda: _env("FDM_KEEP_ALIVE", "30m"))
    # Ollama 기본 컨텍스트(4096)는 사실팩+근거+디베이트 전문을 담기에 부족할 수 있다.
    # 6144는 8GB VRAM에서도 부담을 크게 늘리지 않으면서 잘림을 줄이는 보수적인 기본값이다.
    num_ctx: int = field(default_factory=lambda: int(_env("FDM_NUM_CTX", "6144")))
    # 판매 정황이 "설명했다"고 명시한 항목의 '설명 부족' 주장을 기각한다 (situation.py).
    # 사후 필터라 **라벨을 바꾸지 않는다** — 심판이 판정한 뒤 우려 목록만 손댄다.
    #
    # 사전 손익(rate_A에 얹어본 dry-run): ensemble 오탐 −10 / 정답 손실 0,
    # 깨끗한 건당 오탐 2.58 → 1.75, 우려 recall 96.7% 불변.
    #
    # 기본은 끔. `outputs/rate_A.json` 및 기존 측정치와 나란히 비교하려면 꺼진 값이
    # 기준선이어야 한다. 켜고 한 번 재측정한 뒤 기본값 전환을 판단할 것.
    screen_situation: bool = field(
        default_factory=lambda: _env("FDM_SCREEN_SITUATION", "0") not in ("0", "false", "False")
    )
    # 토론 온도는 성능 측정 중 과잉 변동이 관측되어 기본값을 낮췄다.
    temp_debater: float = field(default_factory=lambda: float(_env("FDM_TEMP_DEBATER", "0.5")))
    temp_judge: float = field(default_factory=lambda: float(_env("FDM_TEMP_JUDGE", "0.2")))

    @property
    def base_url(self) -> str:
        if self.backend == "gemini":
            return self.gemini_base_url
        if self.backend == "openai":
            return self.openai_base_url
        return self.vllm_base_url if self.backend == "vllm" else self.ollama_base_url


SETTINGS = Settings()

# 디베이트 역할 → 모델 배치 (비대칭 배치 전략)
ROLE_MODEL = {
    "advocate": "small",
    "skeptic": "small",
    "persona": "small",
    "judge": "judge",
    "single": "judge",  # 애블레이션의 단발 질문(디베이트 없음)도 심판 모델로 공정 비교
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
