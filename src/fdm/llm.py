"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ROLE_MODEL, SETTINGS, Settings


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] | None = None


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or SETTINGS
        self.last_json_repaired = False
        self.json_repair_count = 0

    def _headers(self) -> dict[str, str]:
        if self.s.llm_api_key:
            return {"Authorization": f"Bearer {self.s.llm_api_key}"}
        return {}

    def model_for(self, role: str) -> str:
        tier = ROLE_MODEL.get(role, "small")
        return self.s.model_judge if tier == "judge" else self.s.model_small

    def chat(
        self,
        *,
        role: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        seed: int | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = self.model_for(role)
        if self.s.backend == "mock":
            return LLMResponse(
                text=_mock_reply(role, system, user, seed, temperature),
                model=f"mock::{model}",
            )
        if self.s.backend == "ollama":
            return self._chat_ollama_native(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                seed=seed,
                json_mode=json_mode,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.s.max_tokens,
            "stream": False,
        }
        if seed is not None and self.s.backend != "gemini":
            payload["seed"] = seed
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": json_schema},
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.s.backend == "gemini" and self.s.gemini_reasoning_effort:
            payload["reasoning_effort"] = self.s.gemini_reasoning_effort
        if self.s.backend == "vllm" and not self.s.think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.s.base_url.rstrip('/')}/chat/completions"
        try:
            r = httpx.post(url, json=payload, headers=self._headers(), timeout=self.s.timeout)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            optional_keys = ("response_format", "chat_template_kwargs", "reasoning_effort")
            if any(k in payload for k in optional_keys):
                for key in optional_keys:
                    payload.pop(key, None)
                try:
                    r = httpx.post(url, json=payload, headers=self._headers(), timeout=self.s.timeout)
                    r.raise_for_status()
                except httpx.HTTPError:
                    raise LLMError(f"{url} 호출 실패: {e} / {e.response.text[:300]}") from e
            else:
                raise LLMError(f"{url} 호출 실패: {e} / {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise LLMError(
                f"{url} 연결 실패: {e}. 배포 환경에서는 OpenAI-compatible LLM 서버 URL과 "
                "API 키(FDM_LLM_BASE_URL/FDM_LLM_API_KEY, FDM_OPENAI_* 또는 FDM_GEMINI_*)를 확인하세요. "
                "로컬에서는 ollama serve / vllm serve 또는 FDM_BACKEND=mock 사용."
            ) from e

        data = r.json()
        text = _extract_chat_text(data)
        return LLMResponse(text=_strip_think(text), model=model, raw=data)

    def _chat_ollama_native(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        seed: int | None,
        json_mode: bool,
        max_tokens: int | None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        base = self.s.ollama_base_url.rstrip("/").removesuffix("/v1")
        url = f"{base}/api/chat"
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens or self.s.max_tokens,
            "num_ctx": self.s.num_ctx,
        }
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.s.think,
            "options": options,
            "keep_alive": self.s.keep_alive,
        }
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"

        try:
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            payload.pop("think", None)
            try:
                r = httpx.post(url, json=payload, timeout=self.s.timeout)
                r.raise_for_status()
            except httpx.HTTPError:
                raise LLMError(f"{url} 호출 실패: {e} / {e.response.text[:300]}") from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"{url} 응답 시간 초과({self.s.timeout:.0f}초): {e}\n"
                f"  FDM_NUM_CTX={self.s.num_ctx}, FDM_MAX_TOKENS={self.s.max_tokens}, "
                "FDM_TIMEOUT 및 GPU 동시 작업을 확인하세요."
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(
                f"{url} 연결 실패: {e}. `ollama serve`가 떠 있는지 확인하세요. "
                "또는 FDM_BACKEND=mock 사용."
            ) from e

        data = r.json()
        msg = data.get("message", {})
        text = _strip_think(msg.get("content") or "")
        if not text and msg.get("thinking"):
            payload["think"] = False
            r = httpx.post(url, json=payload, timeout=self.s.timeout)
            r.raise_for_status()
            data = r.json()
            text = _strip_think(data.get("message", {}).get("content") or "")
        return LLMResponse(text=text, model=model, raw=data)

    def chat_json(
        self,
        *,
        required_keys: tuple[tuple[str, ...], ...] = (),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Parse a JSON object and retry once when required keys are missing."""
        kwargs.setdefault("json_mode", True)
        self.last_json_repaired = False

        res = self.chat(**kwargs)
        obj = extract_json(res.text)
        missing = _missing_key_groups(obj, required_keys)
        if obj is not None and not missing:
            return obj

        problem = (
            "유효한 JSON 객체가 아니다"
            if obj is None
            else f"필수 키가 누락됐다: {['|'.join(g) for g in missing]}"
        )
        repair = dict(kwargs)
        repair["temperature"] = 0.0
        repair["user"] = (
            f"직전 응답에 문제가 있다: {problem}\n"
            "아래 텍스트의 의미를 유지하되 요구된 키를 모두 가진 JSON 객체 하나만 출력하라. "
            "설명과 코드펜스는 금지한다.\n"
            f"필수 키: {[g[0] for g in required_keys] or '(스키마 참조)'}\n\n"
            + res.text[:4000]
        )
        res2 = self.chat(**repair)
        obj2 = extract_json(res2.text)
        missing2 = _missing_key_groups(obj2, required_keys)
        if obj2 is not None and not missing2:
            self.last_json_repaired = True
            self.json_repair_count += 1
            return obj2

        raise LLMError(
            f"JSON 스키마 검증 실패 (2회 시도). {problem}\n"
            f"1차 응답: {res.text[:200]}\n2차 응답: {res2.text[:200]}"
        )


def _missing_key_groups(
    obj: dict[str, Any] | None, required_keys: tuple[tuple[str, ...], ...]
) -> list[tuple[str, ...]]:
    if obj is None:
        return list(required_keys)
    return [
        group
        for group in required_keys
        if not any(k in obj and obj[k] not in (None, "", [], {}) for k in group)
    ]


def _extract_chat_text(data: dict[str, Any]) -> str:
    """Return assistant text from OpenAI-compatible chat responses.

    일부 호환 서버는 차단/토큰 제한 상황에서 `choices[0].message`를 생략하거나,
    legacy 형태의 `choices[0].text`를 돌려준다. 이 차이를 KeyError로 새지 않게
    여기서 흡수하고, 본문이 없으면 run_case가 잡을 수 있는 LLMError로 바꾼다.
    """
    if not isinstance(data, dict):
        raise LLMError(f"LLM 응답 형식 오류: JSON 객체가 아님 {_json_preview(data)}")
    if data.get("error"):
        raise LLMError(f"LLM API 오류 응답: {_json_preview(data['error'])}")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        output_text = _content_to_text(data.get("output_text"))
        if output_text:
            return output_text
        raise LLMError(f"LLM 응답 형식 오류: choices가 비어 있음 {_json_preview(data)}")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMError(f"LLM 응답 형식 오류: choices[0]가 객체가 아님 {_json_preview(choice)}")

    for key in ("message", "delta"):
        message = choice.get(key)
        if isinstance(message, dict):
            text = _content_to_text(message.get("content"))
            if text:
                return text
            refusal = _content_to_text(message.get("refusal"))
            if refusal:
                raise LLMError(f"LLM 응답 거부: {refusal[:300]}")

    text = _content_to_text(choice.get("text"))
    if text:
        return text

    finish_reason = choice.get("finish_reason") or data.get("finish_reason")
    raise LLMError(
        "LLM 응답 본문 없음: choices[0].message.content를 찾지 못했습니다 "
        f"(finish_reason={finish_reason!r}). 응답 일부: {_json_preview(data)}"
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value.strip()
    return ""


def _json_preview(value: Any, limit: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = repr(value)
    return text[:limit]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_think(text: str) -> str:
    out = _THINK.sub("", text)
    if "<think>" in out and "</think>" not in out:
        out = out.split("<think>")[0]
    return out.strip()


def extract_json(text: str) -> dict[str, Any] | None:
    text = _strip_think(text)
    m = _FENCE.search(text)
    candidates = ([m.group(1)] if m else []) + [text]
    for cand in candidates:
        cand = cand.strip()
        obj = _loads_object(cand)
        if obj is not None:
            return obj
        for fixed in _jsonish_repairs(cand):
            obj = _loads_object(fixed)
            if obj is not None:
                return obj

        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = cand[start : i + 1]
                        obj = _loads_object(snippet)
                        if obj is not None:
                            return obj
                        for fixed in _jsonish_repairs(snippet):
                            obj = _loads_object(fixed)
                            if obj is not None:
                                return obj
                        break
            start = cand.find("{", start + 1)
    return None


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text, strict=False)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _jsonish_repairs(text: str) -> list[str]:
    base = text.strip()
    if not base:
        return []
    if "{" in base and "}" in base:
        base = base[base.find("{") : base.rfind("}") + 1]

    variants: list[str] = []
    lines = []
    for line in base.splitlines():
        line = re.sub(
            r'(^|[{,\[]\s*)"([^"\n{}[\]]{1,80}?):(\s*)',
            r'\1"\2":\3',
            line,
        )
        line = re.sub(r'^(\s*)"([^"\n{}[\]]{1,80}?):(\s*)', r'\1"\2":\3', line)
        line = re.sub(
            r'(^|[{,\[]\s*)([A-Za-z가-힣_][^:\n"{\[\]]{0,60}?)(\s*):',
            lambda m: f'{m.group(1)}"{m.group(2).strip()}":',
            line,
        )
        lines.append(line)
    fixed = "\n".join(lines)
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
    variants.append(fixed)

    balanced = _balance_json_brackets(fixed)
    if balanced != fixed:
        variants.append(balanced)
    return variants


def _balance_json_brackets(text: str) -> str:
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return text + "".join(reversed(stack))


def _rand01(*parts: Any) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _mock_reply(role: str, system: str, user: str, seed: int | None, temperature: float) -> str:
    key = hashlib.sha256((system + user).encode()).hexdigest()[:12]
    jitter = (_rand01(key, seed) - 0.5) * 2 * (20 * temperature)
    base = 25 + _rand01(key) * 60
    score = max(0, min(100, int(round(base + jitter))))

    if role in {"judge", "single"}:
        verdict = "pass" if score >= 60 else ("warn" if score >= 40 else "fail")
        principles = ["적합성", "적정성", "설명의무", "불공정영업금지", "부당권유금지", "광고규제"]
        if verdict == "pass":
            violated: list[str] = []
        else:
            i = int(_rand01(key, "p1") * len(principles))
            violated = [principles[i % len(principles)]]
            if verdict == "fail":
                j = int(_rand01(key, "p2") * len(principles))
                candidate = principles[j % len(principles)]
                if candidate not in violated:
                    violated.append(candidate)
        return json.dumps(
            {
                "적합성": verdict,
                "가입의향점수": score,
                "위반원칙": violated,
                "근거": [
                    f"[MOCK] 금소법 제17조(적합성원칙) 대조 결과 score={score}",
                    "[MOCK] 페르소나 소득·부채 대비 납입부담 검토",
                ],
                "위험요인": ["[MOCK] 우대금리 조건 미달 가능성", "[MOCK] 중도해지 시 약정금리 미적용"],
                "우려": [
                    {
                        "유형": "preferential_unattainable",
                        "심각도": "중대" if score < 50 else "주의",
                        "내용": "[MOCK] 우대조건 달성 가능성이 낮다",
                        "앵커": f"[MOCK] 달성률 추정 {score}%",
                    },
                    {
                        "유형": "early_termination_penalty",
                        "심각도": "경미",
                        "내용": "[MOCK] 중도해지 시 우대금리 미적용",
                        "앵커": "",
                    },
                ],
                "개선권고": ["[MOCK] 우대조건 달성률 산출 근거 보완", "[MOCK] 중도해지 불이익 강조 표시"],
                "confidence": round(0.4 + _rand01(key, "conf") * 0.5, 2),
                "요약": "[MOCK] 스텁 판정입니다. 실제 LLM 백엔드로 교체하세요.",
            },
            ensure_ascii=False,
        )
    if role == "persona":
        return json.dumps(
            {
                "발화": "[MOCK] 조건은 괜찮아 보이는데 우대조건을 다 채울 수 있을지 걱정됩니다.",
                "가입의향점수": score,
                "망설임요인": ["[MOCK] 자동이체 유지 부담"],
            },
            ensure_ascii=False,
        )
    label = "옹호자" if role == "advocate" else "회의론자"
    return f"[MOCK/{label}] 근거: 상품 약관 및 페르소나 재무지표 기반 주장 (score_hint={score})"
