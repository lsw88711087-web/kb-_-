"""FDM Product Workbench용 경량 SQLite 저장소와 업무 유틸.

프로토타입 범위에서는 별도 API 서버 없이 Streamlit이 이 모듈을 직접 호출한다.
테이블은 PostgreSQL 전환을 염두에 두고 JSON blob 중심으로 단순하게 둔다.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..config import DATA_DIR, OUTPUT_DIR, SETTINGS
from ..eval.simulate import SensitivityRow, SimulationReport, load_segments
from ..personas.loader import filter_segment, is_nemotron_persona
from ..personas.schema import Persona, Segment
from ..products.schema import Product

DB_PATH = DATA_DIR / "workbench.sqlite3"
PRODUCT_ARTIFACT_DIR = OUTPUT_DIR / "products"

ProductStatus = Literal["초안", "검증 중", "보완 필요", "출시 검토 가능", "승인 완료"]
RunStatus = Literal["대기", "실행 중", "완료", "실패", "취소"]


@dataclass(frozen=True)
class ProductVersionRecord:
    id: int
    product_project_id: str
    version_number: int
    product: Product
    artifact_path: str
    created_by: str
    created_at: str
    change_note: str


@dataclass(frozen=True)
class SegmentDefinitionRecord:
    id: int
    name: str
    segment: Segment
    is_preset: bool
    created_by: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SimulationRunRecord:
    id: int
    product_version_id: int
    product_project_id: str
    product_name: str
    version_number: int
    preset: str
    mode: str
    n_seeds: int
    personas_per_segment: int
    workers: int
    persona_source: str
    status: str
    started_at: str
    finished_at: str
    artifact_path: str
    sensitivity_path: str
    report_path: str
    error_summary: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class PortfolioRow:
    project_id: str
    name: str
    category: str
    status: str
    version_number: int | None
    average_intent: float | None
    risk_segments: int
    low_confidence_segments: int
    last_run_at: str
    next_action: str


@dataclass(frozen=True)
class ProductIssue:
    severity: Literal["warning", "error"]
    message: str


PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "빠른 검증": {
        "purpose": "상품 초안의 큰 위험만 빠르게 확인",
        "mode": "single",
        "n_seeds": 1,
        "personas_per_segment": 2,
        "workers": 2,
        "include_sensitivity": False,
    },
    "표준 검증": {
        "purpose": "내부 검토용 기본 실행",
        "mode": "debate",
        "n_seeds": 3,
        "personas_per_segment": 3,
        "workers": 4,
        "include_sensitivity": False,
    },
    "심층 검증": {
        "purpose": "준법/보고서 제출 전 검증",
        "mode": "debate",
        "n_seeds": 5,
        "personas_per_segment": 5,
        "workers": 4,
        "include_sensitivity": True,
    },
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value).strip("_")
    return slug or "product"


def product_to_json(product: Product) -> str:
    return json_dumps(product.model_dump(mode="json"))


def derive_status(sim: SimulationReport) -> ProductStatus:
    risk_segments = sum(1 for s in sim.segments if s.verdict_mix.get("fail", 0) or s.verdict_mix.get("warn", 0))
    low_segments = sum(1 for s in sim.segments if s.low_confidence_ratio >= 0.5)
    if risk_segments:
        return "보완 필요"
    if low_segments:
        return "보완 필요"
    return "출시 검토 가능"


def validate_product_for_workbench(product: Product) -> list[ProductIssue]:
    """상품 설계 캔버스용 사전 경고.

    스키마 검증은 Pydantic이 담당하고, 여기서는 업무적으로 놓치기 쉬운 근거·설명 누락을 잡는다.
    """
    issues: list[ProductIssue] = []
    if not product.name.strip():
        issues.append(ProductIssue("error", "상품명이 비어 있습니다."))
    if not product.product_id.strip():
        issues.append(ProductIssue("error", "상품 ID가 비어 있습니다."))
    if product.intr_rate2 is not None and product.category in {"saving", "deposit"} and not product.preferentials:
        issues.append(ProductIssue("warning", "최고금리가 있지만 우대조건이 비어 있습니다. 최고금리 근거를 보강하세요."))
    if product.preferentials and not product.clauses:
        condition_label = {
            "loan": "금리 감면/우대 조건",
            "card": "혜택/실적 조건",
            "pension": "세제/납입 우대조건",
            "fund": "수수료/운용 조건",
        }.get(product.category, "우대조건")
        issues.append(ProductIssue("warning", f"{condition_label}이 있지만 인용 가능한 약관/설명서 조항이 없습니다."))
    if product.fees and not (product.risk_notes or product.clauses):
        issues.append(ProductIssue("warning", "수수료가 있지만 설명 문구나 근거 조항이 없습니다."))
    if not product.early_termination.strip():
        noun = {
            "loan": "중도상환",
            "card": "해지/혜택 회수",
            "pension": "중도인출/해지",
            "fund": "환매",
        }.get(product.category, "중도해지")
        issues.append(ProductIssue("warning", f"{noun} 조건이 비어 있습니다."))
    if not product.clauses:
        issues.append(ProductIssue("warning", "`clauses`가 비어 있어 디베이트 근거 품질이 낮아질 수 있습니다."))
    if not product.target_segments:
        issues.append(ProductIssue("warning", "타깃 세그먼트가 비어 있습니다. 실행 전 세그먼트를 선택하세요."))

    if product.category in {"saving", "deposit"}:
        if product.intr_rate is None:
            issues.append(ProductIssue("error", "예적금 상품은 기본금리가 필요합니다."))
        if product.save_trm_months is None:
            issues.append(ProductIssue("warning", "예적금 상품은 기간 입력을 권장합니다."))
        if product.category == "saving" and product.max_monthly_manwon is None:
            issues.append(ProductIssue("warning", "적금 상품은 월 납입 한도를 입력하세요."))
        if product.category == "deposit" and product.max_monthly_manwon is None:
            issues.append(ProductIssue("warning", "예금 상품은 가입금액 범위를 입력하세요."))
    if product.category in {"loan", "card"} and product.limit_manwon is None:
        label = "대출 한도" if product.category == "loan" else "카드 한도"
        issues.append(ProductIssue("warning", f"{label}가 비어 있습니다."))
    if product.category == "loan" and product.intr_rate2 is None:
        issues.append(ProductIssue("warning", "대출 최고금리를 입력하면 고금리 위험 진단이 더 안정적입니다."))
    if product.category == "card" and not product.preferentials:
        issues.append(ProductIssue("warning", "카드 상품은 혜택/실적 조건을 입력해야 오인 가능성을 검증하기 쉽습니다."))
    if product.category in {"pension", "fund"} and not product.risk_notes:
        issues.append(ProductIssue("warning", "투자성/장기 상품은 원금손실, 수수료, 세제 관련 유의사항을 입력하세요."))
    return issues


def estimate_run_cost(
    *,
    n_segments: int,
    mode: str,
    n_seeds: int,
    personas_per_segment: int,
    workers: int,
    include_sensitivity: bool = False,
    n_variants: int = 0,
) -> dict[str, Any]:
    turns_per_seed = 5 if mode == "debate" else 1
    base_cases = max(0, n_segments) * max(0, personas_per_segment)
    base_calls = base_cases * max(1, n_seeds) * turns_per_seed
    sensitivity_calls = 0
    if include_sensitivity:
        # sensitivity_analysis는 기준안을 다시 포함한다.
        variant_cases = max(0, n_segments) * max(1, personas_per_segment // 2)
        sensitivity_calls = variant_cases * max(1, n_seeds - 1) * turns_per_seed * max(1, n_variants + 1)
    total_calls = base_calls + sensitivity_calls
    sec_per_call = 0.04 if SETTINGS.backend == "mock" else (3.5 if mode == "single" else 5.0)
    estimated_seconds = max(1, int(total_calls * sec_per_call / max(1, workers)))
    return {
        "cases": base_cases,
        "llm_calls": total_calls,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": round(estimated_seconds / 60, 1),
    }


def segment_profile(segment: Segment, personas: list[Persona]) -> dict[str, Any]:
    pool = filter_segment(personas, segment)
    finances = [p.finance for p in pool if p.finance]
    nemotron_count = sum(1 for p in pool if is_nemotron_persona(p))
    synthetic_count = len(pool) - nemotron_count

    def avg(attr: str) -> float | None:
        if not finances:
            return None
        return round(sum(getattr(f, attr) for f in finances) / len(finances), 1)

    if not pool:
        status = "실행 불가"
    elif len(pool) < 5:
        status = "표본 부족"
    elif synthetic_count / len(pool) >= 0.5:
        status = "합성 폴백 주의"
    else:
        status = "검증 가능"

    return {
        "name": segment.name,
        "n_personas": len(pool),
        "avg_income_manwon": avg("annual_income_manwon"),
        "avg_dsr_pct": avg("dsr_pct"),
        "avg_surplus_manwon": avg("monthly_surplus_manwon"),
        "nemotron_count": nemotron_count,
        "synthetic_count": synthetic_count,
        "synthetic_ratio": round(synthetic_count / len(pool), 3) if pool else 0.0,
        "status": status,
    }


class WorkbenchDB:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS product_projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT '초안',
                    owner_id TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS product_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_project_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    product_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL,
                    change_note TEXT NOT NULL DEFAULT '',
                    UNIQUE(product_project_id, version_number),
                    FOREIGN KEY(product_project_id) REFERENCES product_projects(id)
                );

                CREATE TABLE IF NOT EXISTS segment_definitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    conditions_json TEXT NOT NULL,
                    is_preset INTEGER NOT NULL DEFAULT 0,
                    created_by TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS simulation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_version_id INTEGER NOT NULL,
                    preset TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    n_seeds INTEGER NOT NULL,
                    personas_per_segment INTEGER NOT NULL,
                    workers INTEGER NOT NULL,
                    persona_source TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    model_small TEXT NOT NULL,
                    model_judge TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    error_summary TEXT NOT NULL DEFAULT '',
                    artifact_path TEXT NOT NULL DEFAULT '',
                    sensitivity_path TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(product_version_id) REFERENCES product_versions(id)
                );

                CREATE TABLE IF NOT EXISTS simulation_case_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_run_id INTEGER NOT NULL,
                    segment TEXT NOT NULL,
                    persona_id TEXT NOT NULL,
                    modal_suitability TEXT NOT NULL,
                    intent_mean REAL NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    needs_review INTEGER NOT NULL,
                    label_counts_json TEXT NOT NULL DEFAULT '{}',
                    risks_json TEXT NOT NULL DEFAULT '[]',
                    recommendations_json TEXT NOT NULL DEFAULT '[]',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    grounding_doc_ids_json TEXT NOT NULL DEFAULT '[]',
                    transcript_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(simulation_run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS report_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_run_id INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_by TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(simulation_run_id) REFERENCES simulation_runs(id)
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
        self.ensure_preset_segments()

    # ------------------------------------------------------------------ audit
    def log_event(
        self,
        action: str,
        target_type: str,
        target_id: str | int,
        payload: dict[str, Any] | None = None,
        *,
        actor_id: str = "local",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs(actor_id, action, target_type, target_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (actor_id, action, target_type, str(target_id), json_dumps(payload or {}), now_iso()),
            )

    # --------------------------------------------------------------- products
    def save_product_version(
        self,
        product: Product,
        *,
        created_by: str = "local",
        change_note: str = "",
        status: ProductStatus = "초안",
        write_artifact: bool = True,
    ) -> ProductVersionRecord:
        now = now_iso()
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM product_projects WHERE id = ?", (product.product_id,)).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE product_projects
                    SET name = ?, category = ?, status = ?, owner_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (product.name, product.category, status, created_by, now, product.product_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO product_projects(id, name, category, status, owner_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (product.product_id, product.name, product.category, status, created_by, now, now),
                )
            version_number = (
                conn.execute(
                    "SELECT COALESCE(MAX(version_number), 0) + 1 AS n FROM product_versions WHERE product_project_id = ?",
                    (product.product_id,),
                ).fetchone()["n"]
            )
            artifact_path = ""
            if write_artifact:
                PRODUCT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                path = PRODUCT_ARTIFACT_DIR / f"{safe_slug(product.product_id)}_v{version_number}.json"
                path.write_text(product_to_json(product), encoding="utf-8")
                artifact_path = str(path)
            cur = conn.execute(
                """
                INSERT INTO product_versions(
                    product_project_id, version_number, product_json, artifact_path,
                    created_by, created_at, change_note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product.product_id,
                    version_number,
                    product_to_json(product),
                    artifact_path,
                    created_by,
                    now,
                    change_note,
                ),
            )
            version_id = int(cur.lastrowid)
        self.log_event(
            "product_version_saved",
            "product_version",
            version_id,
            {"product_id": product.product_id, "version_number": version_number, "artifact_path": artifact_path},
            actor_id=created_by,
        )
        return ProductVersionRecord(
            id=version_id,
            product_project_id=product.product_id,
            version_number=version_number,
            product=product,
            artifact_path=artifact_path,
            created_by=created_by,
            created_at=now,
            change_note=change_note,
        )

    def list_product_versions(self, product_project_id: str | None = None) -> list[ProductVersionRecord]:
        sql = "SELECT * FROM product_versions"
        params: tuple[Any, ...] = ()
        if product_project_id:
            sql += " WHERE product_project_id = ?"
            params = (product_project_id,)
        sql += " ORDER BY created_at DESC, id DESC"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._version_from_row(r) for r in rows]

    def get_product_version(self, version_id: int) -> ProductVersionRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM product_versions WHERE id = ?", (version_id,)).fetchone()
        if not row:
            raise KeyError(f"product_version {version_id} not found")
        return self._version_from_row(row)

    def _version_from_row(self, row: sqlite3.Row) -> ProductVersionRecord:
        return ProductVersionRecord(
            id=int(row["id"]),
            product_project_id=str(row["product_project_id"]),
            version_number=int(row["version_number"]),
            product=Product(**json.loads(row["product_json"])),
            artifact_path=str(row["artifact_path"]),
            created_by=str(row["created_by"]),
            created_at=str(row["created_at"]),
            change_note=str(row["change_note"]),
        )

    def portfolio_rows(self) -> list[PortfolioRow]:
        with self.connect() as conn:
            projects = conn.execute("SELECT * FROM product_projects ORDER BY updated_at DESC").fetchall()
        rows: list[PortfolioRow] = []
        for p in projects:
            versions = self.list_product_versions(str(p["id"]))
            latest = versions[0] if versions else None
            latest_run = self.latest_completed_run_for_project(str(p["id"]))
            sim = self.load_simulation_report(latest_run.id) if latest_run else None
            average_intent = None
            risk_segments = 0
            low_confidence_segments = 0
            if sim and sim.segments:
                average_intent = round(sum(s.mean_intent for s in sim.segments) / len(sim.segments), 1)
                risk_segments = sum(
                    1 for s in sim.segments if s.verdict_mix.get("fail", 0) or s.verdict_mix.get("warn", 0)
                )
                low_confidence_segments = sum(1 for s in sim.segments if s.low_confidence_ratio >= 0.5)
            rows.append(
                PortfolioRow(
                    project_id=str(p["id"]),
                    name=str(p["name"]),
                    category=str(p["category"]),
                    status=str(p["status"]),
                    version_number=latest.version_number if latest else None,
                    average_intent=average_intent,
                    risk_segments=risk_segments,
                    low_confidence_segments=low_confidence_segments,
                    last_run_at=latest_run.finished_at if latest_run else "",
                    next_action=self._next_action(str(p["status"]), latest_run is not None),
                )
            )
        return rows

    def _next_action(self, status: str, has_run: bool) -> str:
        if not has_run:
            return "검증 실행"
        if status == "보완 필요":
            return "조건 수정"
        if status == "출시 검토 가능":
            return "보고서 생성"
        return "상품정보 보완"

    # --------------------------------------------------------------- segments
    def ensure_preset_segments(self) -> None:
        for seg in load_segments():
            self.save_segment_definition(seg, is_preset=True, created_by="system")

    def save_segment_definition(
        self,
        segment: Segment,
        *,
        is_preset: bool = False,
        created_by: str = "local",
    ) -> SegmentDefinitionRecord:
        now = now_iso()
        payload = json_dumps(segment.model_dump(mode="json", exclude_none=True))
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM segment_definitions WHERE name = ?", (segment.name,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE segment_definitions
                    SET conditions_json = ?, is_preset = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (payload, int(is_preset), now, int(existing["id"])),
                )
                segment_id = int(existing["id"])
                created_at = str(existing["created_at"])
            else:
                cur = conn.execute(
                    """
                    INSERT INTO segment_definitions(
                        name, conditions_json, is_preset, created_by, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (segment.name, payload, int(is_preset), created_by, now, now),
                )
                segment_id = int(cur.lastrowid)
                created_at = now
        if not is_preset:
            self.log_event(
                "segment_saved",
                "segment_definition",
                segment_id,
                {"name": segment.name, "conditions": segment.model_dump(mode="json", exclude_none=True)},
                actor_id=created_by,
            )
        return SegmentDefinitionRecord(
            id=segment_id,
            name=segment.name,
            segment=segment,
            is_preset=is_preset,
            created_by=created_by,
            created_at=created_at,
            updated_at=now,
        )

    def list_segment_definitions(self) -> list[SegmentDefinitionRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM segment_definitions ORDER BY is_preset DESC, name ASC"
            ).fetchall()
        return [self._segment_from_row(r) for r in rows]

    def get_segment_by_name(self, name: str) -> SegmentDefinitionRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM segment_definitions WHERE name = ?", (name,)).fetchone()
        return self._segment_from_row(row) if row else None

    def _segment_from_row(self, row: sqlite3.Row) -> SegmentDefinitionRecord:
        segment = Segment(**json.loads(row["conditions_json"]))
        return SegmentDefinitionRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            segment=segment,
            is_preset=bool(row["is_preset"]),
            created_by=str(row["created_by"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    # ------------------------------------------------------------- simulations
    def start_simulation_run(
        self,
        *,
        product_version_id: int,
        preset: str,
        mode: str,
        n_seeds: int,
        personas_per_segment: int,
        workers: int,
        persona_source: str,
        settings: dict[str, Any],
        actor_id: str = "local",
    ) -> int:
        now = now_iso()
        with self.connect() as conn:
            version = conn.execute(
                "SELECT product_project_id FROM product_versions WHERE id = ?", (product_version_id,)
            ).fetchone()
            if not version:
                raise KeyError(f"product_version {product_version_id} not found")
            cur = conn.execute(
                """
                INSERT INTO simulation_runs(
                    product_version_id, preset, mode, n_seeds, personas_per_segment, workers,
                    persona_source, backend, model_small, model_judge, status, started_at, settings_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_version_id,
                    preset,
                    mode,
                    n_seeds,
                    personas_per_segment,
                    workers,
                    persona_source,
                    SETTINGS.backend,
                    SETTINGS.model_small,
                    SETTINGS.model_judge,
                    "실행 중",
                    now,
                    json_dumps(settings),
                ),
            )
            run_id = int(cur.lastrowid)
            conn.execute(
                "UPDATE product_projects SET status = ?, updated_at = ? WHERE id = ?",
                ("검증 중", now, str(version["product_project_id"])),
            )
        self.log_event(
            "simulation_started",
            "simulation_run",
            run_id,
            {"product_version_id": product_version_id, "preset": preset, "settings": settings},
            actor_id=actor_id,
        )
        return run_id

    def complete_simulation_run(
        self,
        run_id: int,
        sim: SimulationReport,
        *,
        artifact_path: str | Path,
        sensitivity_path: str | Path | None = None,
        actor_id: str = "local",
    ) -> None:
        now = now_iso()
        status = derive_status(sim)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT pv.product_project_id
                FROM simulation_runs sr
                JOIN product_versions pv ON pv.id = sr.product_version_id
                WHERE sr.id = ?
                """,
                (run_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"simulation_run {run_id} not found")
            conn.execute(
                """
                UPDATE simulation_runs
                SET status = ?, finished_at = ?, artifact_path = ?, sensitivity_path = ?, summary_json = ?
                WHERE id = ?
                """,
                (
                    "완료",
                    now,
                    str(artifact_path),
                    str(sensitivity_path or ""),
                    sim.model_dump_json(indent=2),
                    run_id,
                ),
            )
            conn.execute("DELETE FROM simulation_case_results WHERE simulation_run_id = ?", (run_id,))
            for segment in sim.segments:
                for case in segment.cases:
                    conn.execute(
                        """
                        INSERT INTO simulation_case_results(
                            simulation_run_id, segment, persona_id, modal_suitability, intent_mean,
                            confidence, confidence_level, needs_review, label_counts_json, risks_json,
                            recommendations_json, evidence_json, grounding_doc_ids_json, transcript_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            segment.segment,
                            case.persona_id,
                            case.modal_suitability,
                            case.intent_mean,
                            case.confidence,
                            case.confidence_level,
                            int(case.needs_review),
                            json_dumps(case.label_counts),
                            json_dumps(case.risks),
                            json_dumps(case.recommendations),
                            json_dumps(case.evidence),
                            json_dumps(case.grounding_doc_ids),
                            case.model_dump_json(indent=2),
                        ),
                    )
            conn.execute(
                "UPDATE product_projects SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, str(row["product_project_id"])),
            )
        self.log_event(
            "simulation_completed",
            "simulation_run",
            run_id,
            {"artifact_path": str(artifact_path), "sensitivity_path": str(sensitivity_path or ""), "status": status},
            actor_id=actor_id,
        )

    def fail_simulation_run(self, run_id: int, error_summary: str, *, actor_id: str = "local") -> None:
        now = now_iso()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT pv.product_project_id
                FROM simulation_runs sr
                JOIN product_versions pv ON pv.id = sr.product_version_id
                WHERE sr.id = ?
                """,
                (run_id,),
            ).fetchone()
            conn.execute(
                "UPDATE simulation_runs SET status = ?, finished_at = ?, error_summary = ? WHERE id = ?",
                ("실패", now, error_summary[:2000], run_id),
            )
            if row:
                conn.execute(
                    "UPDATE product_projects SET status = ?, updated_at = ? WHERE id = ?",
                    ("보완 필요", now, str(row["product_project_id"])),
                )
        self.log_event(
            "simulation_failed",
            "simulation_run",
            run_id,
            {"error_summary": error_summary[:2000]},
            actor_id=actor_id,
        )

    def list_simulation_runs(self, *, limit: int = 100) -> list[SimulationRunRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    sr.*,
                    pv.product_project_id,
                    pv.version_number,
                    pp.name AS product_name,
                    (
                        SELECT ra.path
                        FROM report_artifacts ra
                        WHERE ra.simulation_run_id = sr.id
                        ORDER BY ra.created_at DESC, ra.id DESC
                        LIMIT 1
                    ) AS report_path
                FROM simulation_runs sr
                JOIN product_versions pv ON pv.id = sr.product_version_id
                JOIN product_projects pp ON pp.id = pv.product_project_id
                ORDER BY sr.started_at DESC, sr.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._run_from_row(r) for r in rows]

    def latest_completed_run_for_project(self, product_project_id: str) -> SimulationRunRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    sr.*,
                    pv.product_project_id,
                    pv.version_number,
                    pp.name AS product_name,
                    (
                        SELECT ra.path
                        FROM report_artifacts ra
                        WHERE ra.simulation_run_id = sr.id
                        ORDER BY ra.created_at DESC, ra.id DESC
                        LIMIT 1
                    ) AS report_path
                FROM simulation_runs sr
                JOIN product_versions pv ON pv.id = sr.product_version_id
                JOIN product_projects pp ON pp.id = pv.product_project_id
                WHERE pv.product_project_id = ? AND sr.status = '완료'
                ORDER BY sr.finished_at DESC, sr.id DESC
                LIMIT 1
                """,
                (product_project_id,),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def get_simulation_run(self, run_id: int) -> SimulationRunRecord:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    sr.*,
                    pv.product_project_id,
                    pv.version_number,
                    pp.name AS product_name,
                    (
                        SELECT ra.path
                        FROM report_artifacts ra
                        WHERE ra.simulation_run_id = sr.id
                        ORDER BY ra.created_at DESC, ra.id DESC
                        LIMIT 1
                    ) AS report_path
                FROM simulation_runs sr
                JOIN product_versions pv ON pv.id = sr.product_version_id
                JOIN product_projects pp ON pp.id = pv.product_project_id
                WHERE sr.id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"simulation_run {run_id} not found")
        return self._run_from_row(row)

    def _run_from_row(self, row: sqlite3.Row) -> SimulationRunRecord:
        return SimulationRunRecord(
            id=int(row["id"]),
            product_version_id=int(row["product_version_id"]),
            product_project_id=str(row["product_project_id"]),
            product_name=str(row["product_name"]),
            version_number=int(row["version_number"]),
            preset=str(row["preset"]),
            mode=str(row["mode"]),
            n_seeds=int(row["n_seeds"]),
            personas_per_segment=int(row["personas_per_segment"]),
            workers=int(row["workers"]),
            persona_source=str(row["persona_source"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            finished_at=str(row["finished_at"]),
            artifact_path=str(row["artifact_path"]),
            sensitivity_path=str(row["sensitivity_path"]),
            report_path=str(row["report_path"] or ""),
            error_summary=str(row["error_summary"]),
            settings=json.loads(row["settings_json"] or "{}"),
        )

    def load_simulation_report(self, run_id: int) -> SimulationReport:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT summary_json, artifact_path FROM simulation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"simulation_run {run_id} not found")
        if row["summary_json"]:
            return SimulationReport(**json.loads(row["summary_json"]))
        path = Path(str(row["artifact_path"]))
        if path.exists():
            return SimulationReport.load(path)
        raise FileNotFoundError(f"simulation_run {run_id} has no saved summary")

    # --------------------------------------------------------------- artifacts
    def save_report_artifact(
        self,
        run_id: int,
        *,
        path: str | Path,
        fmt: str = "markdown",
        metadata: dict[str, Any] | None = None,
        actor_id: str = "local",
    ) -> int:
        now = now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO report_artifacts(simulation_run_id, format, path, created_by, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, fmt, str(path), actor_id, now, json_dumps(metadata or {})),
            )
            artifact_id = int(cur.lastrowid)
        self.log_event(
            "report_generated",
            "report_artifact",
            artifact_id,
            {"simulation_run_id": run_id, "format": fmt, "path": str(path)},
            actor_id=actor_id,
        )
        return artifact_id

    def save_sensitivity_artifact(
        self,
        run_id: int,
        rows: list[SensitivityRow],
        *,
        path: str | Path,
        actor_id: str = "local",
    ) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_dumps([r.model_dump(mode="json") for r in rows]), encoding="utf-8")
        with self.connect() as conn:
            conn.execute("UPDATE simulation_runs SET sensitivity_path = ? WHERE id = ?", (str(p), run_id))
        self.log_event(
            "sensitivity_generated",
            "simulation_run",
            run_id,
            {"path": str(p), "rows": len(rows)},
            actor_id=actor_id,
        )
        return p

    def audit_rows(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT actor_id, action, target_type, target_id, payload_json, created_at
                FROM audit_logs
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "actor": r["actor_id"],
                "action": r["action"],
                "target_type": r["target_type"],
                "target_id": r["target_id"],
                "payload": json.loads(r["payload_json"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]


def load_sensitivity_rows(path: str | Path) -> list[SensitivityRow]:
    p = Path(path)
    if not p.exists():
        return []
    return [SensitivityRow(**row) for row in json.loads(p.read_text(encoding="utf-8"))]
