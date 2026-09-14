"""Versioned evaluation manifests, run metadata, and metric gates."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


REPORT_SCHEMA_VERSION = 1
DEFAULT_DATASET_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "datasets" / "v1" / "manifest.json"
)


class DatasetArtifact(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    path: str
    format: Literal["jsonl"]
    case_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    scope: list[str] = Field(default_factory=list)
    runtime_preconditions: list[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def path_must_be_repository_relative(cls, value: str) -> str:
        # Validate both POSIX and Windows spellings even when the evaluator
        # runs on only one of those platforms.
        candidate = PurePosixPath(value.replace("\\", "/"))
        windows_drive_path = len(value) >= 2 and value[1] == ":"
        if (
            Path(value).is_absolute()
            or candidate.is_absolute()
            or windows_drive_path
            or ".." in candidate.parts
        ):
            raise ValueError("dataset path must stay within the repository")
        return value


class QualityGateSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    mode: Literal["deterministic", "live", "retrieval"]
    dataset_id: str
    minimums: dict[str, float] = Field(default_factory=dict)
    maximums: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def metric_names_cannot_overlap(self) -> "QualityGateSpec":
        overlap = set(self.minimums) & set(self.maximums)
        if overlap:
            raise ValueError(
                f"gate metrics cannot have both minimum and maximum: {sorted(overlap)}"
            )
        if not self.minimums and not self.maximums:
            raise ValueError("quality gate must define at least one metric")
        return self


class DatasetManifest(BaseModel):
    schema_version: Literal[1]
    dataset_version: str = Field(min_length=1, max_length=64)
    created_at: str
    description: str = ""
    cases: list[DatasetArtifact] = Field(min_length=1)
    quality_gates: list[QualityGateSpec] = Field(default_factory=list)
    interpretation: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ids_and_gate_references_must_be_valid(self) -> "DatasetManifest":
        artifact_ids = [artifact.id for artifact in self.cases]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("dataset artifact ids must be unique")
        gate_names = [gate.name for gate in self.quality_gates]
        if len(gate_names) != len(set(gate_names)):
            raise ValueError("quality gate names must be unique")
        unknown = {
            gate.dataset_id for gate in self.quality_gates
            if gate.dataset_id not in artifact_ids
        }
        if unknown:
            raise ValueError(f"quality gates reference unknown datasets: {sorted(unknown)}")
        return self


@dataclass(frozen=True)
class DatasetContext:
    dataset_id: str
    dataset_version: str
    manifest_path: str
    path: str
    sha256: str
    case_count: int


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    failures: list[str]
    observed: dict[str, float | int | None]
    minimums: dict[str, float]
    maximums: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def count_jsonl_cases(path: Path | str) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(
            1 for raw_line in handle
            if raw_line.strip() and not raw_line.lstrip().startswith("#")
        )


def load_dataset_manifest(
    path: Path | str = DEFAULT_DATASET_MANIFEST_PATH,
) -> DatasetManifest:
    manifest_path = Path(path)
    try:
        return DatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"invalid evaluation manifest at {manifest_path}: {exc}") from exc


def find_repository_root(start: Path | str) -> Path:
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "evals").is_dir():
            return directory
    raise ValueError(f"repository root not found from {start}")


def validate_dataset_artifact(
    manifest_path: Path | str,
    dataset_id: str,
    cases_path: Path | str | None = None,
) -> tuple[DatasetManifest, DatasetContext]:
    """Validate path containment, count, and checksum before an evaluation run."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_dataset_manifest(manifest_file)
    try:
        artifact = next(item for item in manifest.cases if item.id == dataset_id)
    except StopIteration as exc:
        raise ValueError(f"dataset {dataset_id!r} is not declared in {manifest_file}") from exc

    repository_root = find_repository_root(manifest_file)
    artifact_path = (repository_root / artifact.path).resolve()
    if repository_root != artifact_path and repository_root not in artifact_path.parents:
        raise ValueError(f"dataset path escapes repository root: {artifact.path}")
    if cases_path is not None and Path(cases_path).resolve() != artifact_path:
        raise ValueError(
            f"selected cases do not match manifest dataset {dataset_id!r}: {cases_path}"
        )
    if not artifact_path.is_file():
        raise ValueError(f"dataset file is missing: {artifact.path}")

    observed_count = count_jsonl_cases(artifact_path)
    if observed_count != artifact.case_count:
        raise ValueError(
            f"dataset case count mismatch for {dataset_id}: "
            f"expected {artifact.case_count}, observed {observed_count}"
        )
    observed_hash = sha256_file(artifact_path)
    if observed_hash.casefold() != artifact.sha256.casefold():
        raise ValueError(
            f"dataset checksum mismatch for {dataset_id}: "
            f"expected {artifact.sha256}, observed {observed_hash}"
        )

    return manifest, DatasetContext(
        dataset_id=artifact.id,
        dataset_version=manifest.dataset_version,
        manifest_path=manifest_file.relative_to(repository_root).as_posix(),
        path=artifact_path.relative_to(repository_root).as_posix(),
        sha256=observed_hash,
        case_count=observed_count,
    )


def get_quality_gate(
    manifest: DatasetManifest,
    dataset_id: str,
    mode: str,
) -> QualityGateSpec | None:
    return next(
        (
            gate for gate in manifest.quality_gates
            if gate.dataset_id == dataset_id and gate.mode == mode
        ),
        None,
    )


def evaluate_quality_gate(
    metrics: dict[str, object],
    spec: QualityGateSpec,
) -> GateResult:
    failures: list[str] = []
    observed: dict[str, float | int | None] = {}
    for metric, minimum in spec.minimums.items():
        value = metrics.get(metric)
        numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        observed[metric] = numeric
        if numeric is None:
            failures.append(f"{metric} is missing or non-numeric")
        elif numeric < minimum:
            failures.append(f"{metric}={numeric} is below minimum {minimum}")
    for metric, maximum in spec.maximums.items():
        value = metrics.get(metric)
        numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        observed[metric] = numeric
        if numeric is None:
            failures.append(f"{metric} is missing or non-numeric")
        elif numeric > maximum:
            failures.append(f"{metric}={numeric} exceeds maximum {maximum}")
    return GateResult(
        name=spec.name,
        passed=not failures,
        failures=failures,
        observed=observed,
        minimums=dict(spec.minimums),
        maximums=dict(spec.maximums),
    )


def default_zero_failure_gate(
    metrics: dict[str, object],
    mode: str,
) -> GateResult:
    spec = QualityGateSpec(
        name=f"{mode}-zero-failure",
        mode=mode,
        dataset_id="ad-hoc",
        maximums={"failed": 0, "illegal_tool_calls": 0},
    )
    return evaluate_quality_gate(metrics, spec)


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return revision or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def build_run_metadata(
    *,
    mode: str,
    started_at_monotonic: float,
    started_at: str,
    dataset: DatasetContext | None,
    cases_path: Path | str,
    model: str,
    provider: str,
    external_access: bool,
) -> dict:
    repository_root = find_repository_root(Path(__file__))
    revision, worktree_dirty = _git_state(repository_root)
    prompt_path = repository_root / "src" / "agent" / "shopping_prompts.py"
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "mode": mode,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_ms": round((time.perf_counter() - started_at_monotonic) * 1000),
        "dataset": asdict(dataset) if dataset else {
            "dataset_id": "ad-hoc",
            "dataset_version": "unversioned",
            "manifest_path": None,
            "path": str(Path(cases_path)),
            "sha256": sha256_file(cases_path),
            "case_count": count_jsonl_cases(cases_path),
        },
        "code": {
            "git_revision": revision,
            "worktree_dirty": worktree_dirty,
            "prompt_sha256": sha256_file(prompt_path) if prompt_path.is_file() else None,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "model": model,
            "provider": provider,
            "external_access": external_access,
        },
    }


def evaluation_start() -> tuple[float, str]:
    return time.perf_counter(), _utc_now()
