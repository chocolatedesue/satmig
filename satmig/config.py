"""Configuration: everything the simulator does is driven from a YAML file.

Contract (inherited from the workspace's YQH-827 scaffolding requirement):
change the config, never the source; one command entry point; deterministic
re-runs from a seed; four result files per run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .migration import MigrationParams
from .perf import ModelSpec, SatelliteSpec


@dataclass
class ConstellationConfig:
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    n_planes: int = 12
    n_sats_per_plane: int = 22
    raan_spread_deg: float = 360.0
    raan_offset_deg: float = 0.0
    epoch_days_since_j2000: float = 9000.0
    cross_plane_latitude_deg: float = 0.0
    plane_beta_deg: list[float] | None = None
    """Override the computed per-plane beta angles (for regression tests)."""


@dataclass
class WorkloadConfig:
    n_pipelines: int = 1
    stages: int | None = None
    """Fixed split; ``None`` lets the policy choose."""
    placement: str = "along_track"
    """along_track | cross_plane | snake"""
    snake_width: int = 4
    slo_tpot_ms: float = 100.0
    stacking_k: int = 4
    """PipeLive layer-stacking factor; layer splits are multiples of k."""
    plane_assignment: str = "round_robin"
    """round_robin | best_beta"""
    plane_window: int | None = None
    """Restrict placement to this many planes -- the 'hot service region' of
    SCOPE, and the reason K pipelines actually contend for the same lit arc."""


@dataclass
class SimConfig:
    horizon_s: float = 5729.0 * 3.0
    slot_s: float = 10.0
    seed: int = 20260827
    c_eclipse: float = 0.2
    policies: list[str] = field(
        default_factory=lambda: ["static", "conveyor", "eph"]
    )


@dataclass
class RunConfig:
    name: str = "base"
    description: str = ""
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    model: dict[str, Any] = field(default_factory=dict)
    satellite: dict[str, Any] = field(default_factory=dict)
    migration: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, list[Any]] = field(default_factory=dict)
    """Optional cartesian sweep, e.g. ``{"workload.stages": [4, 8, 14]}``."""

    # -- derived specs -----------------------------------------------------
    def model_spec(self) -> ModelSpec:
        d = dict(DEFAULT_MODEL)
        d.update(self.model)
        return ModelSpec(**d)

    def satellite_spec(self) -> SatelliteSpec:
        d = dict(DEFAULT_SATELLITE)
        d.update(self.satellite)
        return SatelliteSpec(**d)

    def migration_params(self) -> MigrationParams:
        d = dict(DEFAULT_MIGRATION)
        d.update(self.migration)
        return MigrationParams(**d)

    # -- provenance --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "constellation": asdict(self.constellation),
            "workload": asdict(self.workload),
            "sim": asdict(self.sim),
            "model": asdict(self.model_spec()),
            "satellite": asdict(self.satellite_spec()),
            "migration": asdict(self.migration_params()),
            "sweep": self.sweep,
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_override(self, dotted: str, value: Any) -> "RunConfig":
        """Return a copy with one dotted key replaced (used by sweeps)."""
        head, _, tail = dotted.partition(".")
        if not tail:
            if head not in {"name", "description"}:
                raise KeyError(f"cannot override top-level key {head!r}")
            return replace(self, **{head: value})
        if head in {"model", "satellite", "migration", "sweep"}:
            d = dict(getattr(self, head))
            d[tail] = value
            return replace(self, **{head: d})
        section = getattr(self, head, None)
        if section is None:
            raise KeyError(f"unknown config section {head!r}")
        if not hasattr(section, tail):
            raise KeyError(f"unknown key {dotted!r}")
        return replace(self, **{head: replace(section, **{tail: value})})


# Llama-3-70B class model, fp16, calibrated so that the whole-model decode
# compute matches the 42 ms figure LAB-47 used, and the KV working set matches
# its 42.9 GB.
DEFAULT_MODEL: dict[str, Any] = {
    "name": "llama3-70b-fp16",
    "n_layers": 80,
    "weight_gb": 140.0,
    "kv_gb_full": 42.9,
    "decode_ms_per_layer": 0.525,
    "prefill_ms_per_layer": 5.0,
    "microbatches": 32,
    "microbatch_tokens": 1,
}

DEFAULT_SATELLITE: dict[str, Any] = {
    "memory_gb": 24.0,
    "memory_util_cap": 0.9,
    "kv_block_gb": 0.001,
    "compute_power_w": 300.0,
    "battery_wh": 400.0,
}

DEFAULT_MIGRATION: dict[str, Any] = {
    "isl_gbps": 100.0,
    "ack_ms": 20.0,
    "buffer_ms": 500.0,
    "cutover_ms_epoch": 11.1,
    "cutover_ms_global": 555.0,
    "kv_dirty_gb_per_s": 0.05,
    "precopy_tau_gb": 0.01,
    "max_precopy_rounds": 8,
    "weight_prestage_gbps": 1.0,
}


def _build(section_cls, data: dict[str, Any] | None):
    if not data:
        return section_cls()
    known = {f for f in section_cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**data)


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> RunConfig:
    allowed = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise KeyError(f"unknown top-level config keys: {sorted(unknown)}")
    return RunConfig(
        name=raw.get("name", "base"),
        description=raw.get("description", ""),
        constellation=_build(ConstellationConfig, raw.get("constellation")),
        workload=_build(WorkloadConfig, raw.get("workload")),
        sim=_build(SimConfig, raw.get("sim")),
        model=raw.get("model", {}) or {},
        satellite=raw.get("satellite", {}) or {},
        migration=raw.get("migration", {}) or {},
        sweep=raw.get("sweep", {}) or {},
    )
