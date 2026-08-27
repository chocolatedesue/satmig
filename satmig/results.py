"""Result serialisation: the four artefacts every run must produce.

1. ``manifest.json``     provenance, config, fingerprint, model boundaries
2. ``slots.csv``         one row per (policy, pipeline, slot)
3. ``policy_metrics.csv``one row per policy
4. ``summary.json``      headline numbers, comparisons and verdicts
"""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .simulator import PolicyResult, Simulator

SLOT_FIELDS = [
    "t_s",
    "policy",
    "pid",
    "n_stages",
    "lit_stages",
    "min_capacity",
    "throughput_tok_s",
    "tpot_ms",
    "slo_ok",
    "hops_ms",
    "migration_events",
    "stage_moves",
    "migrated_gb",
    "downtime_ms",
    "deadline_miss",
    "peak_link_flows",
    "colocated_stages",
    "battery_wh",
]


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def policy_metrics(cfg: RunConfig, res: PolicyResult) -> dict:
    recs = res.records
    if not recs:
        return {"policy": res.policy}
    horizon_h = cfg.sim.horizon_s / 3600.0
    n_pipes = max(r.pid for r in recs) + 1
    tps = [r.throughput_tok_s for r in recs]
    lat = [r.tpot_ms for r in recs if math.isfinite(r.tpot_ms)]
    agg_by_slot: dict[float, float] = {}
    for r in recs:
        agg_by_slot[r.t_s] = agg_by_slot.get(r.t_s, 0.0) + r.throughput_tok_s
    agg = list(agg_by_slot.values())
    slots_per_pipe = len(recs) / n_pipes
    all_lit = sum(1 for r in recs if r.lit_stages == r.n_stages) / len(recs)
    return {
        "policy": res.policy,
        "n_stages": res.n_stages,
        "n_pipelines": n_pipes,
        "memory_feasible": int(res.memory_feasible),
        "max_stages_under_slo": res.slo_feasible_stages,
        "mean_throughput_tok_s": statistics.fmean(tps),
        "p05_throughput_tok_s": _pct(tps, 5),
        "aggregate_throughput_tok_s": statistics.fmean(agg),
        "mean_tpot_ms": statistics.fmean(lat) if lat else float("inf"),
        "p99_tpot_ms": _pct(lat, 99) if lat else float("inf"),
        "slo_attainment": statistics.fmean([r.slo_ok for r in recs]),
        "p_all_stages_lit": all_lit,
        "migration_events_per_hour": sum(r.migration_events for r in recs)
        / n_pipes
        / horizon_h,
        "stage_moves_per_hour": sum(r.stage_moves for r in recs) / n_pipes / horizon_h,
        "migrated_gb_per_hour": sum(r.migrated_gb for r in recs) / n_pipes / horizon_h,
        "downtime_fraction": sum(r.downtime_ms for r in recs)
        / (slots_per_pipe * cfg.sim.slot_s * 1e3)
        / n_pipes,
        "deadline_misses": sum(r.deadline_miss for r in recs),
        "peak_link_flows": max(r.peak_link_flows for r in recs),
        "colocation_rate": statistics.fmean(
            [r.colocated_stages / max(1, r.n_stages) for r in recs]
        ),
        "battery_wh_per_orbit": sum(r.battery_wh for r in recs)
        / n_pipes
        / (cfg.sim.horizon_s / 5729.0),
        "notes": "; ".join(res.notes),
    }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


def write_results(
    cfg: RunConfig, results: list[PolicyResult], sim: Simulator, outdir: Path
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    slots_path = outdir / "slots.csv"
    with slots_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SLOT_FIELDS)
        w.writeheader()
        for res in results:
            for r in res.records:
                w.writerow(asdict(r))

    metrics = [policy_metrics(cfg, r) for r in results]
    metrics_path = outdir / "policy_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)

    manifest = {
        "run_name": cfg.name,
        "description": cfg.description,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": cfg.fingerprint(),
        "git_rev": _git_rev(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg.to_dict(),
        "derived": {
            "orbital_period_s": sim.con.period_s,
            "hop_period_s": sim.con.hop_period_s(),
            "along_track_hop_ms": sim.con.hop_delays.along_ms,
            "cross_plane_hop_ms": sim.con.hop_delays.cross_ms,
            "plane_beta_deg": [round(b, 3) for b in sim.con.plane_beta_deg],
            "plane_eclipse_fraction": [round(p.f_ecl, 5) for p in sim.con.planes],
            "planes_eclipse_free": sum(1 for p in sim.con.planes if p.f_ecl <= 0),
            "min_stages_for_memory": _safe_min_stages(sim, cfg),
        },
        "model_boundaries": [
            "Capacity is a normalised compute proxy, not PV watts.",
            "Umbra only: cylindrical shadow of a spherical Earth, no penumbra,"
            " no atmosphere, no J2 short-period terms.",
            "Battery is a flat derate c_eclipse, not a state-of-charge model.",
            "Attitude and solar-panel incidence are not modelled, so the"
            " no-power time here is a lower bound (see LAB-67).",
            "Handovers are priced within one slot; sub-slot queueing dynamics"
            " are not resolved.",
        ],
        "files": ["manifest.json", "slots.csv", "policy_metrics.csv", "summary.json"],
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = build_summary(cfg, metrics, manifest)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _safe_min_stages(sim: Simulator, cfg: RunConfig) -> int:
    from .perf import min_stages_for_memory

    try:
        return min_stages_for_memory(sim.model, sim.sat, cfg.workload.stacking_k)
    except ValueError:
        return -1


def build_summary(cfg: RunConfig, metrics: list[dict], manifest: dict) -> dict:
    by = {m["policy"]: m for m in metrics}
    baseline = next(
        (n for n in ("conveyor", "static", "reactive_jit") if n in by), metrics[0]["policy"]
    )
    comparisons = {}
    b = by[baseline]
    for name, m in by.items():
        if name == baseline:
            continue
        comparisons[name] = {
            "vs": baseline,
            "throughput_x": _ratio(m["mean_throughput_tok_s"], b["mean_throughput_tok_s"]),
            "aggregate_throughput_x": _ratio(
                m["aggregate_throughput_tok_s"], b["aggregate_throughput_tok_s"]
            ),
            "tpot_reduction_pct": _reduction(m["mean_tpot_ms"], b["mean_tpot_ms"]),
            "p99_tpot_reduction_pct": _reduction(m["p99_tpot_ms"], b["p99_tpot_ms"]),
            "slo_attainment_delta": m["slo_attainment"] - b["slo_attainment"],
            "migration_events_reduction_pct": _reduction(
                m["migration_events_per_hour"], b["migration_events_per_hour"]
            ),
            "downtime_reduction_pct": _reduction(
                m["downtime_fraction"], b["downtime_fraction"]
            ),
            "deadline_miss_delta": m["deadline_misses"] - b["deadline_misses"],
        }
    return {
        "run_name": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "baseline": baseline,
        "derived": manifest["derived"],
        "policies": metrics,
        "comparisons": comparisons,
    }


def _ratio(a: float, b: float) -> float:
    if b == 0 or not math.isfinite(a) or not math.isfinite(b):
        return float("inf") if a > 0 else float("nan")
    return a / b


def _reduction(a: float, b: float) -> float:
    """Percent reduction of ``a`` relative to ``b`` (positive = a is smaller)."""
    if not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return float("nan")
    return (b - a) / b * 100.0
