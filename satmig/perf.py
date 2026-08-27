"""Performance model of a pipeline-parallel LLM stage chain on satellites.

Formula provenance (all five sources are Paper Library entries; the mapping to
the satellite setting is ours):

* 1F1B pipeline makespan ``(M + P - 1) * max_s t_s``  -- classic PP, the form
  LAB-47 used as ``P * M/(M+P-1)``.
* ``l_req = l_sch + l_exe``, ``l_exe(N|M) = t_exe(M) + sum t_exe(1)``
  -- SpotServe (ASPLOS'24), arXiv:2311.15566.
* ``MaxBlocks(i, L) = floor((M_i * u - L * W) / (L * P_blk))`` and the
  layer-stacking granularity constraint -- PipeLive, arXiv:2604.12171.
* ``F = (M - sum V) / B`` instance freeness -- Llumnix, arXiv:2406.03243.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One LLM, in the units the throughput model needs."""

    name: str
    n_layers: int
    weight_gb: float
    kv_gb_full: float
    """KV bytes when the serving batch is at its working point."""
    decode_ms_per_layer: float
    """Per-layer decode compute time for one micro-batch on a lit satellite."""
    prefill_ms_per_layer: float
    microbatches: int
    microbatch_tokens: int = 1

    def layers_of_stage(self, n_stages: int, stage: int, stacking_k: int = 1) -> int:
        """Balanced split, respecting PipeLive's layer-stacking granularity."""
        if n_stages < 1 or stage < 0 or stage >= n_stages:
            raise ValueError("bad stage index")
        units = self.n_layers // stacking_k
        base = units // n_stages
        extra = units % n_stages
        return (base + (1 if stage < extra else 0)) * stacking_k


@dataclass(frozen=True)
class SatelliteSpec:
    memory_gb: float
    memory_util_cap: float = 0.9
    kv_block_gb: float = 0.001
    compute_power_w: float = 300.0
    battery_wh: float = 400.0


def max_kv_blocks(sat: SatelliteSpec, layers_here: int, weight_gb_per_layer: float) -> int:
    """PipeLive MaxBlocks(i, L) = floor((M_i*u - L*W) / (L*P_blk))."""
    if layers_here <= 0:
        return 0
    avail = sat.memory_gb * sat.memory_util_cap - layers_here * weight_gb_per_layer
    if avail <= 0:
        return 0
    return int(math.floor(avail / (layers_here * sat.kv_block_gb)))


def memory_feasible(
    model: ModelSpec, sat: SatelliteSpec, n_stages: int, stacking_k: int = 1
) -> bool:
    """Can the widest stage of a P-way split fit, weights plus its KV share?"""
    wpl = model.weight_gb / model.n_layers
    worst = max(
        model.layers_of_stage(n_stages, s, stacking_k) for s in range(n_stages)
    )
    kv_here = model.kv_gb_full * worst / model.n_layers
    return worst * wpl + kv_here <= sat.memory_gb * sat.memory_util_cap


def min_stages_for_memory(model: ModelSpec, sat: SatelliteSpec, stacking_k: int = 1) -> int:
    for p in range(1, model.n_layers + 1):
        if memory_feasible(model, sat, p, stacking_k):
            return p
    raise ValueError("model does not fit at any split")


def stage_times_ms(
    model: ModelSpec,
    capacities: list[float],
    stacking_k: int = 1,
    phase: str = "decode",
) -> list[float]:
    """Per-stage service time, inflated by 1/c on a derated (eclipsed) host."""
    p = len(capacities)
    per_layer = (
        model.decode_ms_per_layer if phase == "decode" else model.prefill_ms_per_layer
    )
    out = []
    for s, c in enumerate(capacities):
        if c <= 0:
            out.append(math.inf)
            continue
        out.append(model.layers_of_stage(p, s, stacking_k) * per_layer / c)
    return out


def throughput_tok_s(
    model: ModelSpec,
    capacities: list[float],
    hop_delays_ms: list[float],
    stacking_k: int = 1,
    migration_downtime_fraction: float = 0.0,
) -> float:
    """Steady-state 1F1B decode throughput of one pipeline.

    ``(M + P - 1) * max_s t_s`` clocks M micro-batches, plus the inter-stage
    propagation that a micro-batch must traverse once per sweep.
    """
    p = len(capacities)
    ts = stage_times_ms(model, capacities, stacking_k)
    bottleneck = max(ts)
    if not math.isfinite(bottleneck) or bottleneck <= 0:
        return 0.0
    m = model.microbatches
    span_ms = (m + p - 1) * bottleneck + sum(hop_delays_ms)
    tokens = m * model.microbatch_tokens
    return tokens / (span_ms / 1e3) * max(0.0, 1.0 - migration_downtime_fraction)


def tpot_ms(
    model: ModelSpec,
    capacities: list[float],
    hop_delays_ms: list[float],
    stacking_k: int = 1,
) -> float:
    """Per-output-token latency of one sequence: compute plus a full ring trip.

    The sampled token has to travel back to stage 0 for the next step, so the
    hop budget is ``sum(forward hops) + return hop``.
    """
    ts = stage_times_ms(model, capacities, stacking_k)
    if any(not math.isfinite(x) for x in ts):
        return math.inf
    return sum(ts) + sum(hop_delays_ms)


def instance_freeness_steps(
    memory_blocks_total: int, virtual_usage_blocks: int, blocks_per_step: int
) -> float:
    """Llumnix ``F = (M - sum V) / B``, in steps."""
    if blocks_per_step <= 0:
        return math.inf
    return (memory_blocks_total - virtual_usage_blocks) / blocks_per_step


def energy_freeness_s(
    time_to_eclipse_s: float, battery_wh: float, compute_power_w: float
) -> float:
    """Our satellite analogue of Llumnix freeness.

    Llumnix asks "how many more steps can this instance run before it runs out
    of KV blocks".  On orbit the exhaustible resource is energy, and the answer
    is *exactly predictable*: run until the umbra, then until the battery is
    flat.
    """
    if compute_power_w <= 0:
        return math.inf
    battery_s = battery_wh / compute_power_w * 3600.0
    if math.isinf(time_to_eclipse_s):
        return math.inf
    return time_to_eclipse_s + battery_s


def composite_score(ttft_norm: float, tpot_norm: float, tp_norm: float) -> float:
    """PipeLive's Score = (s_TTFT + s_TPOT + s_TP) / 3."""
    return (ttft_norm + tpot_norm + tp_norm) / 3.0
