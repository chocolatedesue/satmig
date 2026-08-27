"""Migration cost model: what a stage handover actually costs on orbit.

Ported formulas
---------------
SCOPE (APNet'26, doi:10.1145/3820441.3820451), Eq. 3 -- latest safe start of a
proactive transfer::

    T_start = min_i { T_leave,i - ( sum_c Size(c) / B_ISL + T_ack + T_buffer ) }

We substitute ``T_leave`` (a satellite leaving the service region) with
``T_eclipse_entry`` (a satellite leaving the powered set), and ``Size(c)``
with the stage's KV plus any non-resident weight shard.

SCOPE Eq. 2 -- value density ``V_c = lambda_c / Size(c)`` -- becomes the
priority with which competing stage handovers claim scarce ISL capacity.

PipeLive (arXiv:2604.12171) -- incremental KV patching converges when
``T_sched - T_applied < tau``; we model that as geometric pre-copy rounds
driven by the KV dirty rate, with a residual cutover.

CONNEX (SIGCOMM'26, doi:10.1145/3789240.3829200) -- cutover is pair-local
(11.1 ms measured) under epoch routing, versus a stop-the-world communicator
rebuild ~50x slower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GB_BITS = 8.0 * 1024**3


@dataclass(frozen=True)
class MigrationParams:
    isl_gbps: float = 100.0
    ack_ms: float = 20.0
    buffer_ms: float = 500.0
    cutover_ms_epoch: float = 11.1
    """CONNEX pair-local cutover, measured."""
    cutover_ms_global: float = 555.0
    """NCCL-style global communicator rebuild (CONNEX: ~50x slower)."""
    kv_dirty_gb_per_s: float = 0.05
    precopy_tau_gb: float = 0.01
    """PipeLive convergence threshold, expressed as residual dirty bytes."""
    max_precopy_rounds: int = 8
    weight_prestage_gbps: float = 1.0
    """Bandwidth reserved for read-only weight pre-staging."""


def transfer_ms(gb: float, gbps: float) -> float:
    if gbps <= 0:
        return math.inf
    return gb * GB_BITS / (gbps * 1e9) * 1e3


@dataclass(frozen=True)
class HandoverCost:
    bytes_gb: float
    precopy_ms: float
    downtime_ms: float
    rounds: int
    converged: bool

    @property
    def total_ms(self) -> float:
        return self.precopy_ms + self.downtime_ms


def stage_handover_cost(
    kv_gb: float,
    weight_gb_to_move: float,
    params: MigrationParams,
    available_gbps: float | None = None,
    epoch_routing: bool = True,
) -> HandoverCost:
    """Cost of moving one stage, PipeLive-style incremental KV patching."""
    bw = params.isl_gbps if available_gbps is None else available_gbps
    cutover = params.cutover_ms_epoch if epoch_routing else params.cutover_ms_global
    if bw <= 0:
        return HandoverCost(kv_gb + weight_gb_to_move, math.inf, math.inf, 0, False)

    # Weights are read-only: they stream in the background at a reserved rate
    # and never block the inference path (PipeLive weight loader).
    weight_ms = transfer_ms(weight_gb_to_move, min(params.weight_prestage_gbps, bw))

    remaining = kv_gb
    precopy = 0.0
    rounds = 0
    converged = False
    while rounds < params.max_precopy_rounds:
        t = transfer_ms(remaining, bw)
        precopy += t
        rounds += 1
        dirty = params.kv_dirty_gb_per_s * (t / 1e3)
        remaining = dirty
        if remaining <= params.precopy_tau_gb:
            converged = True
            break
    downtime = transfer_ms(remaining, bw) + params.ack_ms + cutover
    return HandoverCost(
        bytes_gb=kv_gb + weight_gb_to_move,
        precopy_ms=max(precopy, weight_ms),
        downtime_ms=downtime,
        rounds=rounds,
        converged=converged,
    )


def scope_trigger_time_s(
    eclipse_entry_s: float,
    bytes_gb: float,
    params: MigrationParams,
    available_gbps: float | None = None,
) -> float:
    """SCOPE Eq. 3, in seconds before the deadline.

    Returns the lead time at which the transfer must start.  Positive means
    "start this many seconds before eclipse entry".
    """
    bw = params.isl_gbps if available_gbps is None else available_gbps
    return (
        transfer_ms(bytes_gb, bw) + params.ack_ms + params.buffer_ms
    ) / 1e3


def value_density(access_rate: float, size_gb: float) -> float:
    """SCOPE Eq. 2, V_c = lambda_c / Size(c)."""
    if size_gb <= 0:
        return math.inf
    return access_rate / size_gb


def stop_and_copy_cost(
    kv_gb: float, weight_gb: float, params: MigrationParams, epoch_routing: bool = False
) -> HandoverCost:
    """No pre-copy, no weight pre-staging: the SpotServe 'reparallelisation'
    baseline that SpotServe itself criticises."""
    bw = params.isl_gbps
    cutover = params.cutover_ms_epoch if epoch_routing else params.cutover_ms_global
    dt = transfer_ms(kv_gb + weight_gb, bw) + params.ack_ms + cutover
    return HandoverCost(kv_gb + weight_gb, 0.0, dt, 0, True)


def recompute_cost_ms(
    prefill_ms_per_layer: float, n_layers: int, tokens_in_flight: int
) -> float:
    """Llumnix ``Loss = T_requeue + T_recompute`` -- the price of dropping KV
    instead of migrating it."""
    return prefill_ms_per_layer * n_layers * max(0, tokens_in_flight)


class LinkLoad:
    """Per-slot ISL bandwidth accounting, so migration storms become visible.

    Usage per slot: ``reset()``, then ``declare(links)`` for every concurrent
    handover, then ``share_for(links)`` to get the per-flow fair share.  The
    fair share is what makes a storm cost something: K simultaneous handovers
    on the same link each get B/K and can miss their SCOPE Eq. 3 deadline.
    """

    def __init__(self, isl_gbps: float) -> None:
        self.isl_gbps = isl_gbps
        self._flows: dict[tuple, int] = {}

    def reset(self) -> None:
        self._flows = {}

    def declare(self, links: list[tuple]) -> None:
        for link in links:
            self._flows[link] = self._flows.get(link, 0) + 1

    def flows_on(self, link: tuple) -> int:
        return self._flows.get(link, 0)

    def share_for(self, links: list[tuple]) -> float:
        """Bottleneck fair share across every link a flow traverses."""
        if not links:
            return self.isl_gbps
        return min(self.isl_gbps / max(1, self._flows.get(l, 1)) for l in links)

    def peak_flows(self) -> int:
        return max(self._flows.values()) if self._flows else 0

    def peak_gbps(self) -> float:
        """Offered migration load on the busiest link, in Gbps."""
        return self.peak_flows() * self.isl_gbps

    def busy_links(self) -> int:
        return len(self._flows)

    def contended_links(self) -> int:
        return sum(1 for v in self._flows.values() if v > 1)
