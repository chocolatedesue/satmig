"""Placement and migration policies.

A pipeline's stages are *ordered*, so their hosts must form a path in the ISL
graph.  That makes the unit of decision a whole **block move**: you cannot
lift one stage out of a 1xP arc and drop it at the other end without breaking
the chain.  Every policy therefore returns ``BlockMove`` objects, and the
simulator prices them.

Baselines, each with the ground-side assumption our setting breaks:

``static``            no migration at all -- LAB-47 baseline A.
``reactive_jit``      SpotServe (arXiv:2311.15566).  Assumes preemption is
                      unpredictable, so the notice is short (30 s) and nothing
                      can be pre-staged: weights ride the critical path and
                      cutover is a global communicator rebuild.
``llumnix_reactive``  Llumnix (arXiv:2406.03243).  Assumes freeness is
                      *observed*.  On orbit the observation only arrives after
                      the host has already gone dark.
``conveyor``          LAB-47 baseline B: shift the block one slot every T/N,
                      unconditionally, with all pipelines on the same grid.

Proposed:

``eph``               Eclipse-aware Proactive Handover: SCOPE Eq. 3 deadline,
                      slack-batched jumps, per-pipeline staggered trigger,
                      PipeLive incremental KV patching, CONNEX epoch cutover.
``eph_compact``       + snake-block embedding, so the stage chain pays cheap
                      cross-plane hops and closes its return hop.
``eph_freeness``      + predictive energy freeness and, when the local lit arc
                      is contended, cross-plane relocation whose stage->host
                      mapping is SpotServe's KM assignment.
``two_timescale``     + outer plane selection by beta angle, re-planned only on
                      the beta-period scale (LAB-67: 66 days for a 53 deg
                      shell).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .matching import solve_min_cost
from .migration import MigrationParams, scope_trigger_time_s, transfer_ms
from .perf import ModelSpec, SatelliteSpec, energy_freeness_s
from .topology import Constellation, SatId


@dataclass
class Pipeline:
    pid: int
    n_stages: int
    path: list[SatId]
    home_plane: int = 0
    resident: dict[SatId, set[int]] = field(default_factory=dict)
    """Stage ids whose weights are already staged on a given satellite."""
    migration_events: int = 0
    stage_moves: int = 0
    migrated_gb: float = 0.0
    downtime_ms: float = 0.0
    deadline_misses: int = 0
    dark_stage_slots: int = 0
    slots: int = 0

    def note_resident(self, path: list[SatId]) -> None:
        for idx, s in enumerate(path):
            self.resident.setdefault(s, set()).add(idx)


@dataclass
class BlockMove:
    pid: int
    new_path: list[SatId]
    deadline_s: float
    priority: float
    """SCOPE Eq. 2 value density -- admission order under ISL contention."""
    reason: str = ""


class Policy:
    name = "base"
    uses_epoch_routing = True
    prestages_weights = True
    incremental_kv = True

    def __init__(
        self,
        con: Constellation,
        model: ModelSpec,
        sat: SatelliteSpec,
        mig: MigrationParams,
        cfg,
    ) -> None:
        self.con = con
        self.model = model
        self.sat = sat
        self.mig = mig
        self.cfg = cfg
        self.pipelines: list[Pipeline] = []
        self._last_shift_index = -1
        self._stagger: list[float] = []

    # ------------------------------------------------------------------ setup
    def choose_stages(self) -> int:
        w = self.cfg.workload
        if w.stages is not None:
            return w.stages
        return self.energy_stage_cap()

    def energy_stage_cap(self) -> int:
        """LAB-47's ``P* = floor(N * (1 - f_ecl))`` on the planes we may use."""
        planes = self.available_planes()
        f = max(self.con.planes[q].f_ecl for q in planes)
        return max(1, int(math.floor(self.con.n_sats * (1.0 - f))))

    def plane_order(self) -> list[int]:
        return sorted(range(self.con.n_planes), key=lambda q: self.con.planes[q].f_ecl)

    def available_planes(self) -> list[int]:
        w = self.cfg.workload
        if w.plane_window:
            return self.plane_order()[: max(1, w.plane_window)]
        if w.plane_assignment == "best_beta":
            return self.plane_order()[: max(1, w.n_pipelines)]
        return list(range(self.con.n_planes))

    def build_path(self, head: SatId, length: int) -> list[SatId]:
        w = self.cfg.workload
        if w.placement == "cross_plane":
            return self.con.cross_plane_column(head, length)
        if w.placement == "snake":
            return self.con.snake_block(head, length, w.snake_width)
        return self.con.along_track_arc(head, length)

    def _umbra_exit_slot(self, q: int, t_s: float = 0.0) -> int:
        """First satellite at or ahead of the umbra exit point.

        "Ahead" matters: picking the nearest slot can land one slot *inside*
        the umbra, which would silently keep a stage dark forever.
        """
        plane = self.con.planes[q]
        target = (plane.anti_sun + math.pi * plane.f_ecl) % (2.0 * math.pi)
        best, best_d = 0, math.inf
        for k in range(self.con.n_sats):
            d = (plane.phase(k, t_s) - target) % (2.0 * math.pi)
            if d < best_d:
                best, best_d = k, d
        return best

    def initial_head(self, pid: int, length: int) -> tuple[SatId, int]:
        order = self.available_planes()
        q = order[pid % len(order)]
        rank = pid // len(order)
        k0 = self._umbra_exit_slot(q)
        return (q, (k0 + rank * length) % self.con.n_sats), q

    def candidate_heads(self, pid: int, length: int) -> list[SatId]:
        """Ordered fallbacks for pipeline ``pid``, best first.

        A snake block occupies ``snake_width`` planes, so two pipelines whose
        home planes are adjacent would overlap.  Rather than special-casing
        each shape we just enumerate candidates and let ``setup`` take the
        first one that is actually free.
        """
        order = self.available_planes()
        out: list[SatId] = []
        primary, _ = self.initial_head(pid, length)
        out.append(primary)
        for q in order:
            k0 = self._umbra_exit_slot(q)
            for shift in range(self.con.n_sats):
                out.append((q, (k0 + shift) % self.con.n_sats))
        return out

    def setup(self) -> None:
        p = self.choose_stages()
        n = self.cfg.workload.n_pipelines
        taken: set[SatId] = set()
        for pid in range(n):
            path: list[SatId] | None = None
            home = 0
            for head in self.candidate_heads(pid, p):
                cand = self.build_path(head, p)
                if len(set(cand)) != len(cand):
                    continue
                if taken.intersection(cand):
                    continue
                path, home = cand, head[0]
                break
            if path is None:
                # Over-subscribed: fall back to the preferred head and let the
                # co-location penalty show up in the metrics.
                head, home = self.initial_head(pid, p)
                path = self.build_path(head, p)
            taken.update(path)
            pl = Pipeline(pid=pid, n_stages=p, path=list(path), home_plane=home)
            pl.note_resident(path)
            self.pipelines.append(pl)
        self._stagger = [self.con.hop_period_s() * pid / max(1, n) for pid in range(n)]
        # The first conveyor tick belongs to t=0, which has no drift to correct
        # yet; shifting there would push the block one slot into the umbra.
        self._last_shift_index = 0

    # ------------------------------------------------------------- per-slot
    def decide(self, t_s: float) -> list[BlockMove]:
        return []

    # ------------------------------------------------------------- helpers
    def stage_kv_gb(self, n_stages: int, stage_idx: int) -> float:
        return (
            self.model.kv_gb_full
            * self.model.layers_of_stage(n_stages, stage_idx, self.cfg.workload.stacking_k)
            / self.model.n_layers
        )

    def stage_weight_gb(self, n_stages: int, stage_idx: int) -> float:
        return (
            self.model.weight_gb
            * self.model.layers_of_stage(n_stages, stage_idx, self.cfg.workload.stacking_k)
            / self.model.n_layers
        )

    def block_kv_gb(self, pl: Pipeline) -> float:
        return sum(self.stage_kv_gb(pl.n_stages, i) for i in range(pl.n_stages))

    def shift_slots(self, path: list[SatId], delta: int) -> list[SatId]:
        return [(q, (k + delta) % self.con.n_sats) for (q, k) in path]

    def lit_arc_width(self, q: int) -> int:
        return self.con.planes[q].lit_slot_count()

    def block_slack(self, pl: Pipeline) -> int:
        """Free lit slots ahead of the block in its own plane."""
        q = pl.path[0][0]
        span = len({k for (_, k) in pl.path})
        return max(0, self.lit_arc_width(q) - span)

    def min_time_to_eclipse(self, pl: Pipeline, t_s: float) -> float:
        return min(self.con.time_to_eclipse_s(s, t_s) for s in pl.path)


class StaticPolicy(Policy):
    name = "static"


class ConveyorPolicy(Policy):
    name = "conveyor"
    incremental_kv = False

    def decide(self, t_s: float) -> list[BlockMove]:
        step = self.con.hop_period_s()
        idx = int(math.floor(t_s / step))
        if idx <= self._last_shift_index:
            return []
        self._last_shift_index = idx
        return [
            BlockMove(
                pid=pl.pid,
                new_path=self.shift_slots(pl.path, -1),
                deadline_s=t_s + step,
                priority=1.0,
                reason="periodic",
            )
            for pl in self.pipelines
        ]


class ReactiveJitPolicy(Policy):
    name = "reactive_jit"
    uses_epoch_routing = False
    prestages_weights = False
    incremental_kv = False
    grace_s = 30.0

    def decide(self, t_s: float) -> list[BlockMove]:
        out: list[BlockMove] = []
        for pl in self.pipelines:
            tte = self.min_time_to_eclipse(pl, t_s)
            if tte > self.grace_s:
                continue
            out.append(
                BlockMove(
                    pid=pl.pid,
                    new_path=self.shift_slots(pl.path, -1),
                    deadline_s=t_s + max(tte, 0.0),
                    priority=1.0,
                    reason="grace_notice",
                )
            )
        return out


class LlumnixReactivePolicy(Policy):
    name = "llumnix_reactive"
    incremental_kv = True

    def decide(self, t_s: float) -> list[BlockMove]:
        out: list[BlockMove] = []
        for pl in self.pipelines:
            dark = [s for s in pl.path if not self.con.is_lit(s, t_s)]
            if not dark:
                continue
            # Freeness only turned bad now, so react now: jump back far enough
            # to clear every dark host.
            need = 1
            for s in dark:
                need = max(need, self._slots_since_entry(s, t_s) + 1)
            out.append(
                BlockMove(
                    pid=pl.pid,
                    new_path=self.shift_slots(pl.path, -need),
                    deadline_s=t_s + self.cfg.sim.slot_s,
                    priority=1.0,
                    reason="observed_freeness",
                )
            )
        return out

    def _slots_since_entry(self, sat: SatId, t_s: float) -> int:
        plane = self.con.planes[sat[0]]
        dt = plane.eclipse_duration_s() - plane.next_eclipse_exit_s(sat[1], t_s)
        return int(math.ceil(dt / self.con.hop_period_s()))


class EphPolicy(Policy):
    """Proposed baseline of the family: deadline-driven, slack-batched.

    Trigger:  SCOPE Eq. 3 on the *earliest* stage deadline in the block.
    Distance: jump the whole block back by its slack, so that its leading edge
              lands at the freshly-lit end of the arc.  ``slack`` fewer
              handover events per orbit than conveyor, each costing one
              cutover instead of ``slack`` cutovers.
    Stagger:  pipeline k's trigger threshold is offset by ``k*(T/N)/K`` so the
              K pipelines do not fire in the same slot.
    Safety:   a batched jump is *not* collision-free the way a lockstep
              conveyor shift is -- two blocks that re-time independently can
              land on each other.  So the jump distance is shortened until the
              landing zone is free.  Skipping this is a correctness bug, not a
              tuning knob (see tests/test_policies.py).
    """

    name = "eph"
    incremental_kv = True

    def lead_time_s(self, pl: Pipeline) -> float:
        worst_kv = max(self.stage_kv_gb(pl.n_stages, i) for i in range(pl.n_stages))
        return scope_trigger_time_s(0.0, worst_kv, self.mig)

    def max_jump(self, pl: Pipeline) -> int:
        """Largest batched jump that still meets its own SCOPE Eq. 3 deadline.

        Batching is not free.  A ``J``-slot jump of a ``P``-stage block routes
        ``P`` KV transfers over the same ``J`` links, so each transfer only gets
        ``B / min(J, P)`` of the link.  Above ~25 Gbps the slack cap binds and
        this reduces to ``slack + 1``; below it, bandwidth binds and the jump
        shortens instead of missing the deadline.
        """
        slack_cap = max(1, self.block_slack(pl) + 1)
        kv = max(self.stage_kv_gb(pl.n_stages, i) for i in range(pl.n_stages))
        budget_ms = (self.lead_time_s(pl) + self.cfg.sim.slot_s) * 1e3
        cutover = (
            self.mig.cutover_ms_epoch
            if self.uses_epoch_routing
            else self.mig.cutover_ms_global
        )
        hop_ms = self.con.hop_delays.along_ms
        for j in range(slack_cap, 0, -1):
            share = self.mig.isl_gbps / max(1, min(j, pl.n_stages))
            need = (
                transfer_ms(kv, share) + self.mig.ack_ms + cutover + j * hop_ms
            )
            if need <= budget_ms:
                return j
        return 1

    def decide(self, t_s: float) -> list[BlockMove]:
        out: list[BlockMove] = []
        slot = self.cfg.sim.slot_s
        others: dict[SatId, int] = {}
        for pl in self.pipelines:
            for s in pl.path:
                others[s] = pl.pid
        claimed: set[SatId] = set()
        due: list[tuple[float, Pipeline]] = []
        for pl in self.pipelines:
            tte = self.min_time_to_eclipse(pl, t_s)
            if math.isinf(tte):
                continue
            thresh = self.lead_time_s(pl) + self._stagger[pl.pid]
            if tte > thresh + slot:
                continue
            due.append((tte, pl))
        due.sort(key=lambda x: (x[0], x[1].pid))  # earliest deadline first
        for tte, pl in due:
            path = self._safe_landing(pl, others, claimed, t_s)
            if path is None:
                continue
            claimed.update(path)
            out.append(
                BlockMove(
                    pid=pl.pid,
                    new_path=path,
                    deadline_s=t_s + max(tte, 0.0),
                    priority=1.0 / max(self.block_kv_gb(pl), 1e-9),
                    reason="scope_deadline",
                )
            )
        return out

    def _free(
        self,
        pl: Pipeline,
        path: list[SatId],
        others: dict[SatId, int],
        claimed: set[SatId],
    ) -> bool:
        if len(set(path)) != len(path):
            return False
        for s in path:
            if s in claimed:
                return False
            owner = others.get(s)
            if owner is not None and owner != pl.pid:
                return False
        return True

    def _safe_landing(
        self,
        pl: Pipeline,
        others: dict[SatId, int],
        claimed: set[SatId],
        t_s: float,
    ) -> list[SatId] | None:
        for j in range(self.max_jump(pl), 0, -1):
            cand = self.shift_slots(pl.path, -j)
            if self._free(pl, cand, others, claimed):
                return cand
        return None


class EphCompactPolicy(EphPolicy):
    name = "eph_compact"

    def build_path(self, head: SatId, length: int) -> list[SatId]:
        return self.con.snake_block(head, length, self.cfg.workload.snake_width)


class EphFreenessPolicy(EphCompactPolicy):
    """Adds predictive energy freeness and cross-plane relocation.

    ``eph`` only ever slides a block backwards inside its own plane, so when a
    neighbour occupies the landing zone it has to settle for a shorter jump --
    and shorter jumps mean more handovers.  Llumnix would pick the destination
    with the largest *observed* memory freeness; we pick the one with the
    largest *predicted* energy freeness (time to umbra plus battery runway) and
    allow the block to change plane.  The stage->host mapping inside the chosen
    landing zone is then SpotServe's Device Mapper: a min-cost bipartite
    matching that keeps a stage on a satellite that already holds its weights.
    """

    name = "eph_freeness"

    def _freeness(self, path: list[SatId], t_s: float) -> float:
        vals = []
        for s in path:
            tte = self.con.time_to_eclipse_s(s, t_s)
            vals.append(
                energy_freeness_s(
                    self.con.period_s if math.isinf(tte) else tte,
                    self.sat.battery_wh,
                    self.sat.compute_power_w,
                )
            )
        return min(vals)

    def _safe_landing(
        self,
        pl: Pipeline,
        others: dict[SatId, int],
        claimed: set[SatId],
        t_s: float,
    ) -> list[SatId] | None:
        cands: list[list[SatId]] = []
        for dq in (0, -1, 1, -2, 2):
            for j in range(self.max_jump(pl), 0, -1):
                cand = [
                    ((q + dq) % self.con.n_planes, (k - j) % self.con.n_sats)
                    for (q, k) in pl.path
                ]
                if self._free(pl, cand, others, claimed):
                    cands.append(cand)
                    break
        if not cands:
            return None
        best = max(cands, key=lambda p: (self._freeness(p, t_s), -self._move_cost(pl, p)))
        return self._km_reorder(pl, best)

    def _move_cost(self, pl: Pipeline, path: list[SatId]) -> float:
        return sum(
            0.0
            if idx in pl.resident.get(h, set())
            else transfer_ms(self.stage_weight_gb(pl.n_stages, idx), self.mig.isl_gbps)
            for idx, h in enumerate(path)
        )

    def _km_reorder(self, pl: Pipeline, hosts: list[SatId]) -> list[SatId]:
        """SpotServe Device Mapper: keep stages where their weights already are."""
        n = pl.n_stages
        cost: list[list[float]] = []
        for stage in range(n):
            row = []
            wgb = self.stage_weight_gb(n, stage)
            for h in hosts:
                reuse = stage in pl.resident.get(h, set())
                row.append(0.0 if reuse else transfer_ms(wgb, self.mig.isl_gbps))
            cost.append(row)
        assign = solve_min_cost(cost)
        reordered = [hosts[assign[s]] for s in range(n)]
        try:
            self.con.chain_delays_ms(reordered)
        except ValueError:
            return hosts
        return reordered


class TwoTimescalePolicy(EphFreenessPolicy):
    """Outer loop picks planes by beta angle; inner loop is EPH.

    LAB-67: inside one shell the per-plane eclipse fraction spans 0..37% and
    changes on a beta-period scale (66 days at 53 deg), so plane choice is a
    slow variable.  When a plane's |beta| exceeds the critical angle it never
    eclipses and the pipeline on it never needs to migrate at all.

    A compact block spans ``snake_width`` planes, so the outer loop must rank
    *bands* of planes by their worst member, not single planes.  Ranking single
    planes is the trap: it puts the best plane at the edge of a band whose other
    members eclipse, and the min-over-stages operator then throws the advantage
    away.
    """

    name = "two_timescale"

    def band_width(self) -> int:
        w = self.cfg.workload
        if w.placement == "snake" or self.name in ("eph_compact", "eph_freeness", "two_timescale"):
            return max(1, w.snake_width)
        return 1

    def plane_order(self) -> list[int]:
        width = self.band_width()
        n = self.con.n_planes

        def band_cost(q: int) -> tuple[float, float]:
            members = [(q + j) % n for j in range(width)]
            fs = [self.con.planes[m].f_ecl for m in members]
            return (max(fs), sum(fs))

        return sorted(range(n), key=band_cost)

    def available_planes(self) -> list[int]:
        w = self.cfg.workload
        order = self.plane_order()
        if w.plane_window:
            return order[: max(1, w.plane_window)]
        return order[: max(1, w.n_pipelines)]


POLICIES: dict[str, type[Policy]] = {
    c.name: c
    for c in (
        StaticPolicy,
        ConveyorPolicy,
        ReactiveJitPolicy,
        LlumnixReactivePolicy,
        EphPolicy,
        EphCompactPolicy,
        EphFreenessPolicy,
        TwoTimescalePolicy,
    )
}


def make_policy(
    name: str,
    con: Constellation,
    model: ModelSpec,
    sat: SatelliteSpec,
    mig: MigrationParams,
    cfg,
) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    p = POLICIES[name](con, model, sat, mig, cfg)
    p.setup()
    return p
