"""Deterministic slot-driven simulator.

Determinism: there is no RNG on the decision path at all -- illumination is
closed form and every policy is a pure function of ``(state, t)``.  The seed in
the config only feeds optional workload jitter, and is recorded in the manifest
so that a re-run is bit-identical.

Timing model (stated plainly because it bounds every number we report):

* One slot is ``sim.slot_s`` seconds.  A handover is priced inside the slot it
  starts in; at 100 Gbps a stage's KV is a sub-second transfer, so a 10 s slot
  is coarse enough to contain it and fine enough to resolve umbra entry.
* ISL contention is per-slot max-min fair: a flow crossing a link shared by
  ``n`` concurrent flows gets ``B/n``.
* A handover misses its deadline when ``precopy + downtime`` exceeds the time
  left before its host enters the umbra.  A missed handover leaves the stage on
  the darkening satellite, whose capacity drops to ``c_eclipse`` -- and because
  throughput is a ``min`` over stages, that alone throttles the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import RunConfig
from .migration import (
    HandoverCost,
    LinkLoad,
    MigrationParams,
    stage_handover_cost,
    stop_and_copy_cost,
    transfer_ms,
)
from .perf import (
    ModelSpec,
    SatelliteSpec,
    memory_feasible,
    min_stages_for_memory,
    throughput_tok_s,
    tpot_ms,
)
from .policies import BlockMove, Pipeline, Policy, make_policy
from .topology import Constellation, SatId


@dataclass
class SlotRecord:
    t_s: float
    policy: str
    pid: int
    n_stages: int
    lit_stages: int
    min_capacity: float
    throughput_tok_s: float
    tpot_ms: float
    slo_ok: int
    hops_ms: float
    migration_events: int
    stage_moves: int
    migrated_gb: float
    downtime_ms: float
    deadline_miss: int
    peak_link_flows: int
    colocated_stages: int
    battery_wh: float


@dataclass
class PolicyResult:
    policy: str
    n_stages: int
    memory_feasible: bool
    slo_feasible_stages: int
    records: list[SlotRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Simulator:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        c = cfg.constellation
        self.con = Constellation.walker(
            altitude_km=c.altitude_km,
            inclination_deg=c.inclination_deg,
            n_planes=c.n_planes,
            n_sats_per_plane=c.n_sats_per_plane,
            epoch_days_since_j2000=c.epoch_days_since_j2000,
            raan_spread_deg=c.raan_spread_deg,
            raan_offset_deg=c.raan_offset_deg,
            cross_plane_latitude_deg=c.cross_plane_latitude_deg,
            plane_beta_deg=c.plane_beta_deg,
        )
        self.model: ModelSpec = cfg.model_spec()
        self.sat: SatelliteSpec = cfg.satellite_spec()
        self.mig: MigrationParams = cfg.migration_params()

    # ------------------------------------------------------------------ run
    def run(self) -> list[PolicyResult]:
        return [self.run_policy(name) for name in self.cfg.sim.policies]

    def run_policy(self, name: str) -> PolicyResult:
        pol = make_policy(name, self.con, self.model, self.sat, self.mig, self.cfg)
        p = pol.pipelines[0].n_stages if pol.pipelines else 0
        res = PolicyResult(
            policy=name,
            n_stages=p,
            memory_feasible=memory_feasible(
                self.model, self.sat, p, self.cfg.workload.stacking_k
            )
            if p
            else False,
            slo_feasible_stages=self.max_stages_under_slo(pol),
        )
        link = LinkLoad(self.mig.isl_gbps)
        slot = self.cfg.sim.slot_s
        t = 0.0
        while t < self.cfg.sim.horizon_s - 1e-9:
            moves = pol.decide(t)
            costs = self._price_moves(pol, moves, link, t)
            per_pipe = self._apply(pol, moves, costs, t)
            occ = self._occupancy(pol)
            for pl in pol.pipelines:
                res.records.append(
                    self._observe(pol, pl, t, per_pipe.get(pl.pid), link, occ)
                )
            t += slot
        self._annotate(pol, res)
        return res

    # -------------------------------------------------------------- pricing
    def _price_moves(
        self, pol: Policy, moves: list[BlockMove], link: LinkLoad, t_s: float
    ) -> dict[int, list[tuple[int, SatId, SatId, HandoverCost, bool]]]:
        link.reset()
        # Pass 1: declare every flow so the fair share is known.
        flows: dict[int, list[tuple[int, SatId, SatId, list[tuple]]]] = {}
        for mv in moves:
            pl = pol.pipelines[mv.pid]
            per_stage = []
            for idx, (src, dst) in enumerate(zip(pl.path, mv.new_path)):
                if src == dst:
                    continue
                links = self._route_links(src, dst)
                link.declare(links)
                per_stage.append((idx, src, dst, links))
            flows[mv.pid] = per_stage

        # Pass 2: price each flow at its bottleneck fair share.
        out: dict[int, list[tuple[int, SatId, SatId, HandoverCost, bool]]] = {}
        for mv in moves:
            pl = pol.pipelines[mv.pid]
            priced = []
            for idx, src, dst, links in flows.get(mv.pid, []):
                bw = link.share_for(links)
                kv = pol.stage_kv_gb(pl.n_stages, idx)
                already = idx in pl.resident.get(dst, set())
                weight_gb = 0.0 if already else pol.stage_weight_gb(pl.n_stages, idx)
                if pol.prestages_weights:
                    # Read-only weights stream in the background, off the
                    # critical path (PipeLive weight loader).
                    critical_weight = 0.0
                else:
                    critical_weight = weight_gb
                if pol.incremental_kv:
                    cost = stage_handover_cost(
                        kv,
                        critical_weight,
                        self.mig,
                        available_gbps=bw,
                        epoch_routing=pol.uses_epoch_routing,
                    )
                else:
                    cost = stop_and_copy_cost(
                        kv, critical_weight, self.mig, epoch_routing=pol.uses_epoch_routing
                    )
                    cost = HandoverCost(
                        bytes_gb=cost.bytes_gb,
                        precopy_ms=0.0,
                        downtime_ms=transfer_ms(kv + critical_weight, bw)
                        + self.mig.ack_ms
                        + (
                            self.mig.cutover_ms_epoch
                            if pol.uses_epoch_routing
                            else self.mig.cutover_ms_global
                        ),
                        rounds=0,
                        converged=True,
                    )
                prop_ms = self.con.manhattan_delay_ms(src, dst)
                budget_ms = max(0.0, (mv.deadline_s - t_s) * 1e3)
                need_ms = cost.total_ms + prop_ms + self.mig.ack_ms
                ok = need_ms <= budget_ms + self.cfg.sim.slot_s * 1e3
                priced.append((idx, src, dst, cost, ok))
            out[mv.pid] = priced
        return out

    def _route_links(self, a: SatId, b: SatId) -> list[tuple]:
        """Lattice route: cross-plane first, then along-track."""
        path = [a]
        q, k = a
        while q != b[0]:
            step = 1 if (b[0] - q) % self.con.n_planes <= self.con.n_planes // 2 else -1
            q = (q + step) % self.con.n_planes
            path.append((q, k))
        while k != b[1]:
            step = 1 if (b[1] - k) % self.con.n_sats <= self.con.n_sats // 2 else -1
            k = (k + step) % self.con.n_sats
            path.append((q, k))
        return self.con.links_of_path(path)

    # ---------------------------------------------------------------- apply
    def _apply(
        self,
        pol: Policy,
        moves: list[BlockMove],
        costs: dict,
        t_s: float,
    ) -> dict[int, dict]:
        summary: dict[int, dict] = {}
        for mv in moves:
            pl = pol.pipelines[mv.pid]
            priced = costs.get(mv.pid, [])
            if not priced:
                continue
            misses = sum(1 for *_x, ok in priced if not ok)
            if misses:
                pl.deadline_misses += misses
                summary[mv.pid] = {
                    "events": 0,
                    "moves": 0,
                    "gb": 0.0,
                    "downtime_ms": 0.0,
                    "misses": misses,
                }
                continue
            new_path = list(pl.path)
            gb = 0.0
            downtime = 0.0
            for idx, _src, dst, cost, _ok in priced:
                new_path[idx] = dst
                gb += cost.bytes_gb
                downtime += cost.downtime_ms
            pl.path = new_path
            pl.note_resident(new_path)
            pl.migration_events += 1
            pl.stage_moves += len(priced)
            pl.migrated_gb += gb
            pl.downtime_ms += downtime
            summary[mv.pid] = {
                "events": 1,
                "moves": len(priced),
                "gb": gb,
                "downtime_ms": downtime,
                "misses": 0,
            }
        return summary

    # -------------------------------------------------------------- observe
    def _occupancy(self, pol: Policy) -> dict[SatId, int]:
        """How many stages (from any pipeline) each satellite hosts.

        A satellite is one accelerator: two co-resident stages halve each
        other's compute and, at these model sizes, would not even fit in
        memory.  Charging co-location is what makes the lit arc a real
        capacity constraint rather than a decoration.
        """
        occ: dict[SatId, int] = {}
        for pl in pol.pipelines:
            for s in pl.path:
                occ[s] = occ.get(s, 0) + 1
        return occ

    def _observe(
        self,
        pol: Policy,
        pl: Pipeline,
        t_s: float,
        ev: dict | None,
        link: LinkLoad,
        occ: dict[SatId, int],
    ) -> SlotRecord:
        caps = []
        colocated = 0
        for s in pl.path:
            share = max(1, occ.get(s, 1))
            if share > 1:
                colocated += 1
            caps.append(self.con.capacity(s, t_s, self.cfg.sim.c_eclipse) / share)
        lit = sum(1 for s in pl.path if self.con.is_lit(s, t_s))
        fwd, ret = self.con.chain_delays_ms(pl.path)
        hops = sum(fwd) + ret
        downtime_ms = ev["downtime_ms"] if ev else 0.0
        frac = min(1.0, downtime_ms / (self.cfg.sim.slot_s * 1e3))
        tp = throughput_tok_s(
            self.model,
            caps,
            fwd + [ret],
            self.cfg.workload.stacking_k,
            migration_downtime_fraction=frac,
        )
        lat = tpot_ms(self.model, caps, fwd + [ret], self.cfg.workload.stacking_k)
        pl.slots += 1
        pl.dark_stage_slots += pl.n_stages - lit
        battery_wh = (
            (pl.n_stages - lit)
            * self.sat.compute_power_w
            * self.cfg.sim.slot_s
            / 3600.0
        )
        return SlotRecord(
            t_s=t_s,
            policy=pol.name,
            pid=pl.pid,
            n_stages=pl.n_stages,
            lit_stages=lit,
            min_capacity=min(caps),
            throughput_tok_s=tp,
            tpot_ms=lat,
            slo_ok=1 if lat <= self.cfg.workload.slo_tpot_ms else 0,
            hops_ms=hops,
            migration_events=ev["events"] if ev else 0,
            stage_moves=ev["moves"] if ev else 0,
            migrated_gb=ev["gb"] if ev else 0.0,
            downtime_ms=downtime_ms,
            deadline_miss=ev["misses"] if ev else 0,
            peak_link_flows=link.peak_flows(),
            colocated_stages=colocated,
            battery_wh=battery_wh,
        )

    # ------------------------------------------------------------ feasibility
    def max_stages_under_slo(self, pol: Policy) -> int:
        """Largest split whose *best case* TPOT still fits the SLO.

        Best case = every stage lit, block placed exactly as this policy would
        place it.  This is the number that decides whether a policy has any
        feasible operating point at all, once memory demands a lower bound.
        """
        slo = self.cfg.workload.slo_tpot_ms
        best = 0
        head, _q = pol.initial_head(0, 1)
        for p in range(1, self.con.n_sats * 2 + 1):
            try:
                path = pol.build_path(head, p)
            except ValueError:
                break
            if len(set(path)) != len(path):
                break
            fwd, ret = self.con.chain_delays_ms(path)
            lat = tpot_ms(self.model, [1.0] * p, fwd + [ret], self.cfg.workload.stacking_k)
            if lat <= slo:
                best = p
        return best

    def _annotate(self, pol: Policy, res: PolicyResult) -> None:
        try:
            p_mem = min_stages_for_memory(self.model, self.sat, self.cfg.workload.stacking_k)
        except ValueError:
            p_mem = -1
        res.notes.append(f"min_stages_for_memory={p_mem}")
        res.notes.append(f"max_stages_under_slo={res.slo_feasible_stages}")
        if p_mem > 0 and res.slo_feasible_stages < p_mem:
            res.notes.append(
                "INFEASIBLE: no split satisfies memory and the TPOT SLO at once"
            )
        res.notes.append(
            "planes_eclipse_free="
            + str(sum(1 for p in self.con.planes if p.f_ecl <= 0.0))
        )
