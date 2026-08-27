"""Policy invariants.

The invariants that matter are structural, not numeric: a stage chain must stay
a valid ISL path, two pipelines must never end up on one satellite, and a
policy must never migrate a block that has nowhere safe to go.
"""

import math
import unittest

from satmig.config import RunConfig, from_dict
from satmig.policies import POLICIES, make_policy
from satmig.simulator import Simulator


def cfg_for(policies, **workload) -> RunConfig:
    raw = {
        "name": "test",
        "constellation": {
            "altitude_km": 550.0,
            "inclination_deg": 53.0,
            "n_planes": 24,
            "n_sats_per_plane": 22,
            "epoch_days_since_j2000": 9210.0,
        },
        "workload": {"n_pipelines": 1, "stages": 6, "stacking_k": 4, **workload},
        "sim": {
            "horizon_s": 5730.0,
            "slot_s": 20.0,
            "c_eclipse": 0.2,
            "policies": list(policies),
        },
    }
    return from_dict(raw)


class TestRegistry(unittest.TestCase):
    def test_every_policy_is_constructible(self):
        cfg = cfg_for(["static"])
        sim = Simulator(cfg)
        for name in POLICIES:
            pol = make_policy(name, sim.con, sim.model, sim.sat, sim.mig, cfg)
            self.assertEqual(len(pol.pipelines), 1)

    def test_unknown_policy_rejected(self):
        cfg = cfg_for(["static"])
        sim = Simulator(cfg)
        with self.assertRaises(KeyError):
            make_policy("nope", sim.con, sim.model, sim.sat, sim.mig, cfg)


class TestStructuralInvariants(unittest.TestCase):
    def _walk(self, cfg, name):
        """Run a policy and assert the invariants at every slot."""
        sim = Simulator(cfg)
        res = sim.run_policy(name)
        return sim, res

    def test_paths_stay_valid_isl_chains(self):
        for name in POLICIES:
            cfg = cfg_for([name], n_pipelines=3, stages=6, snake_width=3)
            sim = Simulator(cfg)
            pol = make_policy(name, sim.con, sim.model, sim.sat, sim.mig, cfg)
            t = 0.0
            while t < 5730.0:
                moves = pol.decide(t)
                for mv in moves:
                    path = mv.new_path
                    self.assertEqual(
                        len(set(path)), len(path), f"{name}: duplicate host in a block"
                    )
                    sim.con.chain_delays_ms(path)  # raises if not a chain
                    pol.pipelines[mv.pid].path = list(path)
                t += 20.0

    def test_no_two_pipelines_share_a_satellite_under_the_proposed_policies(self):
        for name in ("eph", "eph_compact", "eph_freeness", "two_timescale"):
            cfg = cfg_for([name], n_pipelines=4, stages=6, snake_width=3, plane_window=2)
            sim = Simulator(cfg)
            res = sim.run_policy(name)
            worst = max(r.colocated_stages for r in res.records)
            self.assertEqual(worst, 0, f"{name} allowed co-location")

    def test_static_never_migrates(self):
        cfg = cfg_for(["static"])
        sim = Simulator(cfg)
        res = sim.run_policy("static")
        self.assertEqual(sum(r.migration_events for r in res.records), 0)

    def test_conveyor_migrates_once_per_hop_period(self):
        cfg = cfg_for(["conveyor"])
        sim = Simulator(cfg)
        res = sim.run_policy("conveyor")
        events = sum(r.migration_events for r in res.records)
        expected = 5730.0 / sim.con.hop_period_s()
        self.assertAlmostEqual(events, expected, delta=1.5)

    def test_conveyor_holds_every_stage_in_sunlight(self):
        cfg = cfg_for(["conveyor"])
        sim = Simulator(cfg)
        res = sim.run_policy("conveyor")
        self.assertTrue(all(r.lit_stages == r.n_stages for r in res.records))

    def test_static_matches_the_closed_form_p_full(self):
        """LAB-47: p_full(P) = max(0, (1-f) - (P-1)/N) for a fixed block."""
        cfg = cfg_for(["static"], stages=6)
        sim = Simulator(cfg)
        res = sim.run_policy("static")
        pl_plane = sim.con.planes[0]
        expect = max(0.0, (1 - pl_plane.f_ecl) - 5 / 22)
        got = sum(1 for r in res.records if r.lit_stages == r.n_stages) / len(
            res.records
        )
        self.assertAlmostEqual(got, expect, delta=0.05)

    def test_eph_needs_fewer_events_than_conveyor(self):
        cfg = cfg_for(["conveyor", "eph"], stages=6)
        sim = Simulator(cfg)
        conv = sum(r.migration_events for r in sim.run_policy("conveyor").records)
        eph = sum(r.migration_events for r in sim.run_policy("eph").records)
        self.assertLess(eph, conv)

    def test_eph_batches_by_the_arc_slack(self):
        """slack+1 fewer events, because one jump replaces slack+1 shifts."""
        cfg = cfg_for(["conveyor", "eph"], stages=6)
        cfg.sim.horizon_s = 5730.0 * 4
        sim = Simulator(cfg)
        conv = sum(r.migration_events for r in sim.run_policy("conveyor").records)
        eph = sum(r.migration_events for r in sim.run_policy("eph").records)
        slack = sim.con.planes[0].lit_slot_count() - 6
        self.assertAlmostEqual(conv / max(1, eph), slack + 1, delta=1.6)

    def test_eclipse_free_plane_needs_no_migration_at_all(self):
        """The zero-migration regime, and the width condition it depends on.

        A compact block spans ``snake_width`` planes, so it only escapes
        migration entirely when that many *contiguous* planes are eclipse-free.
        At solstice a 24-plane 53 deg shell has exactly three.
        """
        cfg = cfg_for(["two_timescale"], stages=6, snake_width=3)
        cfg.constellation.epoch_days_since_j2000 = 9303.0  # solstice
        sim = Simulator(cfg)
        free = [q for q, p in enumerate(sim.con.planes) if p.f_ecl <= 0]
        self.assertEqual(len(free), 3)
        res = sim.run_policy("two_timescale")
        self.assertEqual(sum(r.migration_events for r in res.records), 0)
        self.assertTrue(all(r.lit_stages == r.n_stages for r in res.records))

    def test_a_block_wider_than_the_free_band_must_still_migrate(self):
        cfg = cfg_for(["two_timescale"], stages=6, snake_width=5)
        cfg.constellation.epoch_days_since_j2000 = 9303.0
        sim = Simulator(cfg)
        res = sim.run_policy("two_timescale")
        self.assertGreater(sum(r.migration_events for r in res.records), 0)

    def test_reactive_policies_pay_more_downtime(self):
        cfg = cfg_for(["reactive_jit", "eph"], stages=6)
        sim = Simulator(cfg)
        jit = sum(r.downtime_ms for r in sim.run_policy("reactive_jit").records)
        eph = sum(r.downtime_ms for r in sim.run_policy("eph").records)
        self.assertGreater(jit, eph * 10)

    def test_jump_distance_shortens_when_bandwidth_is_scarce(self):
        """A J-hop jump shares its J links among P transfers, so J must adapt."""
        wide = cfg_for(["eph"], stages=6)
        wide.migration = {"isl_gbps": 400.0}
        thin = cfg_for(["eph"], stages=6)
        thin.migration = {"isl_gbps": 1.0}
        jumps = []
        for cfg in (wide, thin):
            sim = Simulator(cfg)
            pol = make_policy("eph", sim.con, sim.model, sim.sat, sim.mig, cfg)
            jumps.append(pol.max_jump(pol.pipelines[0]))
        self.assertGreater(jumps[0], jumps[1])
        self.assertEqual(jumps[1], 1)

    def test_jump_never_exceeds_the_arc_slack(self):
        cfg = cfg_for(["eph"], stages=6)
        cfg.migration = {"isl_gbps": 10000.0}
        sim = Simulator(cfg)
        pol = make_policy("eph", sim.con, sim.model, sim.sat, sim.mig, cfg)
        pl = pol.pipelines[0]
        self.assertEqual(pol.max_jump(pl), pol.block_slack(pl) + 1)

    def test_compact_placement_lowers_tpot(self):
        cfg = cfg_for(["eph", "eph_compact"], stages=6, snake_width=3)
        sim = Simulator(cfg)
        a = sim.run_policy("eph").records[0].tpot_ms
        b = sim.run_policy("eph_compact").records[0].tpot_ms
        self.assertLess(b, a * 0.75)


class TestFeasibilityReporting(unittest.TestCase):
    def test_along_track_has_no_feasible_split_at_a_100ms_slo(self):
        """Memory wants P>=10, a 1xP arc's latency allows P<=5."""
        cfg = cfg_for(["eph"], stages=10, slo_tpot_ms=100.0, placement="along_track")
        cfg.constellation.n_planes = 72
        sim = Simulator(cfg)
        res = sim.run_policy("eph")
        self.assertEqual(res.slo_feasible_stages, 5)
        self.assertTrue(any("INFEASIBLE" in n for n in res.notes))

    def test_snake_placement_opens_a_feasible_window(self):
        cfg = cfg_for(
            ["eph_compact"], stages=10, slo_tpot_ms=100.0, snake_width=5
        )
        cfg.constellation.n_planes = 72
        sim = Simulator(cfg)
        res = sim.run_policy("eph_compact")
        self.assertGreaterEqual(res.slo_feasible_stages, 10)
        self.assertFalse(any("INFEASIBLE" in n for n in res.notes))

    def test_max_stages_under_slo_is_monotone_in_the_slo(self):
        prev = -1
        for slo in (60.0, 80.0, 100.0, 140.0):
            cfg = cfg_for(["eph_compact"], stages=6, slo_tpot_ms=slo, snake_width=3)
            sim = Simulator(cfg)
            got = sim.run_policy("eph_compact").slo_feasible_stages
            self.assertGreaterEqual(got, prev)
            prev = got


if __name__ == "__main__":
    unittest.main()
