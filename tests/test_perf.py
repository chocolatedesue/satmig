"""Performance model: splits, memory, throughput, latency, freeness."""

import math
import unittest

from satmig.config import DEFAULT_MODEL, DEFAULT_SATELLITE
from satmig.perf import (
    ModelSpec,
    SatelliteSpec,
    composite_score,
    energy_freeness_s,
    instance_freeness_steps,
    max_kv_blocks,
    memory_feasible,
    min_stages_for_memory,
    stage_times_ms,
    throughput_tok_s,
    tpot_ms,
)

MODEL = ModelSpec(**DEFAULT_MODEL)
SAT = SatelliteSpec(**DEFAULT_SATELLITE)


class TestSplitting(unittest.TestCase):
    def test_layers_conserved(self):
        for p in range(1, 21):
            for k in (1, 2, 4):
                total = sum(MODEL.layers_of_stage(p, s, k) for s in range(p))
                self.assertEqual(total, MODEL.n_layers, f"P={p} k={k}")

    def test_stacking_granularity_respected(self):
        for p in range(1, 21):
            for s in range(p):
                self.assertEqual(MODEL.layers_of_stage(p, s, 4) % 4, 0)

    def test_split_is_balanced_within_one_unit(self):
        for p in range(1, 21):
            xs = [MODEL.layers_of_stage(p, s, 4) for s in range(p)]
            self.assertLessEqual(max(xs) - min(xs), 4)

    def test_bad_stage_index_rejected(self):
        with self.assertRaises(ValueError):
            MODEL.layers_of_stage(4, 4)
        with self.assertRaises(ValueError):
            MODEL.layers_of_stage(0, 0)


class TestMemory(unittest.TestCase):
    def test_pipelive_maxblocks_formula(self):
        sat = SatelliteSpec(memory_gb=24.0, memory_util_cap=0.9, kv_block_gb=0.001)
        # (24*0.9 - 4*1.75) / (4*0.001) = (21.6 - 7.0)/0.004 = 3650
        self.assertEqual(max_kv_blocks(sat, 4, 1.75), 3650)

    def test_maxblocks_zero_when_weights_do_not_fit(self):
        sat = SatelliteSpec(memory_gb=4.0)
        self.assertEqual(max_kv_blocks(sat, 40, 1.75), 0)
        self.assertEqual(max_kv_blocks(sat, 0, 1.75), 0)

    def test_min_stages_for_memory_is_ten_for_the_default_setup(self):
        """140 GB weights + 42.9 GB KV over 21.6 GB usable per satellite."""
        self.assertEqual(min_stages_for_memory(MODEL, SAT, 4), 10)

    def test_feasibility_is_monotone_in_split(self):
        p0 = min_stages_for_memory(MODEL, SAT, 4)
        self.assertFalse(memory_feasible(MODEL, SAT, p0 - 1, 4))
        for p in range(p0, 21):
            self.assertTrue(memory_feasible(MODEL, SAT, p, 4))

    def test_model_that_never_fits_raises(self):
        tiny = SatelliteSpec(memory_gb=1.0)
        with self.assertRaises(ValueError):
            min_stages_for_memory(MODEL, tiny, 1)


class TestThroughput(unittest.TestCase):
    def test_matches_the_1f1b_closed_form(self):
        """Throughput must equal M*b / ((M+P-1)*max_s t_s) with no hops."""
        for p in (1, 4, 8, 16):
            caps = [1.0] * p
            ts = stage_times_ms(MODEL, caps, 4)
            expect = MODEL.microbatches / ((MODEL.microbatches + p - 1) * max(ts) / 1e3)
            self.assertAlmostEqual(
                throughput_tok_s(MODEL, caps, [0.0] * p, 4), expect, places=6
            )

    def test_scales_like_p_over_m_plus_p(self):
        """LAB-47's P * M/(M+P-1) shape -- but only when P divides the unit count.

        With PipeLive's stacking factor k, the model has ``n_layers/k`` splittable
        units, so the clean scaling law holds exactly at divisors of 20 and is
        pessimistic elsewhere (see the granularity test below).
        """
        m = MODEL.microbatches
        base = throughput_tok_s(MODEL, [1.0] * 4, [0.0] * 4, 4)
        for p in (5, 10, 20):
            got = throughput_tok_s(MODEL, [1.0] * p, [0.0] * p, 4)
            ratio = (p * m / (m + p - 1)) / (4 * m / (m + 4 - 1))
            self.assertAlmostEqual(got / base, ratio, delta=0.02)

    def test_stacking_granularity_costs_throughput_at_non_divisors(self):
        """P=16 cannot balance 20 units of 4 layers, so it loses to the law."""
        m = MODEL.microbatches
        base = throughput_tok_s(MODEL, [1.0] * 4, [0.0] * 4, 4)
        got = throughput_tok_s(MODEL, [1.0] * 16, [0.0] * 16, 4)
        ideal = base * (16 * m / (m + 15)) / (4 * m / (m + 3))
        self.assertLess(got, ideal * 0.95)
        # ...and k=1 recovers it, because every split is then balanced.
        got_k1 = throughput_tok_s(MODEL, [1.0] * 16, [0.0] * 16, 1)
        self.assertGreater(got_k1, got)

    def test_one_dark_stage_throttles_the_whole_pipeline(self):
        """The min/max operator: a single derated stage is the bottleneck."""
        good = throughput_tok_s(MODEL, [1.0] * 10, [0.0] * 10, 4)
        caps = [1.0] * 10
        caps[7] = 0.2
        bad = throughput_tok_s(MODEL, caps, [0.0] * 10, 4)
        self.assertAlmostEqual(bad / good, 0.2, delta=0.02)

    def test_all_dark_is_the_same_as_one_dark(self):
        caps_one = [1.0] * 8
        caps_one[0] = 0.2
        self.assertAlmostEqual(
            throughput_tok_s(MODEL, caps_one, [0.0] * 8, 4),
            throughput_tok_s(MODEL, [0.2] * 8, [0.0] * 8, 4),
            delta=1e-6,
        )

    def test_zero_capacity_gives_zero_throughput(self):
        self.assertEqual(throughput_tok_s(MODEL, [1.0, 0.0], [0.0, 0.0], 4), 0.0)

    def test_downtime_fraction_scales_linearly(self):
        full = throughput_tok_s(MODEL, [1.0] * 8, [0.0] * 8, 4)
        half = throughput_tok_s(
            MODEL, [1.0] * 8, [0.0] * 8, 4, migration_downtime_fraction=0.5
        )
        self.assertAlmostEqual(half, full * 0.5, places=9)


class TestLatency(unittest.TestCase):
    def test_compute_floor_is_split_independent(self):
        """Total layer compute per token does not change with P."""
        for p in (1, 5, 10, 20):
            self.assertAlmostEqual(
                tpot_ms(MODEL, [1.0] * p, [0.0] * p, 4),
                MODEL.n_layers * MODEL.decode_ms_per_layer,
                places=6,
            )

    def test_hops_add_directly(self):
        base = tpot_ms(MODEL, [1.0] * 5, [0.0] * 5, 4)
        with_hops = tpot_ms(MODEL, [1.0] * 5, [2.0] * 5, 4)
        self.assertAlmostEqual(with_hops - base, 10.0, places=6)

    def test_derated_stage_inflates_its_own_time(self):
        caps = [1.0] * 4
        caps[1] = 0.5
        base = tpot_ms(MODEL, [1.0] * 4, [0.0] * 4, 4)
        got = tpot_ms(MODEL, caps, [0.0] * 4, 4)
        stage_ms = MODEL.layers_of_stage(4, 1, 4) * MODEL.decode_ms_per_layer
        self.assertAlmostEqual(got - base, stage_ms, places=6)

    def test_dead_stage_is_infinite(self):
        self.assertEqual(tpot_ms(MODEL, [1.0, 0.0], [0.0, 0.0], 4), math.inf)


class TestFreeness(unittest.TestCase):
    def test_llumnix_formula(self):
        self.assertAlmostEqual(instance_freeness_steps(1000, 400, 20), 30.0)
        self.assertEqual(instance_freeness_steps(1000, 400, 0), math.inf)

    def test_energy_freeness_adds_battery_runway(self):
        # 400 Wh at 300 W is 4800 s of runway.
        self.assertAlmostEqual(energy_freeness_s(100.0, 400.0, 300.0), 4900.0, places=6)

    def test_eclipse_free_host_has_infinite_freeness(self):
        self.assertEqual(energy_freeness_s(math.inf, 400.0, 300.0), math.inf)

    def test_freeness_orders_hosts_the_way_we_want(self):
        a = energy_freeness_s(10.0, 400.0, 300.0)
        b = energy_freeness_s(2000.0, 400.0, 300.0)
        self.assertLess(a, b)


class TestScore(unittest.TestCase):
    def test_pipelive_composite_score(self):
        self.assertAlmostEqual(composite_score(1.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(composite_score(0.0, 0.5, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
