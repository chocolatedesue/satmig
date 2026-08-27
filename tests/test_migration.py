"""Migration cost model and the Hungarian solver it relies on."""

import math
import unittest

from satmig.matching import solve_max_reuse, solve_min_cost
from satmig.migration import (
    GB_BITS,
    LinkLoad,
    MigrationParams,
    recompute_cost_ms,
    scope_trigger_time_s,
    stage_handover_cost,
    stop_and_copy_cost,
    transfer_ms,
    value_density,
)

P = MigrationParams()


class TestTransfer(unittest.TestCase):
    def test_transfer_time_of_one_gb_at_100gbps(self):
        # 1 GiB = 8.59 Gbit; at 100 Gbps that is 85.9 ms.
        self.assertAlmostEqual(transfer_ms(1.0, 100.0), GB_BITS / 1e11 * 1e3, places=9)
        self.assertAlmostEqual(transfer_ms(1.0, 100.0), 85.9, delta=0.1)

    def test_zero_bandwidth_is_infinite(self):
        self.assertEqual(transfer_ms(1.0, 0.0), math.inf)

    def test_linear_in_bytes_and_inverse_in_rate(self):
        self.assertAlmostEqual(transfer_ms(2.0, 100.0), 2 * transfer_ms(1.0, 100.0))
        self.assertAlmostEqual(transfer_ms(1.0, 50.0), 2 * transfer_ms(1.0, 100.0))


class TestPrecopy(unittest.TestCase):
    def test_incremental_patching_converges_in_two_rounds(self):
        c = stage_handover_cost(4.29, 0.0, P)
        self.assertTrue(c.converged)
        self.assertEqual(c.rounds, 2)

    def test_downtime_is_dominated_by_cutover_not_bytes(self):
        c = stage_handover_cost(4.29, 0.0, P)
        self.assertLess(c.downtime_ms, 40.0)
        self.assertGreater(c.downtime_ms, P.cutover_ms_epoch)

    def test_stop_and_copy_is_an_order_of_magnitude_worse(self):
        inc = stage_handover_cost(4.29, 0.0, P)
        sac = stop_and_copy_cost(4.29, 0.0, P)
        self.assertGreater(sac.downtime_ms / inc.downtime_ms, 8.0)

    def test_epoch_routing_beats_global_barrier(self):
        a = stage_handover_cost(4.29, 0.0, P, epoch_routing=True)
        b = stage_handover_cost(4.29, 0.0, P, epoch_routing=False)
        self.assertAlmostEqual(
            b.downtime_ms - a.downtime_ms,
            P.cutover_ms_global - P.cutover_ms_epoch,
            places=6,
        )

    def test_high_dirty_rate_fails_to_converge(self):
        hot = MigrationParams(kv_dirty_gb_per_s=2000.0, max_precopy_rounds=4)
        c = stage_handover_cost(4.29, 0.0, hot)
        self.assertFalse(c.converged)
        self.assertEqual(c.rounds, 4)

    def test_narrow_link_inflates_precopy(self):
        wide = stage_handover_cost(4.29, 0.0, P, available_gbps=100.0)
        thin = stage_handover_cost(4.29, 0.0, P, available_gbps=10.0)
        self.assertGreater(thin.precopy_ms, wide.precopy_ms * 5)

    def test_zero_bandwidth_is_reported_not_crashed(self):
        c = stage_handover_cost(4.29, 0.0, P, available_gbps=0.0)
        self.assertFalse(c.converged)
        self.assertEqual(c.downtime_ms, math.inf)

    def test_weights_ride_the_background_stream(self):
        """Read-only weights must not lengthen the blocking window."""
        no_w = stage_handover_cost(4.29, 0.0, P)
        with_w = stage_handover_cost(4.29, 14.0, P)
        self.assertAlmostEqual(no_w.downtime_ms, with_w.downtime_ms, places=6)
        self.assertGreater(with_w.precopy_ms, no_w.precopy_ms)
        self.assertGreater(with_w.bytes_gb, no_w.bytes_gb)


class TestScopeTrigger(unittest.TestCase):
    def test_lead_time_is_transfer_plus_ack_plus_buffer(self):
        lead = scope_trigger_time_s(0.0, 4.29, P)
        expect = (transfer_ms(4.29, 100.0) + P.ack_ms + P.buffer_ms) / 1e3
        self.assertAlmostEqual(lead, expect, places=9)

    def test_lead_time_grows_with_bytes(self):
        self.assertGreater(
            scope_trigger_time_s(0.0, 40.0, P), scope_trigger_time_s(0.0, 4.0, P)
        )

    def test_lead_time_grows_when_the_link_narrows(self):
        self.assertGreater(
            scope_trigger_time_s(0.0, 4.29, P, available_gbps=10.0),
            scope_trigger_time_s(0.0, 4.29, P, available_gbps=100.0),
        )

    def test_value_density_is_scope_eq2(self):
        self.assertAlmostEqual(value_density(100.0, 4.0), 25.0)
        self.assertEqual(value_density(1.0, 0.0), math.inf)


class TestRecompute(unittest.TestCase):
    def test_llumnix_preemption_loss_scales_with_tokens(self):
        self.assertAlmostEqual(recompute_cost_ms(5.0, 80, 100), 40000.0)
        self.assertEqual(recompute_cost_ms(5.0, 80, 0), 0.0)


class TestLinkLoad(unittest.TestCase):
    def test_fair_share_halves_with_two_flows(self):
        ll = LinkLoad(100.0)
        ll.declare([("a", "b")])
        ll.declare([("a", "b")])
        self.assertAlmostEqual(ll.share_for([("a", "b")]), 50.0)

    def test_bottleneck_is_the_worst_link_on_the_route(self):
        ll = LinkLoad(100.0)
        ll.declare([("a", "b")])
        ll.declare([("b", "c")])
        ll.declare([("b", "c")])
        ll.declare([("b", "c")])
        self.assertAlmostEqual(ll.share_for([("a", "b"), ("b", "c")]), 100.0 / 3)

    def test_empty_route_gets_full_capacity(self):
        self.assertAlmostEqual(LinkLoad(100.0).share_for([]), 100.0)

    def test_reset_clears_state(self):
        ll = LinkLoad(100.0)
        ll.declare([("a", "b")])
        ll.reset()
        self.assertEqual(ll.peak_flows(), 0)
        self.assertEqual(ll.busy_links(), 0)

    def test_peak_and_contended_counters(self):
        ll = LinkLoad(100.0)
        ll.declare([("a", "b"), ("b", "c")])
        ll.declare([("b", "c")])
        self.assertEqual(ll.peak_flows(), 2)
        self.assertEqual(ll.busy_links(), 2)
        self.assertEqual(ll.contended_links(), 1)
        self.assertAlmostEqual(ll.peak_gbps(), 200.0)


class TestHungarian(unittest.TestCase):
    def test_trivial_identity(self):
        cost = [[0.0, 9.0], [9.0, 0.0]]
        self.assertEqual(solve_min_cost(cost), [0, 1])

    def test_swap_is_found(self):
        cost = [[9.0, 0.0], [0.0, 9.0]]
        self.assertEqual(solve_min_cost(cost), [1, 0])

    def test_known_optimum_3x3(self):
        cost = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]
        a = solve_min_cost(cost)
        self.assertAlmostEqual(sum(cost[i][a[i]] for i in range(3)), 5.0)

    def test_rectangular_more_columns_than_rows(self):
        cost = [[5.0, 1.0, 8.0], [7.0, 6.0, 2.0]]
        a = solve_min_cost(cost)
        self.assertEqual(len(set(a)), 2)
        self.assertAlmostEqual(sum(cost[i][a[i]] for i in range(2)), 3.0)

    def test_too_few_columns_rejected(self):
        with self.assertRaises(ValueError):
            solve_min_cost([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_empty_problem(self):
        self.assertEqual(solve_min_cost([]), [])

    def test_max_reuse_prefers_resident_hosts(self):
        reuse = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        self.assertEqual(solve_max_reuse(reuse), [0, 2, 1])

    def test_brute_force_agreement_on_random_instances(self):
        import itertools
        import random

        rng = random.Random(7)
        for _ in range(40):
            n = rng.randint(1, 5)
            cost = [[rng.uniform(0, 20) for _ in range(n)] for _ in range(n)]
            a = solve_min_cost(cost)
            got = sum(cost[i][a[i]] for i in range(n))
            best = min(
                sum(cost[i][p[i]] for i in range(n))
                for p in itertools.permutations(range(n))
            )
            self.assertAlmostEqual(got, best, places=9)

    def test_infinite_costs_are_avoided_when_possible(self):
        cost = [[math.inf, 1.0], [2.0, math.inf]]
        self.assertEqual(solve_min_cost(cost), [1, 0])


if __name__ == "__main__":
    unittest.main()
