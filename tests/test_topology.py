"""Topology: hop delays, block shapes, and the return-hop correction."""

import math
import unittest

from satmig.orbit import C_LIGHT_KM_S, R_EARTH_KM
from satmig.topology import Constellation, chord_km


def shell(n_planes=72, n_sats=22, **kw):
    return Constellation.walker(
        550.0, 53.0, n_planes, n_sats, epoch_days_since_j2000=9303.0, **kw
    )


class TestHopGeometry(unittest.TestCase):
    def setUp(self):
        self.c = shell()

    def test_along_track_hop_matches_hand_calculation(self):
        r = R_EARTH_KM + 550.0
        expect = chord_km(r, 2 * math.pi / 22) / C_LIGHT_KM_S * 1e3
        self.assertAlmostEqual(self.c.hop_delays.along_ms, expect, places=9)
        self.assertAlmostEqual(self.c.hop_delays.along_ms, 6.571, delta=0.01)

    def test_cross_plane_hop_is_much_cheaper_in_a_dense_shell(self):
        d = self.c.hop_delays
        self.assertAlmostEqual(d.cross_ms, 2.014, delta=0.01)
        self.assertGreater(d.along_ms / d.cross_ms, 3.0)

    def test_sparse_shell_inverts_the_advantage(self):
        d = shell(n_planes=12).hop_delays
        self.assertGreater(d.cross_ms, d.along_ms)

    def test_cross_plane_hop_shrinks_with_latitude(self):
        eq = shell().hop_delays.cross_ms
        hi = shell(cross_plane_latitude_deg=50.0).hop_delays.cross_ms
        self.assertLess(hi, eq)

    def test_hop_period_is_period_over_n(self):
        self.assertAlmostEqual(
            self.c.hop_period_s(), self.c.period_s / 22, places=9
        )

    def test_non_neighbours_are_not_links(self):
        with self.assertRaises(ValueError):
            self.c.hop_delay_ms((0, 0), (0, 5))
        with self.assertRaises(ValueError):
            self.c.hop_delay_ms((0, 0), (3, 1))

    def test_ring_wraparound_is_a_link(self):
        self.assertAlmostEqual(
            self.c.hop_delay_ms((0, 21), (0, 0)), self.c.hop_delays.along_ms
        )
        self.assertAlmostEqual(
            self.c.hop_delay_ms((71, 4), (0, 4)), self.c.hop_delays.cross_ms
        )


class TestBlockShapes(unittest.TestCase):
    def setUp(self):
        self.c = shell()

    def _assert_valid_chain(self, path):
        self.assertEqual(len(set(path)), len(path), f"duplicate hosts in {path}")
        for a, b in zip(path, path[1:]):
            self.c.hop_delay_ms(a, b)  # raises if not adjacent

    def test_along_track_arc_is_a_chain(self):
        self._assert_valid_chain(self.c.along_track_arc((3, 5), 8))

    def test_cross_plane_column_is_a_chain(self):
        self._assert_valid_chain(self.c.cross_plane_column((3, 5), 8))

    def test_snake_block_is_a_chain_for_many_shapes(self):
        for length in range(1, 25):
            for width in range(1, 7):
                if math.ceil(length / width) > self.c.n_sats:
                    continue
                self._assert_valid_chain(self.c.snake_block((2, 7), length, width))

    def test_snake_with_width_one_is_an_arc(self):
        self.assertEqual(
            self.c.snake_block((0, 0), 5, 1), self.c.along_track_arc((0, 0), 5)
        )

    def test_snake_rejects_bad_width(self):
        with self.assertRaises(ValueError):
            self.c.snake_block((0, 0), 4, 0)
        with self.assertRaises(ValueError):
            self.c.snake_block((0, 0), 4, self.c.n_planes + 1)

    def test_snake_rejects_a_block_that_wraps_the_ring(self):
        """Silently wrapping would put two stages on one satellite."""
        with self.assertRaises(ValueError):
            self.c.snake_block((0, 0), self.c.n_sats + 1, 1)


class TestChainDelays(unittest.TestCase):
    def setUp(self):
        self.c = shell()

    def test_along_track_pays_the_return_trip_twice(self):
        """The correction that changes LAB-47's conclusion.

        A 1xP arc's return hop is P-1 hops, not 1, so the per-token hop budget
        is 2*(P-1) along-track hops.
        """
        p = 10
        path = self.c.along_track_arc((0, 0), p)
        fwd, ret = self.c.chain_delays_ms(path)
        along = self.c.hop_delays.along_ms
        self.assertAlmostEqual(sum(fwd), (p - 1) * along, places=6)
        self.assertAlmostEqual(ret, (p - 1) * along, places=6)

    def test_even_column_snake_closes_its_return_hop(self):
        """An even number of columns lands the last stage beside the first."""
        path = self.c.snake_block((0, 0), 10, 5)
        fwd, ret = self.c.chain_delays_ms(path)
        self.assertAlmostEqual(ret, self.c.hop_delays.along_ms, places=6)
        self.assertLess(sum(fwd) + ret, 10 * self.c.hop_delays.along_ms)

    def test_snake_beats_along_track_on_total_hop_budget(self):
        for p in (6, 10, 14):
            w = p // 2
            a_fwd, a_ret = self.c.chain_delays_ms(self.c.along_track_arc((0, 0), p))
            s_fwd, s_ret = self.c.chain_delays_ms(self.c.snake_block((0, 0), p, w))
            self.assertLess(sum(s_fwd) + s_ret, sum(a_fwd) + a_ret)

    def test_single_stage_has_no_hops(self):
        fwd, ret = self.c.chain_delays_ms([(0, 0)])
        self.assertEqual(fwd, [])
        self.assertEqual(ret, 0.0)

    def test_manhattan_is_symmetric_and_zero_on_self(self):
        self.assertEqual(self.c.manhattan_delay_ms((1, 2), (1, 2)), 0.0)
        self.assertAlmostEqual(
            self.c.manhattan_delay_ms((1, 2), (4, 9)),
            self.c.manhattan_delay_ms((4, 9), (1, 2)),
        )

    def test_manhattan_takes_the_short_way_round(self):
        # 21 -> 0 is one hop, not 21.
        self.assertAlmostEqual(
            self.c.manhattan_delay_ms((0, 21), (0, 0)), self.c.hop_delays.along_ms
        )


class TestIllumination(unittest.TestCase):
    def test_plane_eclipse_fractions_are_heterogeneous(self):
        c = shell()
        fs = [p.f_ecl for p in c.planes]
        self.assertEqual(min(fs), 0.0)
        self.assertGreater(max(fs), 0.35)

    def test_beta_override_is_honoured(self):
        c = shell(plane_beta_deg=[0.0] * 72)
        self.assertTrue(all(abs(b) < 1e-9 for b in c.plane_beta_deg))
        self.assertTrue(all(abs(p.f_ecl - 0.372) < 0.001 for p in c.planes))

    def test_lit_count_is_bounded_by_fleet_size(self):
        c = shell()
        n = c.lit_count(1234.0)
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 72 * 22)

    def test_bad_beta_list_length_rejected(self):
        with self.assertRaises(ValueError):
            Constellation(550.0, 4, 22, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
