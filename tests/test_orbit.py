"""Orbital geometry, checked against numbers verified elsewhere in the lab.

The reference values come from LAB-47 / LAB-58 / LAB-67, which derived them
with satellite.js + SGP4 rather than the closed forms used here.  Agreement
between two independent derivations is the point of these tests.
"""

import math
import unittest

from satmig.orbit import (
    PlaneIllumination,
    beta_angle_deg,
    beta_critical_deg,
    eclipse_fraction,
    nodal_precession_deg_per_day,
    orbital_period_s,
    sun_direction,
    umbra_center_arg_lat_rad,
)


class TestClosedForms(unittest.TestCase):
    def test_period_at_550km(self):
        self.assertAlmostEqual(orbital_period_s(550.0) / 60.0, 95.5, delta=0.05)

    def test_beta_critical_matches_lab47(self):
        self.assertAlmostEqual(beta_critical_deg(550.0), 67.0, delta=0.01)

    def test_eclipse_fraction_at_beta_zero_matches_lab67(self):
        self.assertAlmostEqual(eclipse_fraction(550.0, 0.0), 0.372, delta=0.001)

    def test_eclipse_fraction_monotone_in_beta(self):
        fs = [eclipse_fraction(550.0, b) for b in range(0, 68, 4)]
        for a, b in zip(fs, fs[1:]):
            self.assertGreaterEqual(a + 1e-12, b)

    def test_no_eclipse_above_critical(self):
        crit = beta_critical_deg(550.0)
        self.assertEqual(eclipse_fraction(550.0, crit + 0.5), 0.0)
        self.assertGreater(eclipse_fraction(550.0, crit - 0.5), 0.0)

    def test_higher_altitude_shrinks_eclipse(self):
        self.assertLess(
            eclipse_fraction(1200.0, 0.0), eclipse_fraction(550.0, 0.0)
        )

    def test_sso_precession_near_sun_rate(self):
        """97.6 deg at 550 km should precess at about +0.9856 deg/day."""
        rate = nodal_precession_deg_per_day(550.0, 97.6)
        self.assertAlmostEqual(rate, 0.9856, delta=0.06)


class TestSunAndBeta(unittest.TestCase):
    def test_solstice_declination(self):
        # ~2025-06-21 is day 9303 after J2000.
        dec = sun_direction(9303.0).dec_deg
        self.assertAlmostEqual(dec, 23.4, delta=0.2)

    def test_equinox_declination_near_zero(self):
        # ~2025-03-20 is day 9210 after J2000.
        self.assertLess(abs(sun_direction(9210.0).dec_deg), 1.5)

    def test_max_beta_is_inclination_plus_declination(self):
        sun = sun_direction(9303.0)
        best = max(
            beta_angle_deg(53.0, raan, sun) for raan in [x * 0.25 for x in range(1440)]
        )
        self.assertAlmostEqual(best, 53.0 + sun.dec_deg, delta=0.2)

    def test_beta_spread_within_one_shell(self):
        """LAB-67: one shell holds eclipse-free and heavily eclipsed planes."""
        sun = sun_direction(9303.0)
        fs = [
            eclipse_fraction(550.0, beta_angle_deg(53.0, 360.0 * q / 72, sun))
            for q in range(72)
        ]
        self.assertEqual(min(fs), 0.0)
        self.assertGreater(max(fs), 0.35)

    def test_umbra_centre_is_opposite_the_sun(self):
        sun = sun_direction(9303.0)
        # In an equatorial plane the umbra centre must be 180 deg from the
        # sub-solar argument of latitude.
        u = umbra_center_arg_lat_rad(0.0, 0.0, sun)
        self.assertAlmostEqual(
            math.degrees(u) % 360.0, (sun.ra_deg + 180.0) % 360.0, delta=1.5
        )


class TestPlaneIllumination(unittest.TestCase):
    def setUp(self):
        self.p = PlaneIllumination(550.0, 0.0, 22)

    def test_eclipse_fraction_matches_measured_dark_time(self):
        n = 20000
        dark = sum(
            1 for i in range(n) if not self.p.is_lit(0, self.p.period_s * i / n)
        )
        self.assertAlmostEqual(dark / n, self.p.f_ecl, delta=0.002)

    def test_periodicity(self):
        for t in (0.0, 137.0, 999.0, 4000.0):
            self.assertEqual(
                self.p.is_lit(3, t), self.p.is_lit(3, t + self.p.period_s)
            )

    def test_entry_prediction_is_the_transition(self):
        t = 0.0
        while not self.p.is_lit(0, t):
            t += 1.0
        dt = self.p.next_eclipse_entry_s(0, t)
        self.assertTrue(self.p.is_lit(0, t + dt - 2.0))
        self.assertFalse(self.p.is_lit(0, t + dt + 2.0))

    def test_exit_prediction_is_the_transition(self):
        t = 0.0
        while self.p.is_lit(0, t):
            t += 1.0
        dt = self.p.next_eclipse_exit_s(0, t)
        self.assertFalse(self.p.is_lit(0, t + dt - 2.0))
        self.assertTrue(self.p.is_lit(0, t + dt + 2.0))

    def test_lit_slot_count_matches_lab47_arc_width(self):
        """floor(N*(1-f)) is LAB-47's usable arc; f=0.34 -> 14 of 22."""
        p = PlaneIllumination(550.0, 0.0, 22)
        expected = int(math.floor(22 * (1 - p.f_ecl)))
        self.assertEqual(p.lit_slot_count(), expected)
        self.assertEqual(expected, 13)

    def test_eclipse_free_plane_never_dark(self):
        p = PlaneIllumination(550.0, 75.0, 22)
        self.assertEqual(p.f_ecl, 0.0)
        self.assertTrue(all(p.is_lit(k, 123.0) for k in range(22)))
        self.assertEqual(p.next_eclipse_entry_s(0, 0.0), math.inf)

    def test_neighbours_are_one_hop_period_apart_in_phase(self):
        p = self.p
        t0 = 0.0
        e0 = p.next_eclipse_entry_s(0, t0)
        e1 = p.next_eclipse_entry_s(1, t0)
        step = p.period_s / p.n_sats
        self.assertAlmostEqual((e0 - e1) % p.period_s, step, delta=1.0)


if __name__ == "__main__":
    unittest.main()
