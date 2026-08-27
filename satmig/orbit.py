"""Deterministic orbital illumination model.

Everything here is closed-form and side-effect free so that it can be unit
tested against the numbers already independently verified in this workspace
(LAB-47 / LAB-58 / LAB-67):

* no-eclipse critical beta angle at 550 km  -> 66.99 deg
* eclipse fraction of a beta = 0 orbit      -> 0.372
* orbital period at 550 km                  -> 5729 s (95.5 min)
* max reachable |beta| for a 53 deg shell   -> 76.44 deg (i + obliquity)

Shadow model: cylindrical umbra of a spherical Earth.  Penumbra, atmosphere
and Earth oblateness are deliberately not modelled -- see docs/MODEL.md for
the credibility tiering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R_EARTH_KM = 6371.0
MU_EARTH = 398600.4418  # km^3 / s^2
C_LIGHT_KM_S = 299792.458
DEG = math.pi / 180.0


def orbital_period_s(altitude_km: float) -> float:
    """Keplerian period of a circular orbit."""
    r = R_EARTH_KM + altitude_km
    return 2.0 * math.pi * math.sqrt(r**3 / MU_EARTH)


def beta_critical_deg(altitude_km: float) -> float:
    """Smallest |beta| for which the orbit never enters the umbra.

    Geometry: the shadow cylinder has radius R_EARTH, so a circular orbit of
    radius r misses it entirely iff r*sin|beta| >= R_EARTH.
    """
    r = R_EARTH_KM + altitude_km
    return math.degrees(math.asin(min(1.0, R_EARTH_KM / r)))


def eclipse_fraction(altitude_km: float, beta_deg: float) -> float:
    """Fraction of one revolution spent in the umbra.

    f(beta) = (1/pi) * arccos( sqrt(1 - (R/r)^2) / cos(beta) ), clipped to 0
    once |beta| >= beta_crit.
    """
    r = R_EARTH_KM + altitude_km
    ratio = R_EARTH_KM / r
    num = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    cb = math.cos(beta_deg * DEG)
    if abs(cb) < 1e-12:
        return 0.0
    arg = num / abs(cb)
    if arg >= 1.0:
        return 0.0
    return math.acos(arg) / math.pi


# --------------------------------------------------------------------------
# Low precision sun position (Vallado); same model family as satellite.js,
# which LAB-67 used.  Valid 1950-2050 to ~0.01 deg.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SunDirection:
    ra_deg: float
    dec_deg: float


def sun_direction(days_since_j2000: float) -> SunDirection:
    d = days_since_j2000
    mean_long = 280.460 + 0.9856474 * d
    mean_anom = (357.528 + 0.9856003 * d) * DEG
    ecl_long = (
        mean_long + 1.915 * math.sin(mean_anom) + 0.020 * math.sin(2 * mean_anom)
    ) * DEG
    obliquity = (23.439 - 4.0e-7 * d) * DEG
    ra = math.atan2(math.cos(obliquity) * math.sin(ecl_long), math.cos(ecl_long))
    dec = math.asin(math.sin(obliquity) * math.sin(ecl_long))
    return SunDirection(math.degrees(ra) % 360.0, math.degrees(dec))


def beta_angle_deg(inclination_deg: float, raan_deg: float, sun: SunDirection) -> float:
    """Sun beta angle of an orbital plane.

    sin(beta) = cos(dec)*sin(i)*sin(raan - ra) + sin(dec)*cos(i)
    """
    i = inclination_deg * DEG
    dra = (raan_deg - sun.ra_deg) * DEG
    dec = sun.dec_deg * DEG
    s = math.cos(dec) * math.sin(i) * math.sin(dra) + math.sin(dec) * math.cos(i)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def plane_basis(inclination_deg: float, raan_deg: float):
    """Orthonormal in-plane basis (node direction, 90 deg ahead) plus normal."""
    i = inclination_deg * DEG
    om = raan_deg * DEG
    p_hat = (math.cos(om), math.sin(om), 0.0)
    q_hat = (
        -math.cos(i) * math.sin(om),
        math.cos(i) * math.cos(om),
        math.sin(i),
    )
    h_hat = (
        math.sin(i) * math.sin(om),
        -math.sin(i) * math.cos(om),
        math.cos(i),
    )
    return p_hat, q_hat, h_hat


def sun_unit_vector(sun: SunDirection) -> tuple[float, float, float]:
    a = sun.ra_deg * DEG
    d = sun.dec_deg * DEG
    return (math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d))


def umbra_center_arg_lat_rad(
    inclination_deg: float, raan_deg: float, sun: SunDirection
) -> float:
    """Argument of latitude of the umbra centre for this plane.

    The umbra sits on the anti-sun side, so we project ``-s_hat`` onto the
    orbital plane basis.  This is what makes two satellites in *different*
    planes eclipse at different times even at the same slot index.
    """
    p_hat, q_hat, _ = plane_basis(inclination_deg, raan_deg)
    s = sun_unit_vector(sun)
    a_p = -(s[0] * p_hat[0] + s[1] * p_hat[1] + s[2] * p_hat[2])
    a_q = -(s[0] * q_hat[0] + s[1] * q_hat[1] + s[2] * q_hat[2])
    return math.atan2(a_q, a_p) % (2.0 * math.pi)


def nodal_precession_deg_per_day(altitude_km: float, inclination_deg: float) -> float:
    """J2 nodal regression rate, deg/day (negative for prograde orbits)."""
    j2 = 1.08262668e-3
    r = R_EARTH_KM + altitude_km
    n = math.sqrt(MU_EARTH / r**3)  # rad/s
    rate = -1.5 * j2 * n * (R_EARTH_KM / r) ** 2 * math.cos(inclination_deg * DEG)
    return math.degrees(rate) * 86400.0


# --------------------------------------------------------------------------
# Per-satellite binary illumination
# --------------------------------------------------------------------------


class PlaneIllumination:
    """Illumination oracle for one orbital plane.

    The umbra occupies a contiguous arc of angular half width ``pi * f`` about
    the anti-sun point.  Satellite ``k`` of ``n_sats`` sits at in-plane phase
    ``u0 + 2*pi*t/T + 2*pi*k/n_sats``, so the shadow arc is fixed (over one
    day) and satellites sweep through it -- the "conveyor" picture of LAB-47.
    """

    def __init__(
        self,
        altitude_km: float,
        beta_deg: float,
        n_sats: int,
        phase0_rad: float = 0.0,
        anti_sun_phase_rad: float = math.pi,
    ) -> None:
        self.altitude_km = altitude_km
        self.beta_deg = beta_deg
        self.n_sats = n_sats
        self.period_s = orbital_period_s(altitude_km)
        self.f_ecl = eclipse_fraction(altitude_km, beta_deg)
        self.phase0 = phase0_rad
        self.anti_sun = anti_sun_phase_rad

    def phase(self, sat_index: int, t_s: float) -> float:
        u = (
            self.phase0
            + 2.0 * math.pi * (t_s / self.period_s)
            + 2.0 * math.pi * sat_index / self.n_sats
        )
        return u % (2.0 * math.pi)

    def is_lit(self, sat_index: int, t_s: float) -> bool:
        if self.f_ecl <= 0.0:
            return True
        d = _ang_dist(self.phase(sat_index, t_s), self.anti_sun)
        return d > math.pi * self.f_ecl

    def next_eclipse_entry_s(self, sat_index: int, t_s: float) -> float:
        """Seconds from ``t_s`` until this satellite next enters the umbra.

        ``inf`` for an eclipse-free plane; ``0.0`` if already eclipsed.
        """
        if self.f_ecl <= 0.0:
            return math.inf
        if not self.is_lit(sat_index, t_s):
            return 0.0
        entry = (self.anti_sun - math.pi * self.f_ecl) % (2.0 * math.pi)
        u = self.phase(sat_index, t_s)
        dtheta = (entry - u) % (2.0 * math.pi)
        return dtheta / (2.0 * math.pi) * self.period_s

    def next_eclipse_exit_s(self, sat_index: int, t_s: float) -> float:
        """Seconds from ``t_s`` until this satellite is lit again (0 if lit)."""
        if self.f_ecl <= 0.0 or self.is_lit(sat_index, t_s):
            return 0.0
        exit_phase = (self.anti_sun + math.pi * self.f_ecl) % (2.0 * math.pi)
        u = self.phase(sat_index, t_s)
        dtheta = (exit_phase - u) % (2.0 * math.pi)
        return dtheta / (2.0 * math.pi) * self.period_s

    def eclipse_duration_s(self) -> float:
        return self.f_ecl * self.period_s

    def lit_slot_count(self) -> int:
        """Number of ring positions simultaneously lit -- LAB-47's arc width."""
        return int(math.floor(self.n_sats * (1.0 - self.f_ecl)))


def _ang_dist(a: float, b: float) -> float:
    d = abs((a - b) % (2.0 * math.pi))
    return min(d, 2.0 * math.pi - d)
