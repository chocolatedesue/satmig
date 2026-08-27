"""Constellation topology: the (plane, slot) lattice and its ISL delays.

The single fact this module exists to expose is that the two ISL directions of
a Walker shell have very different propagation costs:

    along-track hop  (same plane, adjacent slot)  : 2*r*sin(pi/N)
    cross-plane hop  (adjacent plane, same slot)  : 2*r*sin(pi*cos(lat)/Q)

For the 550 km / 72-plane / 22-sat Starlink-like shell that is 6.57 ms vs
2.01 ms -- a 3.3x difference that decides whether a P-stage pipeline can meet
an interactive TPOT budget at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

from .orbit import (
    C_LIGHT_KM_S,
    DEG,
    R_EARTH_KM,
    PlaneIllumination,
    beta_angle_deg,
    orbital_period_s,
    sun_direction,
    umbra_center_arg_lat_rad,
)

SatId = tuple[int, int]  # (plane, slot)


def chord_km(radius_km: float, angle_rad: float) -> float:
    return 2.0 * radius_km * math.sin(angle_rad / 2.0)


@dataclass(frozen=True)
class HopDelays:
    along_ms: float
    cross_ms: float


class Constellation:
    """A Walker-delta shell on a (plane, slot) torus.

    Each plane gets its own sun beta angle *and* its own umbra centre phase,
    which is what makes planes non-equivalent -- the LAB-67 finding that one
    shell holds planes from 0% to 37% eclipse at the same instant, and the
    reason a cross-plane stage chain does not see a single co-moving lit arc.
    """

    def __init__(
        self,
        altitude_km: float,
        n_planes: int,
        n_sats_per_plane: int,
        plane_beta_deg: list[float],
        plane_umbra_phase_rad: list[float] | None = None,
        plane_phase0_rad: list[float] | None = None,
        cross_plane_latitude_deg: float = 0.0,
    ) -> None:
        if len(plane_beta_deg) != n_planes:
            raise ValueError("plane_beta_deg must have one entry per plane")
        self.altitude_km = altitude_km
        self.n_planes = n_planes
        self.n_sats = n_sats_per_plane
        self.radius_km = R_EARTH_KM + altitude_km
        self.period_s = orbital_period_s(altitude_km)
        self.cross_plane_latitude_deg = cross_plane_latitude_deg
        umbra = plane_umbra_phase_rad or [math.pi] * n_planes
        phase0 = plane_phase0_rad or [0.0] * n_planes
        self.plane_beta_deg = list(plane_beta_deg)
        self.planes: list[PlaneIllumination] = [
            PlaneIllumination(
                altitude_km,
                plane_beta_deg[q],
                n_sats_per_plane,
                phase0_rad=phase0[q],
                anti_sun_phase_rad=umbra[q],
            )
            for q in range(n_planes)
        ]

    @classmethod
    def walker(
        cls,
        altitude_km: float,
        inclination_deg: float,
        n_planes: int,
        n_sats_per_plane: int,
        epoch_days_since_j2000: float = 9000.0,
        raan_spread_deg: float = 360.0,
        raan_offset_deg: float = 0.0,
        phasing_f: int = 1,
        cross_plane_latitude_deg: float = 0.0,
        plane_beta_deg: list[float] | None = None,
    ) -> "Constellation":
        sun = sun_direction(epoch_days_since_j2000)
        betas: list[float] = []
        umbra: list[float] = []
        phase0: list[float] = []
        for q in range(n_planes):
            raan = raan_offset_deg + raan_spread_deg * q / n_planes
            betas.append(beta_angle_deg(inclination_deg, raan, sun))
            umbra.append(umbra_center_arg_lat_rad(inclination_deg, raan, sun))
            phase0.append(
                2.0 * math.pi * phasing_f * q / (n_planes * n_sats_per_plane)
            )
        if plane_beta_deg is not None:
            betas = list(plane_beta_deg)
        return cls(
            altitude_km,
            n_planes,
            n_sats_per_plane,
            betas,
            plane_umbra_phase_rad=umbra,
            plane_phase0_rad=phase0,
            cross_plane_latitude_deg=cross_plane_latitude_deg,
        )

    # -- geometry -----------------------------------------------------------
    @property
    def hop_delays(self) -> HopDelays:
        along = chord_km(self.radius_km, 2.0 * math.pi / self.n_sats)
        cross_angle = 2.0 * math.pi / self.n_planes * math.cos(
            self.cross_plane_latitude_deg * DEG
        )
        cross = chord_km(self.radius_km, cross_angle)
        return HopDelays(
            along_ms=along / C_LIGHT_KM_S * 1e3,
            cross_ms=cross / C_LIGHT_KM_S * 1e3,
        )

    def hop_delay_ms(self, a: SatId, b: SatId) -> float:
        """Delay of one ISL hop.  Only lattice-adjacent pairs are linked."""
        d = self.hop_delays
        dp = _ring_delta(a[0], b[0], self.n_planes)
        ds = _ring_delta(a[1], b[1], self.n_sats)
        if dp == 0 and abs(ds) == 1:
            return d.along_ms
        if ds == 0 and abs(dp) == 1:
            return d.cross_ms
        raise ValueError(f"{a} and {b} are not ISL neighbours")

    def path_delay_ms(self, path: list[SatId]) -> float:
        return sum(self.hop_delay_ms(path[i], path[i + 1]) for i in range(len(path) - 1))

    def manhattan_delay_ms(self, a: SatId, b: SatId) -> float:
        """Shortest lattice-route delay between any two satellites."""
        d = self.hop_delays
        dp = abs(_ring_delta(a[0], b[0], self.n_planes))
        ds = abs(_ring_delta(a[1], b[1], self.n_sats))
        return dp * d.cross_ms + ds * d.along_ms

    def chain_delays_ms(self, path: list[SatId]) -> tuple[list[float], float]:
        """Forward inter-stage hops plus the return hop of the token loop.

        The last stage has to ship the sampled token back to the first stage
        before the next decode step, and on an open arc that return is *not*
        one hop -- it is the whole way back.  A 1xP along-track arc therefore
        pays ``2*(P-1)`` along-track hops per token, which is the correction
        that kills LAB-47's P*=14 once latency is put back.
        """
        fwd = [
            self.hop_delay_ms(path[i], path[i + 1]) for i in range(len(path) - 1)
        ]
        ret = self.manhattan_delay_ms(path[-1], path[0]) if len(path) > 1 else 0.0
        return fwd, ret

    def hop_period_s(self) -> float:
        """Time for the shadow arc to advance one along-track slot."""
        return self.period_s / self.n_sats

    # -- illumination -------------------------------------------------------
    def is_lit(self, sat: SatId, t_s: float) -> bool:
        return self.planes[sat[0]].is_lit(sat[1], t_s)

    def capacity(self, sat: SatId, t_s: float, c_eclipse: float) -> float:
        return 1.0 if self.is_lit(sat, t_s) else c_eclipse

    def time_to_eclipse_s(self, sat: SatId, t_s: float) -> float:
        return self.planes[sat[0]].next_eclipse_entry_s(sat[1], t_s)

    def time_to_sunlight_s(self, sat: SatId, t_s: float) -> float:
        return self.planes[sat[0]].next_eclipse_exit_s(sat[1], t_s)

    def all_sats(self) -> Iterator[SatId]:
        for q in range(self.n_planes):
            for k in range(self.n_sats):
                yield (q, k)

    def lit_count(self, t_s: float) -> int:
        return sum(1 for s in self.all_sats() if self.is_lit(s, t_s))

    # -- link identity, for contention accounting --------------------------
    def link_id(self, a: SatId, b: SatId) -> tuple[SatId, SatId]:
        return (a, b) if a <= b else (b, a)

    def links_of_path(self, path: list[SatId]) -> list[tuple[SatId, SatId]]:
        return [self.link_id(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # -- placement shapes --------------------------------------------------
    def along_track_arc(self, head: SatId, length: int) -> list[SatId]:
        q, k = head
        return [(q, (k + i) % self.n_sats) for i in range(length)]

    def cross_plane_column(self, head: SatId, length: int) -> list[SatId]:
        q, k = head
        return [((q + i) % self.n_planes, k) for i in range(length)]

    def snake_block(self, head: SatId, length: int, width: int) -> list[SatId]:
        """Boustrophedon block: ``width`` cross-plane columns joined along track.

        Keeps the stage chain physically compact so that the number of
        expensive along-track hops is ``ceil(length/width) - 1`` instead of
        ``length - 1``.
        """
        if width < 1:
            raise ValueError("width must be >= 1")
        if width > self.n_planes:
            raise ValueError("width exceeds the number of planes")
        columns = math.ceil(length / width)
        if columns > self.n_sats:
            raise ValueError(
                f"block of {length} stages at width {width} needs {columns} "
                f"slots but the ring only has {self.n_sats}"
            )
        q0, k0 = head
        out: list[SatId] = []
        col = 0
        while len(out) < length:
            slot = (k0 + col) % self.n_sats
            rng: Iterable[int] = range(width) if col % 2 == 0 else range(width - 1, -1, -1)
            for j in rng:
                if len(out) == length:
                    break
                out.append(((q0 + j) % self.n_planes, slot))
            col += 1
        return out


def _ring_delta(a: int, b: int, n: int) -> int:
    d = (b - a) % n
    if d > n // 2:
        d -= n
    return d
