"""Kuhn-Munkres (Hungarian) assignment, O(n^3), no third-party dependency.

SpotServe's Device Mapper solves a weighted bipartite matching to maximise
reuse of weights already resident on a GPU.  We reuse it verbatim to pick
which satellites host which stage, so that a re-placement moves the fewest
bytes.
"""

from __future__ import annotations

import math

INF = float("inf")


def solve_min_cost(cost: list[list[float]]) -> list[int]:
    """Minimum-cost perfect assignment of rows to columns.

    Returns ``assign`` with ``assign[row] = col``.  Requires ``rows <= cols``.
    Infeasible pairs are expressed as very large finite costs by the caller;
    ``inf`` is tolerated but every row must have at least one finite option.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m < n:
        raise ValueError("need at least as many columns as rows")

    # JV-style shortest augmenting path with potentials.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[col] = row matched to col (1-indexed, 0 = free)
    way = [0] * (m + 1)

    big = 1e18
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [big] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                c = cost[i0 - 1][j - 1]
                c = big if not math.isfinite(c) else c
                cur = c - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:
                raise ValueError("assignment infeasible")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assign[p[j] - 1] = j - 1
    return assign


def solve_max_reuse(reuse: list[list[float]]) -> list[int]:
    """Maximise total reuse weight -- SpotServe's Device Mapper objective."""
    if not reuse:
        return []
    hi = max(max(r) for r in reuse)
    cost = [[hi - x for x in row] for row in reuse]
    return solve_min_cost(cost)
