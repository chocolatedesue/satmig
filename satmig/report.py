"""Figures and a self-contained HTML/Markdown report from a results tree.

Reads whatever ``python -m satmig all`` wrote and produces:

* ``fig_*.png`` -- one figure per experiment
* ``report.md`` -- numbers in tables, with the credibility tier of each
* ``index.html`` -- the same, standalone, for publishing
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "static": "#94a3b8",
    "reactive_jit": "#f97316",
    "llumnix_reactive": "#eab308",
    "conveyor": "#3b82f6",
    "eph": "#14b8a6",
    "eph_compact": "#8b5cf6",
    "eph_freeness": "#ec4899",
    "two_timescale": "#22c55e",
}
LABEL = {
    "static": "static (no migration)",
    "reactive_jit": "reactive JIT (SpotServe port)",
    "llumnix_reactive": "reactive freeness (Llumnix port)",
    "conveyor": "conveyor (LAB-47 baseline)",
    "eph": "EPH",
    "eph_compact": "EPH + compact block",
    "eph_freeness": "EPH + energy freeness",
    "two_timescale": "two-timescale",
}


def _load_summary(d: Path) -> dict | None:
    p = d / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_sweep(d: Path) -> list[tuple[float | str, dict]]:
    idx = d / "sweep_index.json"
    if not idx.exists():
        return []
    data = json.loads(idx.read_text(encoding="utf-8"))
    out = []
    for run in data["runs"]:
        s = _load_summary(d / run["dir"])
        if s is None:
            continue
        raw = run["label"].split("=", 1)[1]
        try:
            key: float | str = float(raw)
        except ValueError:
            key = raw
        out.append((key, s))
    out.sort(key=lambda kv: (isinstance(kv[0], str), kv[0]))
    return out


def _metric(summary: dict, policy: str, key: str) -> float:
    for p in summary["policies"]:
        if p["policy"] == policy:
            v = p.get(key, float("nan"))
            return float(v) if v is not None else float("nan")
    return float("nan")


def _policies(summary: dict) -> list[str]:
    return [p["policy"] for p in summary["policies"]]


def _sweep_figure(
    runs, key: str, ylabel: str, title: str, xlabel: str, path: Path, logy=False
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [k for k, _ in runs]
    for pol in _policies(runs[0][1]):
        ys = [_metric(s, pol, key) for _, s in runs]
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=4,
            color=PALETTE.get(pol, "#64748b"),
            label=LABEL.get(pol, pol),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if logy:
        ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_feasibility(runs, path: Path) -> Path:
    """E1/E4: the SLO ceiling against the memory floor.

    The ceiling is a property of the *placement*, not of the migration policy,
    so the curves are grouped by placement family; policies inside a family
    coincide exactly.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [k for k, _ in runs]
    floor = runs[0][1]["derived"]["min_stages_for_memory"]
    families = [
        ("conveyor", "1xP along-track arc (LAB-47, ground PP)", "#3b82f6"),
        ("eph_compact", "compact snake block (proposed)", "#8b5cf6"),
    ]
    crossings = {}
    for pol, label, colour in families:
        if pol not in _policies(runs[0][1]):
            continue
        ys = [_metric(s, pol, "max_stages_under_slo") for _, s in runs]
        ax.plot(xs, ys, marker="o", ms=5, color=colour, label=label)
        for x, y in zip(xs, ys):
            if y >= floor:
                crossings[label] = x
                break
    ax.axhline(floor, ls="--", color="#dc2626", label=f"memory floor P >= {floor}")
    ax.fill_between(xs, 0, floor, color="#dc2626", alpha=0.07)
    for label, x in crossings.items():
        ax.axvline(x, ls=":", color="#64748b", lw=1)
        ax.annotate(
            f"feasible from {x:.0f} ms",
            xy=(x, floor),
            xytext=(x + 4, floor + 4),
            fontsize=7,
            color="#334155",
        )
    ax.set_xlabel("TPOT SLO (ms)")
    ax.set_ylabel("largest split P admitted by the SLO")
    ax.set_title("Feasible split window: latency ceiling vs memory floor", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_split_sweep(runs, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = [k for k, _ in runs]
    for pol in _policies(runs[0][1]):
        axes[0].plot(
            xs,
            [_metric(s, pol, "mean_throughput_tok_s") for _, s in runs],
            marker="o",
            ms=4,
            color=PALETTE.get(pol, "#64748b"),
            label=LABEL.get(pol, pol),
        )
        axes[1].plot(
            xs,
            [_metric(s, pol, "mean_tpot_ms") for _, s in runs],
            marker="o",
            ms=4,
            color=PALETTE.get(pol, "#64748b"),
        )
    slo = runs[0][1]["policies"][0].get("slo_tpot_ms")
    axes[1].axhline(100.0 if slo is None else slo, ls="--", color="#dc2626", label="SLO")
    for ax, yl, t in (
        (axes[0], "mean throughput (tok/s)", "Throughput vs split"),
        (axes[1], "mean TPOT (ms)", "Latency vs split"),
    ):
        ax.set_xlabel("stages P")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_policy_bars(summary: dict, path: Path) -> Path:
    pols = _policies(summary)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    panels = [
        ("mean_throughput_tok_s", "mean throughput (tok/s)", False),
        ("mean_tpot_ms", "mean TPOT (ms)", False),
        ("migration_events_per_hour", "handover events / hour", False),
        ("downtime_fraction", "handover downtime fraction", True),
    ]
    for ax, (key, ylabel, logy) in zip(axes.flat, panels):
        vals = [_metric(summary, p, key) for p in pols]
        ax.bar(
            range(len(pols)),
            vals,
            color=[PALETTE.get(p, "#64748b") for p in pols],
        )
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([LABEL.get(p, p) for p in pols], rotation=32, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Per-policy summary -- {summary['run_name']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_season(runs, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [k for k, _ in runs]
    free = [s["derived"]["planes_eclipse_free"] for _, s in runs]
    ax.bar(xs, free, width=15, color="#fbbf24", alpha=0.6, label="eclipse-free planes")
    ax.set_xlabel("epoch (days since J2000)")
    ax.set_ylabel("eclipse-free planes (of 72)")
    ax2 = ax.twinx()
    for pol in ("conveyor", "eph_freeness", "two_timescale"):
        if pol not in _policies(runs[0][1]):
            continue
        ax2.plot(
            xs,
            [_metric(s, pol, "migration_events_per_hour") for _, s in runs],
            marker="o",
            ms=4,
            color=PALETTE[pol],
            label=LABEL[pol],
        )
    ax2.set_ylabel("handover events / hour")
    ax.set_title("Seasonality: the zero-migration regime comes and goes", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _fmt(x: float, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def build_report(results: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    sections: list[str] = []

    base = _load_summary(results / "base")
    if base:
        figs.append(fig_policy_bars(base, out / "fig_base_policies.png"))
        rows = []
        for p in base["policies"]:
            rows.append(
                [
                    LABEL.get(p["policy"], p["policy"]),
                    str(p["n_stages"]),
                    str(p["max_stages_under_slo"]),
                    _fmt(p["mean_throughput_tok_s"], 1),
                    _fmt(p["mean_tpot_ms"], 1),
                    _fmt(p["slo_attainment"], 3),
                    _fmt(p["p_all_stages_lit"], 3),
                    _fmt(p["migration_events_per_hour"], 1),
                    _fmt(p["downtime_fraction"] * 100, 4) + "%",
                    _fmt(p["battery_wh_per_orbit"], 1),
                ]
            )
        sections.append(
            "## Base run (1 pipeline, P=10, 3 orbits)\n\n"
            + _table(
                rows,
                [
                    "policy",
                    "P",
                    "P max @SLO",
                    "tok/s",
                    "TPOT ms",
                    "SLO att.",
                    "all-lit",
                    "events/h",
                    "downtime",
                    "battery Wh/orbit",
                ],
            )
            + "\n\n![base](fig_base_policies.png)\n"
        )

    e1 = _load_sweep(results / "exp1_split")
    if e1:
        figs.append(fig_split_sweep(e1, out / "fig_e1_split.png"))
        sections.append(
            "## E1 -- split sweep\n\n"
            "Throughput rises with P while latency rises faster; the useful\n"
            "reading is where each curve crosses the SLO line.\n\n"
            "![e1](fig_e1_split.png)\n"
        )

    e2 = _load_sweep(results / "exp2_multipipeline")
    if e2:
        figs.append(
            _sweep_figure(
                e2,
                "aggregate_throughput_tok_s",
                "aggregate throughput (tok/s)",
                "E2 -- K pipelines on a 2-plane hot region",
                "pipelines K",
                out / "fig_e2_aggregate.png",
            )
        )
        figs.append(
            _sweep_figure(
                e2,
                "colocation_rate",
                "fraction of stages sharing a satellite",
                "E2 -- where multi-tenancy actually breaks",
                "pipelines K",
                out / "fig_e2_coloc.png",
            )
        )
        rows = [
            [
                _fmt(k, 0),
                _fmt(_metric(s, "conveyor", "aggregate_throughput_tok_s"), 1),
                _fmt(_metric(s, "eph_freeness", "aggregate_throughput_tok_s"), 1),
                _fmt(_metric(s, "conveyor", "colocation_rate"), 3),
                _fmt(_metric(s, "eph_freeness", "colocation_rate"), 3),
                str(int(_metric(s, "eph_freeness", "peak_link_flows"))),
            ]
            for k, s in e2
        ]
        sections.append(
            "## E2 -- multi-pipeline contention\n\n"
            + _table(
                rows,
                [
                    "K",
                    "conveyor agg tok/s",
                    "eph_freeness agg tok/s",
                    "conveyor coloc.",
                    "eph_freeness coloc.",
                    "peak ISL flows",
                ],
            )
            + "\n\n![e2a](fig_e2_aggregate.png)\n\n![e2b](fig_e2_coloc.png)\n"
        )

    e3 = _load_sweep(results / "exp3_battery")
    if e3:
        figs.append(
            _sweep_figure(
                e3,
                "mean_throughput_tok_s",
                "mean throughput (tok/s)",
                "E3 -- sensitivity to the eclipse derate c_e",
                "c_eclipse",
                out / "fig_e3_battery.png",
            )
        )
        sections.append(
            "## E3 -- how much depends on the battery derate\n\n"
            "At c_e = 1 the battery carries full compute and every migration\n"
            "policy collapses onto `static`; at c_e = 0 the gap is maximal.\n"
            "Every ratio elsewhere in this report should be read against this\n"
            "curve.\n\n![e3](fig_e3_battery.png)\n"
        )

    e4 = _load_sweep(results / "exp4_slo")
    if e4:
        figs.append(fig_feasibility(e4, out / "fig_e4_feasibility.png"))
        rows = []
        floor = e4[0][1]["derived"]["min_stages_for_memory"]
        for k, s in e4:
            rows.append(
                [
                    _fmt(k, 0),
                    str(int(_metric(s, "conveyor", "max_stages_under_slo"))),
                    str(int(_metric(s, "eph_compact", "max_stages_under_slo"))),
                    str(floor),
                    "yes"
                    if _metric(s, "eph_compact", "max_stages_under_slo") >= floor
                    else "no",
                    "yes"
                    if _metric(s, "conveyor", "max_stages_under_slo") >= floor
                    else "no",
                ]
            )
        sections.append(
            "## E4 -- the feasible split window\n\n"
            + _table(
                rows,
                [
                    "SLO ms",
                    "P max, 1xP arc",
                    "P max, snake",
                    "memory floor",
                    "snake feasible?",
                    "arc feasible?",
                ],
            )
            + "\n\n![e4](fig_e4_feasibility.png)\n"
        )

    e5 = _load_sweep(results / "exp5_season")
    if e5:
        figs.append(fig_season(e5, out / "fig_e5_season.png"))
        sections.append(
            "## E5 -- seasonality\n\n"
            "The number of eclipse-free planes in a 53 deg shell is a function\n"
            "of solar declination, so the two-timescale policy's zero-migration\n"
            "regime exists near solstice and vanishes near equinox.\n\n"
            "![e5](fig_e5_season.png)\n"
        )

    e6 = _load_sweep(results / "exp6_bandwidth")
    if e6:
        figs.append(
            _sweep_figure(
                e6,
                "mean_throughput_tok_s",
                "mean throughput (tok/s)",
                "E6 -- ISL rate: where bandwidth starts to bind",
                "ISL rate (Gbps)",
                out / "fig_e6_bandwidth.png",
            )
        )
        rows = [
            [
                _fmt(k, 1),
                _fmt(_metric(s, "conveyor", "mean_throughput_tok_s"), 1),
                _fmt(_metric(s, "eph_freeness", "mean_throughput_tok_s"), 1),
                str(int(_metric(s, "eph_freeness", "deadline_misses"))),
                str(int(_metric(s, "conveyor", "deadline_misses"))),
            ]
            for k, s in e6
        ]
        sections.append(
            "## E6 -- is the ISL the bottleneck?\n\n"
            "A single-hop lockstep shift gives every stage the full link, so\n"
            "`conveyor` never misses a deadline.  A batched J-hop jump routes P\n"
            "transfers over the same J links, so it needs roughly `P/J` times\n"
            "more headroom -- which is why the jump distance has to be chosen\n"
            "against the link rate, not just the arc slack.\n\n"
            + _table(
                rows,
                [
                    "ISL Gbps",
                    "conveyor tok/s",
                    "eph_freeness tok/s",
                    "eph_freeness misses",
                    "conveyor misses",
                ],
            )
            + "\n\n![e6](fig_e6_bandwidth.png)\n"
        )

    md = _header(base) + "\n\n" + "\n\n".join(sections) + "\n\n" + _footer()
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "index.html").write_text(_html(md), encoding="utf-8")
    return [out / "report.md", out / "index.html", *figs]


def _header(base: dict | None) -> str:
    d = base["derived"] if base else {}
    lines = [
        "# satmig -- eclipse-aware live migration of LLM pipeline stages on a LEO shell",
        "",
        "Every number below is regenerated by `python -m satmig all && python -m satmig report`.",
        "",
        "## Constants this study rests on",
        "",
        _table(
            [
                ["orbital period", _fmt(d.get("orbital_period_s", float("nan")), 1) + " s"],
                ["slot advance T/N", _fmt(d.get("hop_period_s", float("nan")), 1) + " s"],
                [
                    "along-track ISL hop",
                    _fmt(d.get("along_track_hop_ms", float("nan")), 3) + " ms",
                ],
                [
                    "cross-plane ISL hop",
                    _fmt(d.get("cross_plane_hop_ms", float("nan")), 3) + " ms",
                ],
                ["memory floor on P", str(d.get("min_stages_for_memory", "-"))],
                ["eclipse-free planes", str(d.get("planes_eclipse_free", "-"))],
            ],
            ["quantity", "value"],
        ),
    ]
    return "\n".join(lines)


def _footer() -> str:
    return (
        "## Credibility tiers\n\n"
        "**Cross-checked against independent derivations in this lab** -- orbital\n"
        "period, critical beta angle (67.0 deg), eclipse fraction at beta=0\n"
        "(0.372), per-plane beta spread, the closed-form `p_full`, the KV-only\n"
        "handover bubble (~1.4%) and the ~162 Wh per satellite per orbit of\n"
        "eclipse compute energy.  These agree with LAB-47 / LAB-58 / LAB-67,\n"
        "which reached them with satellite.js + SGP4 rather than closed forms.\n\n"
        "**Ported from published work, parameters taken as given** -- CONNEX's\n"
        "11.1 ms pair-local cutover and its ~50x slower global rebuild;\n"
        "PipeLive's MaxBlocks feasibility test, layer-stacking granularity and\n"
        "incremental KV patching; SCOPE's Eq. 1-3; SpotServe's KM device mapping\n"
        "and 30 s grace period; Llumnix's freeness formula.\n\n"
        "**Our assumptions, not measured** -- normalised compute capacity as a\n"
        "power proxy, the eclipse derate `c_e` (see E3), per-layer decode time\n"
        "calibrated to a 42 ms whole-model decode step, 24 GB of usable memory\n"
        "per satellite, and the KV dirty rate that sets pre-copy convergence.\n\n"
        "**Not modelled** -- attitude and panel incidence (so no-power time here\n"
        "is a lower bound), battery state of charge, penumbra, atmospheric\n"
        "drag, J2 short-period terms, prefill/decode mixing, request arrivals,\n"
        "and sub-slot queueing.\n"
    )


def _html(md: str) -> str:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except Exception:
        body = _mini_markdown(md)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>satmig report</title>
<style>
 body{{font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
   max-width:1080px;margin:0 auto;padding:2rem 1.25rem;color:#0f172a}}
 h1{{font-size:1.7rem;border-bottom:2px solid #e2e8f0;padding-bottom:.4rem}}
 h2{{font-size:1.2rem;margin-top:2.2rem;color:#1e3a8a}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:13px}}
 th,td{{border:1px solid #dbe3ec;padding:.4rem .55rem;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f1f5f9}}
 tr:nth-child(even) td{{background:#fafcff}}
 img{{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;margin:.6rem 0}}
 code{{background:#f1f5f9;padding:.1rem .3rem;border-radius:3px;font-size:.9em}}
 strong{{color:#0f172a}}
</style></head><body>
{body}
</body></html>
"""


def _mini_markdown(md: str) -> str:
    """Enough Markdown for this report, with no third-party dependency."""
    html: list[str] = []
    in_table = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-:") and c for c in cells):
                continue
            if not in_table:
                html.append("<table>")
                in_table = True
                html.append(
                    "<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells) + "</tr>"
                )
                continue
            html.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            html.append("</table>")
            in_table = False
        if line.startswith("### "):
            html.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            html.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            html.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("!["):
            alt, _, rest = line[2:].partition("](")
            html.append(f'<img alt="{alt}" src="{rest.rstrip(")")}">')
        elif not line:
            html.append("")
        else:
            html.append(f"<p>{_inline(line)}</p>")
    if in_table:
        html.append("</table>")
    return "\n".join(html)


def _inline(s: str) -> str:
    import re

    s = (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s
