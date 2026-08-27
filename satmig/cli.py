"""Single command entry point.

    python -m satmig run    --config configs/base.yaml --out results/base
    python -m satmig sweep  --config configs/exp1_split.yaml --out results/exp1
    python -m satmig all    --out results          # every configs/*.yaml
    python -m satmig report --results results      # figures + report.md/html
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

from .config import RunConfig, load_config
from .results import write_results
from .simulator import Simulator


def run_one(cfg: RunConfig, outdir: Path) -> dict:
    sim = Simulator(cfg)
    results = sim.run()
    return write_results(cfg, results, sim, outdir)


def run_sweep(cfg: RunConfig, outdir: Path) -> dict:
    keys = list(cfg.sweep.keys())
    if not keys:
        return run_one(cfg, outdir)
    combos = list(itertools.product(*(cfg.sweep[k] for k in keys)))
    index = []
    for combo in combos:
        child = replace(cfg, sweep={})
        label_parts = []
        for k, v in zip(keys, combo):
            child = child.with_override(k, v)
            label_parts.append(f"{k.split('.')[-1]}={v}")
        label = ",".join(label_parts)
        sub = outdir / _slug(label)
        summary = run_one(replace(child, name=f"{cfg.name}[{label}]"), sub)
        index.append({"label": label, "dir": sub.name, "summary": summary})
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sweep_index.json").write_text(
        json.dumps(
            {"run_name": cfg.name, "sweep": cfg.sweep, "runs": index},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"run_name": cfg.name, "sweep_runs": len(index)}


def _slug(s: str) -> str:
    return (
        s.replace("=", "").replace(",", "_").replace(".", "").replace("/", "-")
        or "run"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="satmig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("run", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)

    pa = sub.add_parser("all")
    pa.add_argument("--configs", default="configs")
    pa.add_argument("--out", default="results")

    pr = sub.add_parser("report")
    pr.add_argument("--results", default="results")
    pr.add_argument("--out", default="results/report")

    args = ap.parse_args(argv)

    if args.cmd == "run":
        s = run_one(load_config(args.config), Path(args.out))
        print(json.dumps({"wrote": args.out, "baseline": s.get("baseline")}, indent=2))
        return 0
    if args.cmd == "sweep":
        s = run_sweep(load_config(args.config), Path(args.out))
        print(json.dumps(s, indent=2))
        return 0
    if args.cmd == "all":
        cfgdir = Path(args.configs)
        out = Path(args.out)
        done = []
        for path in sorted(cfgdir.glob("*.yaml")):
            cfg = load_config(path)
            target = out / path.stem
            if cfg.sweep:
                run_sweep(cfg, target)
            else:
                run_one(cfg, target)
            done.append(path.stem)
        print(json.dumps({"ran": done, "out": str(out)}, indent=2))
        return 0
    if args.cmd == "report":
        from .report import build_report

        paths = build_report(Path(args.results), Path(args.out))
        print(json.dumps({"wrote": [str(p) for p in paths]}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
