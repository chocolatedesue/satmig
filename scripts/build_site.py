#!/usr/bin/env python3
"""Assemble docs/ into the published site.

    python3 scripts/build_site.py

Copies the generated report and figures into ``docs/`` and renders the two
Markdown docs to HTML with the same minimal renderer the report uses, so the
published site has no build dependency beyond the standard library.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from satmig.report import _html  # noqa: E402

REPORT = ROOT / "results" / "report"
DOCS = ROOT / "docs"

NAV = """<p style="margin:0 0 1.5rem 0;padding:.6rem .8rem;background:#f1f5f9;
border-radius:6px;font-size:14px">
<a href="./">Results report</a> &middot;
<a href="proposals.html">Four proposals</a> &middot;
<a href="model.html">Formal model</a> &middot;
<a href="https://github.com/chocolatedesue/satmig">Source</a>
</p>"""


def _inject_nav(html: str) -> str:
    return html.replace("<body>", "<body>\n" + NAV, 1)


def main() -> int:
    if not REPORT.exists():
        print(
            "results/report is missing -- run:\n"
            "  python3 -m satmig all --out results\n"
            "  python3 -m satmig report --results results --out results/report",
            file=sys.stderr,
        )
        return 1
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")

    for png in sorted(REPORT.glob("*.png")):
        shutil.copy2(png, DOCS / png.name)
    shutil.copy2(REPORT / "report.md", DOCS / "report.md")
    (DOCS / "index.html").write_text(
        _inject_nav((REPORT / "index.html").read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    for src, dst, title in (
        ("PROPOSALS.md", "proposals.html", "satmig -- four proposals"),
        ("MODEL.md", "model.html", "satmig -- formal model"),
    ):
        md = (DOCS / src).read_text(encoding="utf-8")
        page = _inject_nav(_html(md)).replace("<title>satmig report</title>", f"<title>{title}</title>")
        (DOCS / dst).write_text(page, encoding="utf-8")

    written = sorted(p.name for p in DOCS.iterdir())
    print("docs/ now contains:", ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
