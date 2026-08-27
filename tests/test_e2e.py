"""End-to-end: artefacts, schema, determinism, and cross-checks against LAB-47.

The cross-checks are the reason this file exists.  LAB-47 derived p_full, the
KV-only handover bubble and the per-satellite eclipse energy by hand; this
simulator reaches them from a different direction (closed-form illumination
plus a priced handover).  If those three numbers drift, one of the two is
wrong.
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from satmig.cli import main, run_one, run_sweep
from satmig.config import load_config
from satmig.results import SLOT_FIELDS

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


class TestArtefacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name) / "base"
        cls.summary = run_one(load_config(CONFIGS / "base.yaml"), cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_four_files_exist(self):
        for name in ("manifest.json", "slots.csv", "policy_metrics.csv", "summary.json"):
            self.assertTrue((self.out / name).exists(), name)

    def test_slot_csv_schema_and_size(self):
        with (self.out / "slots.csv").open(encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            self.assertEqual(r.fieldnames, SLOT_FIELDS)
            rows = list(r)
        cfg = load_config(CONFIGS / "base.yaml")
        slots = math.ceil(cfg.sim.horizon_s / cfg.sim.slot_s)
        expect = slots * len(cfg.sim.policies) * cfg.workload.n_pipelines
        self.assertEqual(len(rows), expect)

    def test_manifest_records_provenance_and_boundaries(self):
        m = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("config_fingerprint", m)
        self.assertIn("git_rev", m)
        self.assertIn("config", m)
        self.assertGreaterEqual(len(m["model_boundaries"]), 4)
        self.assertAlmostEqual(m["derived"]["orbital_period_s"], 5730.1, delta=0.2)
        self.assertAlmostEqual(m["derived"]["along_track_hop_ms"], 6.571, delta=0.01)
        self.assertAlmostEqual(m["derived"]["cross_plane_hop_ms"], 2.014, delta=0.01)

    def test_policy_metrics_covers_every_policy(self):
        cfg = load_config(CONFIGS / "base.yaml")
        with (self.out / "policy_metrics.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["policy"] for r in rows], cfg.sim.policies)

    def test_summary_has_comparisons_against_a_baseline(self):
        s = self.summary
        self.assertEqual(s["baseline"], "conveyor")
        self.assertIn("eph_compact", s["comparisons"])
        c = s["comparisons"]["eph_compact"]
        for k in ("throughput_x", "tpot_reduction_pct", "migration_events_reduction_pct"):
            self.assertIn(k, c)


class TestDeterminism(unittest.TestCase):
    def test_two_runs_agree_bit_for_bit(self):
        cfg = load_config(CONFIGS / "base.yaml")
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            run_one(cfg, a)
            run_one(cfg, b)
            for name in ("slots.csv", "policy_metrics.csv", "summary.json"):
                self.assertEqual(
                    (a / name).read_bytes(),
                    (b / name).read_bytes(),
                    f"{name} is not reproducible",
                )

    def test_manifest_differs_only_in_the_timestamp(self):
        cfg = load_config(CONFIGS / "base.yaml")
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            run_one(cfg, a)
            run_one(cfg, b)
            ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
            mb = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
            ma.pop("generated_at_utc")
            mb.pop("generated_at_utc")
            self.assertEqual(ma, mb)


class TestCrossChecksAgainstLab47(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        out = Path(cls._tmp.name) / "x"
        cfg = load_config(CONFIGS / "base.yaml")
        cfg.constellation.epoch_days_since_j2000 = 9210.0
        cls.summary = run_one(cfg, out)
        cls.by = {p["policy"]: p for p in cls.summary["policies"]}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_static_p_full_matches_the_closed_form(self):
        """p_full(P) = max(0, (1-f) - (P-1)/N); f is the home plane's."""
        f = self.summary["derived"]["plane_eclipse_fraction"][0]
        expect = max(0.0, (1 - f) - 9 / 22)
        self.assertAlmostEqual(self.by["static"]["p_all_stages_lit"], expect, delta=0.05)

    def test_conveyor_bubble_is_about_one_and_a_half_percent(self):
        """LAB-47 predicted 1.32% for weights-pre-staged, KV stop-and-copy."""
        self.assertAlmostEqual(self.by["conveyor"]["downtime_fraction"], 0.014, delta=0.006)

    def test_static_battery_draw_is_about_160wh_per_satellite_per_orbit(self):
        """LAB-47's 162 Wh figure, reached from the other side."""
        per_stage = self.by["static"]["battery_wh_per_orbit"] / self.by["static"]["n_stages"]
        self.assertAlmostEqual(per_stage, 162.0, delta=25.0)

    def test_conveyor_keeps_every_stage_lit(self):
        self.assertAlmostEqual(self.by["conveyor"]["p_all_stages_lit"], 1.0, places=6)

    def test_incremental_patching_cuts_the_bubble_by_an_order_of_magnitude(self):
        self.assertGreater(
            self.by["conveyor"]["downtime_fraction"]
            / self.by["eph"]["downtime_fraction"],
            10.0,
        )

    def test_compact_placement_roughly_halves_tpot(self):
        """A width-5 snake replaces 18 along-track hops with 8 cross-plane ones."""
        red = self.summary["comparisons"]["eph_compact"]["tpot_reduction_pct"]
        self.assertGreater(red, 40.0)
        self.assertLess(red, 65.0)


class TestCli(unittest.TestCase):
    def test_run_subcommand(self):
        with tempfile.TemporaryDirectory() as d:
            rc = main(["run", "--config", str(CONFIGS / "base.yaml"), "--out", d + "/r"])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "r" / "summary.json").exists())

    def test_sweep_writes_an_index(self):
        cfg = load_config(CONFIGS / "exp4_slo.yaml")
        cfg.sweep = {"workload.slo_tpot_ms": [60.0, 120.0]}
        cfg.sim.horizon_s = 1200.0
        with tempfile.TemporaryDirectory() as d:
            run_sweep(cfg, Path(d) / "s")
            idx = json.loads(
                (Path(d) / "s" / "sweep_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(idx["runs"]), 2)
            for r in idx["runs"]:
                self.assertTrue((Path(d) / "s" / r["dir"] / "summary.json").exists())

    def test_unknown_subcommand_fails(self):
        with self.assertRaises(SystemExit):
            main(["nope"])


if __name__ == "__main__":
    unittest.main()
