"""Config loading, overrides, fingerprints."""

import json
import tempfile
import unittest
from pathlib import Path

from satmig.config import RunConfig, from_dict, load_config


class TestLoading(unittest.TestCase):
    def test_defaults_are_complete(self):
        cfg = RunConfig()
        d = cfg.to_dict()
        for key in ("constellation", "workload", "sim", "model", "satellite", "migration"):
            self.assertIn(key, d)
        self.assertEqual(cfg.model_spec().n_layers, 80)
        self.assertAlmostEqual(cfg.satellite_spec().memory_gb, 24.0)
        self.assertAlmostEqual(cfg.migration_params().isl_gbps, 100.0)

    def test_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text(
                "name: t\nworkload:\n  stages: 7\nsim:\n  slot_s: 5.0\n", encoding="utf-8"
            )
            cfg = load_config(p)
        self.assertEqual(cfg.name, "t")
        self.assertEqual(cfg.workload.stages, 7)
        self.assertAlmostEqual(cfg.sim.slot_s, 5.0)

    def test_unknown_top_level_key_rejected(self):
        with self.assertRaises(KeyError):
            from_dict({"nonsense": 1})

    def test_unknown_section_key_rejected(self):
        with self.assertRaises(KeyError):
            from_dict({"workload": {"nope": 1}})

    def test_all_shipped_configs_load(self):
        root = Path(__file__).resolve().parent.parent / "configs"
        found = sorted(root.glob("*.yaml"))
        self.assertGreaterEqual(len(found), 5)
        for p in found:
            cfg = load_config(p)
            self.assertTrue(cfg.sim.policies, f"{p.name} has no policies")
            for key in cfg.sweep:
                cfg.with_override(key, cfg.sweep[key][0])


class TestOverrides(unittest.TestCase):
    def test_nested_override(self):
        cfg = RunConfig().with_override("workload.stages", 12)
        self.assertEqual(cfg.workload.stages, 12)
        self.assertIsNone(RunConfig().workload.stages)

    def test_dict_section_override(self):
        cfg = RunConfig().with_override("migration.isl_gbps", 10.0)
        self.assertAlmostEqual(cfg.migration_params().isl_gbps, 10.0)

    def test_top_level_override(self):
        self.assertEqual(RunConfig().with_override("name", "x").name, "x")

    def test_unknown_override_rejected(self):
        with self.assertRaises(KeyError):
            RunConfig().with_override("workload.bogus", 1)
        with self.assertRaises(KeyError):
            RunConfig().with_override("bogus.x", 1)
        with self.assertRaises(KeyError):
            RunConfig().with_override("horizon", 1)


class TestFingerprint(unittest.TestCase):
    def test_stable_across_instances(self):
        self.assertEqual(RunConfig().fingerprint(), RunConfig().fingerprint())

    def test_changes_with_any_field(self):
        a = RunConfig().fingerprint()
        self.assertNotEqual(a, RunConfig().with_override("sim.seed", 1).fingerprint())
        self.assertNotEqual(a, RunConfig().with_override("workload.stages", 3).fingerprint())
        self.assertNotEqual(
            a, RunConfig().with_override("migration.ack_ms", 21.0).fingerprint()
        )

    def test_to_dict_is_json_serialisable(self):
        json.dumps(RunConfig().to_dict())


if __name__ == "__main__":
    unittest.main()
