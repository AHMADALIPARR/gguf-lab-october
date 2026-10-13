# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Edge case tests for sandbox router and runner logic.

Tests pure-Python logic without requiring models or llama.cpp.
Run: python3 -m unittest tests.test_sandbox -v
"""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sandbox" / "moe"))
sys.path.insert(0, str(ROOT / "sandbox"))

from router import cosine
import router as router_mod


class CosineEdgeCases(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(cosine([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)

    def test_opposite(self):
        self.assertAlmostEqual(cosine([1, 2], [-1, -2]), -1.0)

    def test_zero_vector_no_crash(self):
        # Must not divide by zero; defined as 0.0
        self.assertEqual(cosine([0, 0, 0], [1, 2, 3]), 0.0)
        self.assertEqual(cosine([1, 2], [0, 0]), 0.0)

    def test_scale_invariance(self):
        self.assertAlmostEqual(cosine([2, 4, 6], [1, 2, 3]), 1.0)

    def test_768dim_bounded(self):
        import random
        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(768)]
        b = [random.gauss(0, 1) for _ in range(768)]
        c = cosine(a, b)
        self.assertGreaterEqual(c, -1.0)
        self.assertLessEqual(c, 1.0)


class RouterSelectionTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tdp = pathlib.Path(self.td.name)
        self.exp_path = self.tdp / "experts.json"
        self.cen_path = self.tdp / "centroids.json"
        self.exp_path.write_text(json.dumps({
            "experts": [
                {"id": "coder", "model": "m1.gguf", "anchor_prompts": []},
                {"id": "writer", "model": "m2.gguf", "anchor_prompts": []},
            ]}))
        self.cen_path.write_text(json.dumps({
            "centroids": {"coder": [1.0, 0.0, 0.0],
                          "writer": [0.0, 1.0, 0.0]}}))
        self._orig = router_mod.embed_via_server

        def mock_embed(url, text):
            if "code" in text.lower() or "python" in text.lower():
                return [0.9, 0.1, 0.0]
            return [0.1, 0.9, 0.0]
        router_mod.embed_via_server = mock_embed

    def tearDown(self):
        router_mod.embed_via_server = self._orig
        self.td.cleanup()

    def test_routes_coding_to_coder(self):
        sel, _ = router_mod.route("http://x", "Write Python code",
                                  self.exp_path, self.cen_path, top_k=1)
        self.assertEqual(sel[0]["id"], "coder")

    def test_routes_writing_to_writer(self):
        sel, _ = router_mod.route("http://x", "Write an essay",
                                  self.exp_path, self.cen_path, top_k=1)
        self.assertEqual(sel[0]["id"], "writer")

    def test_top_k_ordered(self):
        sel, _ = router_mod.route("http://x", "Write Python code",
                                  self.exp_path, self.cen_path, top_k=2)
        self.assertEqual(len(sel), 2)
        self.assertEqual(sel[0]["id"], "coder")
        self.assertGreater(sel[0]["similarity"], sel[1]["similarity"])

    def test_missing_centroids_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            router_mod.route("http://x", "test",
                             self.exp_path, self.tdp / "nope.json")


class SandboxConfigTests(unittest.TestCase):
    def test_config_valid(self):
        cfg = json.loads((ROOT / "sandbox" / "config.json").read_text())
        self.assertEqual(cfg["sandbox_version"], "1.0")
        self.assertIn("resource_limits", cfg)
        self.assertIn("models", cfg)

    def test_prompts_valid(self):
        prompts = json.loads(
            (ROOT / "sandbox" / "inputs" / "prompts.json").read_text())
        self.assertGreaterEqual(len(prompts), 4)
        for p in prompts:
            self.assertIn("id", p)
            self.assertIn("prompt", p)
            self.assertIn("must_contain", p)

    def test_experts_valid(self):
        experts = json.loads(
            (ROOT / "sandbox" / "moe" / "experts.json").read_text())
        self.assertEqual(experts["embedding_dim"], 768)
        for e in experts["experts"]:
            self.assertGreaterEqual(len(e["anchor_prompts"]), 1)


if __name__ == "__main__":
    unittest.main()
