# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Tests for the WORM metadata training pipeline."""
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from worm_train import (
    align_up, stage_ingest, stage_validate, stage_layout, stage_seal,
    stage_emit, train_one, STAGING_ALIGN,
)


def make_manifest(tmpdir, tensors, tensor_count=None):
    """Write a minimal JSONL manifest. Returns path."""
    mp = pathlib.Path(tmpdir) / "test.jsonl"
    with open(mp, "w") as f:
        f.write(json.dumps({
            "type": "manifest", "model_id": "test",
            "tensor_count": tensor_count if tensor_count is not None else len(tensors),
        }) + "\n")
        for t in tensors:
            f.write(json.dumps({"type": "tensor", **t}) + "\n")
    return mp


def good_tensor(name="a.weight", shape=[4, 4], dtype="F32"):
    return {"name": name, "shape": shape, "dtype": dtype,
            "offset": 0, "n_elements": 16, "model_id": "test"}


class IngestTests(unittest.TestCase):
    def test_ingest_valid(self):
        with tempfile.TemporaryDirectory() as td:
            mp = make_manifest(td, [good_tensor(), good_tensor("b.weight")])
            manifest, tensors = stage_ingest(mp)
            self.assertEqual(manifest["model_id"], "test")
            self.assertEqual(len(tensors), 2)

    def test_ingest_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as td:
            mp = pathlib.Path(td) / "bad.jsonl"
            mp.write_text('{"type": "manifest", "model_id": "x"}\nnot json\n')
            with self.assertRaises(ValueError):
                stage_ingest(mp)

    def test_ingest_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            mp = pathlib.Path(td) / "nomanifest.jsonl"
            mp.write_text(json.dumps({"type": "tensor", **good_tensor()}) + "\n")
            with self.assertRaises(ValueError):
                stage_ingest(mp)


class ValidateTests(unittest.TestCase):
    def test_validate_ok(self):
        with tempfile.TemporaryDirectory() as td:
            mp = make_manifest(td, [good_tensor()])
            manifest, tensors = stage_ingest(mp)
            self.assertTrue(stage_validate(manifest, tensors, mp))

    def test_validate_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            mp = make_manifest(td, [good_tensor(), good_tensor()])
            manifest, tensors = stage_ingest(mp)
            with self.assertRaises(ValueError):
                stage_validate(manifest, tensors, mp)

    def test_validate_rejects_count_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            mp = make_manifest(td, [good_tensor()], tensor_count=99)
            manifest, tensors = stage_ingest(mp)
            with self.assertRaises(ValueError):
                stage_validate(manifest, tensors, mp)

    def test_validate_rejects_missing_field(self):
        with tempfile.TemporaryDirectory() as td:
            t = good_tensor()
            del t["shape"]
            mp = make_manifest(td, [t])
            manifest, tensors = stage_ingest(mp)
            with self.assertRaises(ValueError):
                stage_validate(manifest, tensors, mp)


class LayoutTests(unittest.TestCase):
    def test_alignment(self):
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), 64)
        self.assertEqual(align_up(64), 64)
        self.assertEqual(align_up(65), 128)

    def test_layout_offsets_aligned(self):
        tensors = [(1, good_tensor("a", [10], "F32")),
                   (2, good_tensor("b", [10], "F32"))]
        layout, total = stage_layout(tensors)
        for entry in layout:
            self.assertEqual(entry["staging_offset"] % STAGING_ALIGN, 0)
        self.assertEqual(total % STAGING_ALIGN, 0)


class SealTests(unittest.TestCase):
    def test_chain_links(self):
        manifest = {"model_id": "test"}
        layout = [
            {"name": "a", "shape": [2], "dtype": "F32",
             "staging_offset": 0, "byte_size": 8, "source_offset": 0},
            {"name": "b", "shape": [2], "dtype": "F32",
             "staging_offset": 64, "byte_size": 8, "source_offset": 100},
        ]
        seal_data = stage_seal(manifest, layout, 128)
        chain = seal_data["seal_chain"]
        # Each seal's prev must equal the previous seal
        self.assertEqual(chain[1]["prev"], chain[0]["seal"])
        # Genesis: first prev is hash of genesis string
        genesis = hashlib.sha256(b"worm-genesis:test").hexdigest()
        self.assertEqual(chain[0]["prev"], genesis)

    def test_tamper_detected(self):
        """Modifying a tensor name breaks its seal."""
        manifest = {"model_id": "test"}
        layout = [{"name": "a", "shape": [2], "dtype": "F32",
                   "staging_offset": 0, "byte_size": 8, "source_offset": 0}]
        seal_data = stage_seal(manifest, layout, 64)
        original_seal = seal_data["seal_chain"][0]["seal"]
        # Recompute with tampered name
        tampered = dict(layout[0], name="evil")
        seal_input = json.dumps({
            "name": tampered["name"], "shape": tampered["shape"],
            "dtype": tampered["dtype"],
            "staging_offset": tampered["staging_offset"],
            "byte_size": tampered["byte_size"],
            "prev": seal_data["seal_chain"][0]["prev"],
        }, sort_keys=True).encode()
        self.assertNotEqual(
            hashlib.sha256(seal_input).hexdigest(), original_seal)


class EndToEndTests(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            mp = make_manifest(td, [good_tensor("a"), good_tensor("b")])
            out = pathlib.Path(td) / "out.worm.jsonl"
            seal_data = train_one(mp, out)
            self.assertTrue(out.exists())
            lines = out.read_text().strip().split("\n")
            self.assertEqual(len(lines), 3)  # header + 2 tensors
            header = json.loads(lines[0])
            self.assertEqual(header["type"], "worm_archive")
            self.assertEqual(header["root_seal"], seal_data["root_seal"])


if __name__ == "__main__":
    unittest.main()
