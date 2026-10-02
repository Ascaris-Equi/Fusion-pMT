"""CPU-only checks for the pilot's aggregation and saved results."""

import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import run_tutorial
import run_af2_pilot


ROOT = Path(__file__).resolve().parent


class PilotTests(unittest.TestCase):
    def test_aggregation_sample_std_and_exclusions(self):
        rows = pd.DataFrame({
            "complex_id": ["x", "x"], "af3_output_directory": ["dir", "dir"],
            "signal": [1., 3.], "af3_seed": [1, 1], "af3_sample": [0, 1],
            "af3_ranking_score": [.2, .3],
        })
        result = run_tutorial.aggregate(rows)
        self.assertAlmostEqual(result.signal.iloc[0], 2.)
        self.assertAlmostEqual(result.signal_std.iloc[0], math.sqrt(2.))
        self.assertFalse(any(c.startswith("af3_seed") for c in result))
        self.assertFalse(any(c.startswith("af3_sample") for c in result))
        self.assertFalse(any(c.startswith("af3_ranking") for c in result))

    def test_tampered_cached_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            source = run_tutorial.FILES["input"]
            target = cache / source
            target.parent.mkdir(parents=True)
            target.write_bytes(b"untrusted replacement")
            with self.assertRaisesRegex(ValueError, "pinned Git blob"):
                run_tutorial.fetch(source, cache)

    def test_committed_msa_integrity(self):
        info = json.loads((ROOT / "inputs/6kzw_domain_complex.json").read_text())
        manifest = json.loads((ROOT / "inputs/msa_manifest.json").read_text())
        content = (ROOT / "inputs/6zkw_domain_msa.a3m").read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), manifest["a3m_sha256"])
        lines = content.decode().splitlines()
        sequences = list(info["chain_sequences"].values())
        self.assertEqual(lines[0], "#112,115,9,180\t1,1,1,1")
        self.assertEqual(lines[2], "".join(sequences))
        self.assertEqual((len(lines) - 1) % 2, 0)
        for header, sequence in zip(lines[1::2], lines[2::2]):
            self.assertTrue(header.startswith(">"))
            self.assertEqual(len(re.sub("[a-z]", "", sequence)), sum(map(len, sequences)))

    def test_saved_tutorial_metrics(self):
        payload = json.loads((ROOT / "results/tutorial_results.json").read_text())
        pairs = payload["pairs"]
        self.assertEqual(len(pairs), 18)
        self.assertEqual(len({p["complex_id"] for p in pairs}), 18)
        labels = np.array([p["recognition_label"] for p in pairs])
        self.assertEqual(int(labels.sum()), 6)
        for model, metrics in payload["metrics"].items():
            scores = np.array([p[model] for p in pairs])
            self.assertTrue(np.isfinite(scores).all())
            self.assertAlmostEqual(roc_auc_score(labels, scores), metrics["roc_auc"])
            self.assertAlmostEqual(average_precision_score(labels, scores), metrics["average_precision"])
            self.assertAlmostEqual(roc_auc_score(labels, scores, max_fpr=.1), metrics["standardized_pauc_fpr_0_1"])
            # Direct Mann-Whitney pair comparison, independent of sklearn AUC.
            differences = scores[labels == 1, None] - scores[labels == 0]
            direct_auc = np.mean((differences > 0) + .5 * (differences == 0))
            self.assertAlmostEqual(float(direct_auc), metrics["roc_auc"])
        self.assertLess(payload["official_prediction_max_abs_error"], 1e-10)

    def test_effective_msa_model_depth(self):
        run = json.loads((ROOT / "results/af2_msa_run.json").read_text())
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["effective_config"]["max_seq"], 64)
        self.assertEqual(run["effective_config"]["max_extra_seq"], 128)
        self.assertEqual(run["effective_config"]["msa_mode"], "mmseqs2_uniref_env")
        self.assertEqual(run["model_weight_sha256"],
                         "611da8fc7478928f68de12e8b226260ef1f4ce62bcc29b008572e52f4f212959")

    def test_completed_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            (root / "inputs/6zkw_domain_msa.a3m").write_text("test")
            raw = root / "results/af2_raw/msa_unpaired"
            raw.mkdir(parents=True)
            (raw / "test.done.txt").touch()
            record = root / "results/af2_msa_run.json"
            record.write_text('{"elapsed_seconds": 148.18387579917908}')
            original = record.read_bytes()
            with patch.object(run_af2_pilot, "ROOT", root), patch.object(
                run_af2_pilot, "prepare", return_value=(root / "input.fasta", {}, None)
            ):
                with self.assertRaises(FileExistsError):
                    run_af2_pilot.predict(SimpleNamespace(protocol="msa"))
            self.assertEqual(record.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
