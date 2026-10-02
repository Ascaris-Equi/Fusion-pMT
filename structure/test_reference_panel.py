"""CPU-only validation of the fixed reference panel and its cached input hashes."""

import json
import tempfile
import unittest
from pathlib import Path

from run_reference_panel import inputs, validate_case, sha256, completed_files


ROOT = Path(__file__).resolve().parent


class ReferencePanelTests(unittest.TestCase):
    def test_fixed_panel_and_domain_lengths(self):
        cases = inputs()
        self.assertEqual([c["pdb"] for c in cases], ["1OGA", "5WKH", "6UON"])
        self.assertEqual([sum(len(s) for s in c["chain_sequences"].values()) for c in cases], [412, 416, 411])
        self.assertEqual(len({c["mhc_allele"] for c in cases}), 3)
        for case in cases:
            self.assertEqual(list(case["chain_sequences"]), ["A", "B", "C", "D"])
            self.assertEqual(len(set(case["reference_chain_mapping"].values())), 4)
            for chain in ["A", "B"]:
                self.assertIn(case["cdr3_sequences"][chain], case["chain_sequences"][chain])
            for chain, (start, end) in case["canonical_sequence_intervals_1based"].items():
                self.assertEqual(end - start + 1, len(case["chain_sequences"][chain]))

    def test_reference_sequences_and_msa_queries(self):
        for case in inputs():
            reference = ROOT / f"cache/reference_panel/{case['pdb']}.pdb"
            if not reference.exists():
                self.skipTest("Download public reference PDBs with prepare_reference_panel.py first")
            a3m, validated_reference = validate_case(case)
            self.assertEqual(reference, validated_reference)
            manifest = json.loads(a3m.with_suffix(".manifest.json").read_text())
            self.assertEqual(sha256(a3m), manifest["a3m_sha256"])
            self.assertFalse(manifest["homolog_pairing"])
            self.assertEqual(manifest["rows_per_chain"]["C"], 1)
            self.assertEqual(manifest["public_input_pdb"], case["pdb"])

    def test_partial_prediction_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                completed_files(folder, "case")
            (folder / "case.done.txt").touch()
            (folder / "case_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_042.pdb").touch()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                completed_files(folder, "case")
            (folder / "case_scores_rank_001_alphafold2_multimer_v3_model_1_seed_042.json").touch()
            model, scores = completed_files(folder, "case")
            self.assertTrue(model.exists() and scores.exists())

    def test_saved_results_are_complete_and_consistent(self):
        output = ROOT / "results/reference_panel"
        if not (output / "metrics.json").exists():
            self.skipTest("Run and analyze the fixed panel first")
        run = json.loads((output / "run.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["effective_config"]["num_queries"], 3)
        self.assertEqual(run["effective_config"]["max_seq"], 64)
        self.assertEqual(run["effective_config"]["max_extra_seq"], 128)
        self.assertEqual([r["pdb"] for r in metrics["cases"]], ["1OGA", "5WKH", "6UON", "6ZKW"])
        expected_native = {"1OGA": 19, "5WKH": 22, "6UON": 11, "6ZKW": 16}
        for result in metrics["cases"]:
            contacts = result["geometry"]["pep_tcr_contacts"]["combined"]
            self.assertEqual(contacts["native"], expected_native[result["pdb"]])
            self.assertLessEqual(contacts["tp"], min(contacts["native"], contacts["pred"]))
            self.assertAlmostEqual(contacts["recall"], contacts["tp"] / contacts["native"])
            self.assertAlmostEqual(contacts["precision"], contacts["tp"] / contacts["pred"])
            self.assertEqual(sha256(output / f"{result['case_id']}_prediction.pdb"), result["prediction_sha256"])
            for mapping in result["geometry"]["chain_mapping"].values():
                self.assertGreaterEqual(mapping["input_ca_coverage"], .9)


if __name__ == "__main__":
    unittest.main()
