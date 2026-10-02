"""CPU-only geometry tests; temporary structures do not alter published outputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from geometry import analyze_geometry, _contact_set, _contact_statistics, _fit, _rmsd


SEQUENCES = {"A": "AAA", "B": "AAA", "C": "AAA", "D": "AAA"}
MAPPING = {chain: chain for chain in SEQUENCES}


def write_structure(path, offsets=None, rotation=None, translation=None, missing=None):
    offsets = offsets or {}
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    missing = missing or set()
    anchors = {"A": np.array([0., 0., 0.]), "B": np.array([0., 0., 15.]),
               "C": np.array([0., 0., 3.]), "D": np.array([0., 0., 30.])}
    ca_shape = [np.array([0., 0., 0.]), np.array([4., 0., 0.]), np.array([0., 4., 0.])]
    lines, serial = [], 1
    for chain in SEQUENCES:
        for residue_number, ca in enumerate(ca_shape, 1):
            if (chain, residue_number) in missing:
                continue
            for atom_name, displacement, element in [
                ("N", [-.3, 0, 0], "N"), ("CA", [0, 0, 0], "C"),
                ("C", [.3, 0, 0], "C"), ("O", [.6, 0, 0], "O"),
                ("CB", [0, .3, 0], "C"),
            ]:
                xyz = (anchors[chain] + ca + displacement + offsets.get(chain, np.zeros(3))) @ rotation + translation
                lines.append(f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue_number:4d}    "
                             f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.:6.2f}{90.:6.2f}          {element:>2s}")
                serial += 1
        lines.append("TER")
    path.write_text("\n".join(lines) + "\nEND\n")


class GeometryTests(unittest.TestCase):
    def test_heavy_atom_cutoff_and_unique_residue_pair(self):
        from Bio.PDB.Atom import Atom
        from Bio.PDB.Residue import Residue

        def residue(atoms):
            result = Residue((" ", 1, " "), "ALA", " ")
            for serial, (name, xyz, element) in enumerate(atoms, 1):
                result.add(Atom(name, np.asarray(xyz, dtype=float), 0., 1.,
                                " ", f"{name:^4s}", serial, element=element))
            return result

        left = residue([("CA", [0., 0., 0.], "C"), ("H", [4.99, 0., 0.], "H")])
        boundary = residue([("CA", [5., 0., 0.], "C")])
        mapped = {"A": {1: (left, left)}, "C": {1: (boundary, boundary)}}
        self.assertEqual(_contact_set(mapped, ["A"], ["C"], 0, 5.), set())
        left = residue([("CA", [0., 0., 0.], "C"), ("CB", [.1, 0., 0.], "C")])
        nearby = residue([("CA", [4.9, 0., 0.], "C"), ("CB", [4.8, 0., 0.], "C")])
        mapped = {"A": {1: (left, left)}, "C": {1: (nearby, nearby)}}
        self.assertEqual(_contact_set(mapped, ["A"], ["C"], 0, 5.), {("A", 1, "C", 1)})

    def test_rigid_transform_is_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            native, model = Path(directory) / "native.pdb", Path(directory) / "model.pdb"
            write_structure(native)
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            write_structure(model, rotation=rotation, translation=np.array([11., -5., 7.]))
            result = analyze_geometry(model, native, SEQUENCES, MAPPING)
            for metric in ["per_chain_self_fit_ca_rmsd", "mhc_aligned_ca_rmsd"]:
                for value in result[metric].values():
                    self.assertLess(value, 1e-6)
            contacts = result["pep_tcr_contacts"]["combined"]
            self.assertEqual(contacts["tp"], contacts["native"])
            self.assertEqual(contacts["f1"], 1.)
            self.assertLess(result["native_ca_interface_rmsd_10a"]["combined"]["rmsd_angstrom"], 1e-6)

    def test_docking_displacement_does_not_change_self_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            native, model = Path(directory) / "native.pdb", Path(directory) / "model.pdb"
            write_structure(native)
            write_structure(model, offsets={"A": np.array([0., 0., -20.])})
            result = analyze_geometry(model, native, SEQUENCES, MAPPING)
            self.assertLess(result["per_chain_self_fit_ca_rmsd"]["A"], 1e-6)
            self.assertAlmostEqual(result["mhc_aligned_ca_rmsd"]["A"], 20., places=6)
            contacts = result["pep_tcr_contacts"]["alpha"]
            self.assertGreater(contacts["native"], 0)
            self.assertEqual(contacts["pred"], 0)
            self.assertEqual(contacts["recall"], 0.)
            self.assertIsNone(contacts["precision"])

    def test_invalid_sequence_or_chain_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            native, model = Path(directory) / "native.pdb", Path(directory) / "model.pdb"
            write_structure(native)
            write_structure(model)
            wrong = dict(SEQUENCES, A="GGG")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                analyze_geometry(model, native, wrong, MAPPING)
            with self.assertRaisesRegex(ValueError, "one-to-one"):
                analyze_geometry(model, native, SEQUENCES, dict(MAPPING, A="B"))

    def test_insufficient_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            native, model = Path(directory) / "native.pdb", Path(directory) / "model.pdb"
            write_structure(native, missing={("A", 3)})
            write_structure(model)
            with self.assertRaisesRegex(ValueError, "coverage"):
                analyze_geometry(model, native, SEQUENCES, MAPPING)

    def test_empty_contact_denominators_and_degenerate_fit(self):
        result = _contact_statistics(set(), set())
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])
        line = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        self.assertIsNone(_fit(line, line))
        self.assertIsNone(_rmsd(line, line, None))

    def test_existing_6zkw_msa_regression(self):
        root = Path(__file__).resolve().parent
        native = root / "cache/tcrmodel2/6ZKW.pdb"
        model = root / "results/af2_msa_6zkw_prediction.pdb"
        if not native.exists() or not model.exists():
            self.skipTest("Existing public reference/cache not available")
        import json
        info = json.loads((root / "inputs/6kzw_domain_complex.json").read_text())
        result = analyze_geometry(model, native, info["chain_sequences"],
                                  {"A": "D", "B": "E", "C": "C", "D": "A"})
        expected = {"A": 6.9455977466, "B": 5.4477622244,
                    "C": .5708172417, "D": .7770803413}
        for chain, value in expected.items():
            self.assertAlmostEqual(result["mhc_aligned_ca_rmsd"][chain], value, places=5)
        self.assertEqual(result["chain_mapping"]["A"]["matched_ca"], 112)
        self.assertEqual(result["chain_mapping"]["B"]["matched_ca"], 114)
        self.assertEqual(result["chain_mapping"]["C"]["matched_ca"], 9)
        self.assertLess(result["per_chain_self_fit_ca_rmsd"]["A"], result["mhc_aligned_ca_rmsd"]["A"])


if __name__ == "__main__":
    unittest.main()
