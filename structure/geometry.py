"""Sequence-mapped geometry diagnostics, not a binding score or DockQ implementation."""

from pathlib import Path

import numpy as np
from Bio import Align
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


ROLES = {"A": "tcra", "B": "tcrb", "C": "pep", "D": "mhca"}
CONTACT_CUTOFF = 5.0
INTERFACE_CUTOFF = 10.0


def _residues(chain):
    result = []
    for residue in chain:
        if residue.is_disordered() == 2:
            raise ValueError("Alternate residue identities require an explicit reference choice")
        if is_aa(residue, standard=True) and "CA" in residue:
            result.append(residue)
    return result


def _sequence(residues):
    return "".join(seq1(residue.resname) for residue in residues)


def _heavy_atoms(residue):
    atoms = {}
    for atom in residue.get_unpacked_list():
        if atom.element.strip().upper() in {"H", "D"}:
            continue
        name = atom.get_name()
        # One atom per name; highest occupancy, then blank/A/lexical altloc.
        preference = (-(atom.get_occupancy() or 0.0),
                      0 if atom.get_altloc() in {" ", "", "A"} else 1,
                      atom.get_altloc())
        if name not in atoms or preference < atoms[name][0]:
            atoms[name] = (preference, atom)
    return {name: np.asarray(value[1].coord, dtype=float)
            for name, value in atoms.items()}


def _noncollinear(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    return (len(coordinates) >= 3 and
            np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0), tol=1e-7) >= 2)


def _fit(moving, fixed):
    moving, fixed = np.asarray(moving, dtype=float), np.asarray(fixed, dtype=float)
    if not _noncollinear(moving) or not _noncollinear(fixed):
        return None
    moving_center, fixed_center = moving.mean(axis=0), fixed.mean(axis=0)
    left, _, right = np.linalg.svd((moving - moving_center).T @ (fixed - fixed_center))
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(left @ right) >= 0 else -1.0
    rotation = left @ correction @ right
    translation = fixed_center - moving_center @ rotation
    return rotation, translation


def _rmsd(moving, fixed, transform):
    if transform is None:
        return None
    rotation, translation = transform
    delta = np.asarray(moving) @ rotation + translation - np.asarray(fixed)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _contact_set(mapped, left_chains, right_chains, side, cutoff):
    contacts = set()
    for left_chain in left_chains:
        for right_chain in right_chains:
            for left_index, left_pair in mapped[left_chain].items():
                left = np.array(list(_heavy_atoms(left_pair[side]).values()))
                for right_index, right_pair in mapped[right_chain].items():
                    right = np.array(list(_heavy_atoms(right_pair[side]).values()))
                    distances_squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
                    if np.any(distances_squared < cutoff * cutoff):
                        contacts.add((left_chain, left_index, right_chain, right_index))
    return contacts


def _contact_statistics(native, predicted):
    common = native & predicted
    n_native, n_predicted, true_positive = len(native), len(predicted), len(common)
    return {
        "native": n_native, "pred": n_predicted, "tp": true_positive,
        "precision": true_positive / n_predicted if n_predicted else None,
        "recall": true_positive / n_native if n_native else None,
        "fnat": true_positive / n_native if n_native else None,
        "f1": 2 * true_positive / (n_native + n_predicted) if n_native + n_predicted else None,
    }


def analyze_geometry(predicted_path, reference_path, chain_sequences,
                     reference_chain_map, min_coverage=0.75):
    """Evaluate A=alpha, B=beta, C=peptide, D=MHC against explicit native chains.

    RMSDs and contacts use only sequence-identical, CA-resolved residue positions.
    Interface CA RMSD is a custom metric, not standard backbone DockQ iRMSD.
    Native coordinates define interfaces for evaluation only, not pose selection.
    """
    if set(chain_sequences) != set(ROLES) or set(reference_chain_map) != set(ROLES):
        raise ValueError("Exactly A/B/C/D sequence and reference mappings are required")
    if len(set(reference_chain_map.values())) != 4:
        raise ValueError("Reference chain mapping must be one-to-one")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    parser = PDBParser(QUIET=True)
    predicted = parser.get_structure("pred", str(Path(predicted_path)))[0]
    reference = parser.get_structure("ref", str(Path(reference_path)))[0]
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
    mapped, mapping_report, ca_coordinates = {}, {}, {}
    for chain in ROLES:
        prs = _residues(predicted[chain])
        rrs = _residues(reference[reference_chain_map[chain]])
        supplied = chain_sequences[chain]
        if not supplied or _sequence(prs) != supplied:
            raise ValueError(f"Predicted chain {chain} does not exactly match the supplied sequence")
        if not rrs:
            raise ValueError(f"No resolved reference residues for chain {chain}")
        alignment = aligner.align(supplied, _sequence(rrs))[0]
        pairs, aligned_positions = {}, 0
        for (pstart, pend), (rstart, rend) in zip(*alignment.aligned):
            if pend - pstart != rend - rstart:
                raise ValueError("Unexpected alignment block lengths")
            for pi, ri in zip(range(pstart, pend), range(rstart, rend)):
                aligned_positions += 1
                if prs[pi].resname == rrs[ri].resname:
                    pairs[pi + 1] = (prs[pi], rrs[ri])
        coverage = len(pairs) / len(supplied)
        if coverage < min_coverage:
            raise ValueError(f"Insufficient identical reference coverage for chain {chain}: {coverage:.3f}")
        mapped[chain] = pairs
        ca_coordinates[chain] = (
            np.array([pair[0]["CA"].coord for pair in pairs.values()], dtype=float),
            np.array([pair[1]["CA"].coord for pair in pairs.values()], dtype=float),
        )
        predicted_atom_count = sum(len(_heavy_atoms(pair[0])) for pair in pairs.values())
        reference_atom_count = sum(len(_heavy_atoms(pair[1])) for pair in pairs.values())
        common_atom_count = sum(len(set(_heavy_atoms(pair[0])) & set(_heavy_atoms(pair[1])))
                                for pair in pairs.values())
        mapping_report[chain] = {
            "role": ROLES[chain], "predicted_chain": chain,
            "reference_chain": reference_chain_map[chain], "input_length": len(supplied),
            "reference_resolved_ca_length": len(rrs), "matched_ca": len(pairs),
            "input_ca_coverage": coverage,
            "aligned_sequence_identity": len(pairs) / aligned_positions if aligned_positions else 0.0,
            "unmatched_input_positions": [i for i in range(1, len(supplied) + 1) if i not in pairs],
            "predicted_heavy_atoms_at_mapped_positions": predicted_atom_count,
            "reference_heavy_atoms_at_mapped_positions": reference_atom_count,
            "common_heavy_atom_names_at_mapped_positions": common_atom_count,
            "reference_common_heavy_atom_coverage": common_atom_count / predicted_atom_count,
            "residue_correspondence": [
                {"input_position": index, "reference_number": int(pair[1].id[1]),
                 "reference_insertion_code": pair[1].id[2].strip()}
                for index, pair in pairs.items()
            ],
        }
    self_fit = {chain: _rmsd(*coordinates, _fit(*coordinates))
                for chain, coordinates in ca_coordinates.items()}
    mhc_transform = _fit(*ca_coordinates["D"])
    if mhc_transform is None:
        raise ValueError("MHC fitting requires at least three non-collinear mapped CA atoms")
    mhc_aligned = {chain: _rmsd(*coordinates, mhc_transform)
                   for chain, coordinates in ca_coordinates.items()}
    contact_report, interface_report = {}, {}
    for name, tcr_chains in {"alpha": ["A"], "beta": ["B"], "combined": ["A", "B"]}.items():
        native = _contact_set(mapped, tcr_chains, ["C"], 1, CONTACT_CUTOFF)
        model = _contact_set(mapped, tcr_chains, ["C"], 0, CONTACT_CUTOFF)
        contact_report[name] = _contact_statistics(native, model)
        interface_contacts = _contact_set(mapped, tcr_chains, ["C"], 1, INTERFACE_CUTOFF)
        positions = set()
        for left_chain, left_index, right_chain, right_index in interface_contacts:
            positions.add((left_chain, left_index))
            positions.add((right_chain, right_index))
        selected = [mapped[chain][index] for chain, index in sorted(positions)]
        model_ca = np.array([pair[0]["CA"].coord for pair in selected], dtype=float)
        native_ca = np.array([pair[1]["CA"].coord for pair in selected], dtype=float)
        transform = _fit(model_ca, native_ca) if selected else None
        interface_report[name] = {
            "matched_ca_count": len(selected),
            "residues_per_chain": {chain: sum(c == chain for c, _ in positions)
                                   for chain in tcr_chains + ["C"]},
            "rmsd_angstrom": _rmsd(model_ca, native_ca, transform),
        }
    return {
        "metric_protocol": {
            "version": "1.0", "contact_cutoff_angstrom": CONTACT_CUTOFF,
            "interface_cutoff_angstrom": INTERFACE_CUTOFF, "cutoff_comparison": "strictly less than",
            "contact_unit": "unique residue pairs, not atom pairs",
            "atom_selection": "observed non-hydrogen atoms; one highest-occupancy altloc per atom name",
            "coverage_policy": "sequence-identical mapped positions with CA in both structures only",
            "minimum_input_ca_coverage": min_coverage,
            "missing_atom_limitation": "unresolved native sidechains may reduce native contacts; heavy-atom coverage is reported",
            "interface_rmsd_definition": "joint self-fit of native-interface CA atoms, custom metric, not DockQ iRMSD",
            "reference_use": "post-prediction geometry evaluation only; no reference-based pose selection",
        },
        "chain_mapping": mapping_report,
        "per_chain_self_fit_ca_rmsd": self_fit,
        "mhc_aligned_ca_rmsd": mhc_aligned,
        "pep_tcr_contacts": contact_report,
        "native_ca_interface_rmsd_10a": interface_report,
    }
