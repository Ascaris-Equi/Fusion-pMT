"""Select a fixed three-complex panel from the pinned official TCRdock manifest."""

import csv
import hashlib
import io
import json
from pathlib import Path

from run_af2_pilot import fetch


ROOT = Path(__file__).resolve().parent
COMMIT = "c5a7af42eeb0c2a4492a4d4fe803f1f9aafb6193"
SOURCE_SHA256 = "93359888501e34b156293db6649cb001b03cb182b72f837984eaf6d86db25c6d"
CASES = {"1oga": {"A": "D", "B": "E", "C": "C", "D": "A"},
         "5wkh": {"A": "D", "B": "E", "C": "C", "D": "A"},
         "6uon": {"A": "I", "B": "J", "C": "C", "D": "A"}}
CANONICAL_INTERVALS = {
    "1oga": {"A": [2, 111], "B": [5, 116], "C": [1, 9], "D": [1, 181]},
    "5wkh": {"A": [1, 109], "B": [1, 116], "C": [1, 10], "D": [1, 181]},
    "6uon": {"A": [4, 112], "B": [3, 113], "C": [1, 10], "D": [2, 182]},
}


def main():
    url = f"https://raw.githubusercontent.com/phbradley/TCRdock/{COMMIT}/tcrdock/db/ternary_templates_v2.tsv"
    source = fetch(url, ROOT / "cache/tcrdock/ternary_templates_v2.tsv")
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("TCRdock input manifest does not match pinned SHA256")
    rows = list(csv.DictReader(io.StringIO(source.read_text()), delimiter="\t"))
    destination = ROOT / "inputs/panel"
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, reference_mapping in CASES.items():
        matches = [r for r in rows if r["pdbid"] == case_id]
        if len(matches) != 1:
            raise ValueError("Expected one input row per fixed PDB")
        row = matches[0]
        if row["organism"] != "human" or row["mhc_class"] != "1":
            raise ValueError("Panel is restricted to human MHCI")
        mhc, peptide, alpha, beta = row["chainseq"].split("/")
        sequences = {"A": alpha, "B": beta, "C": peptide, "D": mhc}
        if row["cdr3a"] not in alpha or row["cdr3b"] not in beta or peptide != row["pep_seq"]:
            raise ValueError("Published domain/CDR3/peptide mismatch")
        if any(set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in sequences.values()):
            raise ValueError("Non-standard amino acid in input")
        pdb = case_id.upper()
        reference_url = f"https://files.rcsb.org/download/{pdb}.pdb"
        reference = fetch(reference_url, ROOT / f"cache/reference_panel/{pdb}.pdb")
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb}"
        entry = json.loads(fetch(entry_url, ROOT / f"cache/reference_panel/{pdb}.entry.json").read_text())
        info = {
            "case_id": case_id, "pdb": pdb, "mhc_allele": "HLA-" + row["mhc_allele"],
            "release_date": entry["rcsb_accession_info"]["initial_release_date"],
            "primary_citation": entry.get("rcsb_primary_citation", {}),
            "sequence_source": url, "source_commit": COMMIT,
            "source_manifest_sha256": SOURCE_SHA256, "source_pdbfile": row["pdbfile"],
            "reference_url": reference_url, "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "entry_url": entry_url, "chain_roles": {"A": "tcra", "B": "tcrb", "C": "pep", "D": "mhca"},
            "chain_sequences": sequences, "reference_chain_mapping": reference_mapping,
            "chain_id_convention": "original PDB auth chain IDs, not mmCIF label_asym_id",
            "canonical_sequence_intervals_1based": CANONICAL_INTERVALS[case_id],
            "cdr3_sequences": {"A": row["cdr3a"], "B": row["cdr3b"]},
            "scope": "fixed geometry reference panel, not binding classification; public TCRdock cropped domains, not TCRdock algorithm reproduction",
            "beta2m_input": False, "templates": False,
            "training_overlap": "model structure-training/near-neighbor overlap not excluded; not independent generalization validation",
        }
        info_path = destination / f"{case_id}.json"
        if info_path.exists():
            previous = json.loads(info_path.read_text())
            for key in ["reference_sha256", "chain_sequences", "reference_chain_mapping", "source_commit"]:
                if previous[key] != info[key]:
                    raise ValueError(f"Frozen case changed ({case_id}, {key}); do not silently refresh the reference protocol")
        info_path.write_text(json.dumps(info, indent=2) + "\n", newline="\n")
        (destination / f"{case_id}.fasta").write_text(">" + case_id + "\n" + ":".join(sequences.values()) + "\n", newline="\n")
        cases.append({"case_id": case_id, "pdb": pdb, "mhc_allele": info["mhc_allele"],
                      "input_json": f"{case_id}.json", "release_date": info["release_date"],
                      "chain_lengths": {c: len(s) for c, s in sequences.items()}})
    manifest = {"selection": "fixed 1OGA/5WKH/6UON before inference; three different human MHCI alleles; no result-based exclusion",
                "source_commit": COMMIT, "source_manifest_sha256": SOURCE_SHA256, "cases": cases}
    (destination / "panel.json").write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
