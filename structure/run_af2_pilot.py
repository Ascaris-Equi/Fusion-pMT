"""Run one public reference complex with existing ColabFold, without altering it."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV = Path("/home/ubantu/Achieve/ImmuReduceR/localcolabfold/colabfold-conda")
DEFAULT_DATA = Path("/home/ubantu/Achieve/ImmuReduceR/localcolabfold/colabfold")
SOURCE_COMMIT = "90536deaf766fe169f51679ce9456f3b165be5c9"


def fetch(url, target):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "structure-pilot"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    return target


def prepare():
    cache = ROOT / "cache" / "tcrmodel2"
    commit = SOURCE_COMMIT
    url = f"https://raw.githubusercontent.com/piercelab/tcrmodel2/{commit}/README.md"
    readme = fetch(url, cache / "README.md").read_text()
    example = readme.split("--job_id=test_clsI_6kzw", 1)[1].split("```", 1)[0]
    roles = {"A": "tcra", "B": "tcrb", "C": "pep", "D": "mhca"}
    sequences = {}
    for chain, role in roles.items():
        match = re.search(rf"--{role}_seq=([A-Z]+)", example)
        if not match:
            raise ValueError(f"No published sequence for {role}")
        sequences[chain] = match.group(1)
    # The upstream example job_id has transposed letters: 6KZW is unrelated.
    # The published alpha/beta/peptide/MHC sequences correspond to 6ZKW.
    reference = fetch("https://files.rcsb.org/download/6ZKW.pdb", cache / "6ZKW.pdb")
    if not reference.read_text().startswith(("HEADER", "TITLE")):
        raise ValueError("Invalid PDB download")
    inputs = ROOT / "inputs"
    inputs.mkdir(exist_ok=True)
    fasta = inputs / "6kzw_domain_complex.fasta"
    fasta.write_text(">6kzw_domain_complex\n" + ":".join(sequences.values()) + "\n")
    info = {
        "pdb": "6ZKW", "upstream_example_job_id": "test_clsI_6kzw",
        "sequence_source": url, "source_commit": commit,
        "source_readme_sha256": hashlib.sha256(readme.encode()).hexdigest(),
        "reference_url": "https://files.rcsb.org/download/6ZKW.pdb",
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "chain_roles": roles, "chain_sequences": sequences,
        "scope": "published TCRmodel2 domain inputs only; not TCRmodel2 algorithm reproduction",
        "msa_mode": "single_sequence", "templates": False,
        "training_overlap": "model training and nearby-structure overlap not systematically excluded; single-reference pilot, not independent validation",
    }
    (inputs / "6kzw_domain_complex.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"input": str(fasta), "chain_lengths": {k: len(v) for k, v in sequences.items()}}))
    return fasta, info, reference


def predict(args):
    fasta, info, _ = prepare()
    if args.protocol == "msa":
        fasta = ROOT / "inputs" / "6zkw_domain_msa.a3m"
        if not fasta.exists():
            raise FileNotFoundError("Run fetch_msa.py before the MSA protocol")
        info = dict(info, msa_mode="server_env_unpaired", homolog_pairing=False)
    raw = ROOT / "results" / "af2_raw"
    if args.protocol == "msa":
        raw = raw / "msa_unpaired"
    prefix = "af2_msa" if args.protocol == "msa" else "af2"
    raw.mkdir(parents=True, exist_ok=True)
    if list(raw.glob("*.done.txt")) or list(raw.glob("*_scores_rank_001_*.json")):
        raise FileExistsError("Completed predictions already exist; analyze them or use a fresh output directory. The original run record is preserved.")
    env = dict(os.environ)
    env.update({
        "LD_LIBRARY_PATH": str(args.env / "lib") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.35",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    command = [str(args.env / "bin/python"), "-m", "colabfold.batch", str(fasta), str(raw),
               "--data", str(args.data), "--model-type", "alphafold2_multimer_v3",
               "--num-models", "1", "--model-order", "1", "--num-seeds", "1",
               "--random-seed", "42", "--num-recycle", "3", "--msa-mode",
               "mmseqs2_uniref_env" if args.protocol == "msa" else "single_sequence",
               "--max-msa", "64:128" if args.protocol == "msa" else "16:32",
               "--num-relax", "0", "--disable-unified-memory"]
    started = time.time()
    record = {"command": command, "input_sha256": hashlib.sha256(fasta.read_bytes()).hexdigest(),
              "environment": {k: env[k] for k in ["XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS"]}, "protocol": info, "status": "running"}
    record_path = ROOT / "results" / f"{prefix}_run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        with (raw / "launcher.log").open("a") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        record.update(status="failed", returncode=error.returncode, elapsed_seconds=time.time() - started)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise
    config = json.loads((raw / "config.json").read_text())
    if args.protocol == "msa" and (config["max_seq"] <= 1 or config["max_extra_seq"] <= 1):
        record.update(status="failed", error="MSA depth was unexpectedly truncated", effective_config=config)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise ValueError("MSA inference was unexpectedly truncated to single-sequence depth")
    weight_file = args.data / "params" / "params_model_1_multimer_v3.npz"
    record.update(status="completed", elapsed_seconds=time.time() - started,
                  effective_config=config,
                  model_weight_sha256=hashlib.sha256(weight_file.read_bytes()).hexdigest())
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"]}))


def analyze(args):
    import numpy as np
    from Bio import Align
    from Bio.PDB import PDBParser, Superimposer
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    _, info, reference = prepare()
    raw = ROOT / "results" / "af2_raw"
    prefix = "af2_msa" if args.protocol == "msa" else "af2"
    if args.protocol == "msa":
        raw = raw / "msa_unpaired"
    models = list(raw.glob("*_unrelaxed_rank_001_*.pdb"))
    scores = list(raw.glob("*_scores_rank_001_*.json"))
    if len(models) != 1 or len(scores) != 1:
        raise ValueError("Expected one complete rank-one prediction and score file")
    summary = json.loads(scores[0].read_text())
    parser = PDBParser(QUIET=True)
    predicted = parser.get_structure("pred", models[0])[0]
    truth = parser.get_structure("ref", reference)[0]

    def residues(chain):
        return [r for r in chain if is_aa(r, standard=True) and "CA" in r]

    def sequence(rs):
        return "".join(seq1(r.resname) for r in rs)

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    pairs, mappings = {}, {}
    for chain, role in info["chain_roles"].items():
        prs = residues(predicted[chain])
        if sequence(prs) != info["chain_sequences"][chain]:
            raise ValueError(f"Unexpected predicted sequence in chain {chain}")
        candidates = []
        for reference_chain in truth:
            rrs = residues(reference_chain)
            if not rrs:
                continue
            alignment = aligner.align(sequence(prs), sequence(rrs))[0]
            candidates.append((alignment.score, reference_chain.id, rrs, alignment))
        _, reference_chain, rrs, alignment = max(candidates, key=lambda x: x[0])
        atom_pairs = []
        for (pstart, pend), (rstart, rend) in zip(*alignment.aligned):
            if pend - pstart != rend - rstart:
                raise ValueError("Unexpected alignment block")
            for pi, ri in zip(range(pstart, pend), range(rstart, rend)):
                if prs[pi].resname == rrs[ri].resname:
                    atom_pairs.append((prs[pi]["CA"], rrs[ri]["CA"]))
        if len(atom_pairs) < 0.75 * len(prs):
            raise ValueError(f"Insufficient coordinate alignment coverage for {role}")
        pairs[chain] = atom_pairs
        mappings[role] = {"predicted_chain": chain, "reference_chain": reference_chain, "matched_ca": len(atom_pairs), "input_length": len(prs)}
    if len({v["reference_chain"] for v in mappings.values()}) != 4:
        raise ValueError("Reference chain mapping is not one-to-one")
    fit = Superimposer()
    fit.set_atoms([b for a, b in pairs["D"]], [a for a, b in pairs["D"]])
    rotation, translation = fit.rotran

    def rmsd(atom_pairs):
        delta = np.array([a.coord @ rotation + translation - b.coord for a, b in atom_pairs])
        return float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))

    lengths = [len(s) for s in info["chain_sequences"].values()]
    bounds = np.cumsum([0] + lengths)
    pae = np.asarray(summary["pae"])
    if pae.shape != (sum(lengths), sum(lengths)):
        raise ValueError("PAE matrix length does not match the four input chains")
    tcr_indices = np.arange(bounds[0], bounds[2])
    peptide_indices = np.arange(bounds[2], bounds[3])
    peptide_tcr_pae = float(np.mean(np.concatenate([pae[np.ix_(peptide_indices, tcr_indices)].ravel(), pae[np.ix_(tcr_indices, peptide_indices)].ravel()])))
    result = {
        "scope": "new local AF2-Multimer reference pilot; not binding classification",
        "msa_protocol": args.protocol, "templates": False,
        "pdb": "6ZKW", "model": "alphafold2_multimer_v3_model_1", "seed": 42,
        "num_recycles_requested": 3, "ptm": float(summary["ptm"]), "iptm": float(summary["iptm"]),
        "mean_plddt": float(np.mean(summary["plddt"])), "peptide_tcr_mean_bidirectional_pae_angstrom": peptide_tcr_pae,
        "mhc_aligned_ca_rmsd_angstrom": {role: rmsd(pairs[chain]) for chain, role in info["chain_roles"].items()},
        "chain_mapping": mappings, "reference_training_overlap": info["training_overlap"],
    }
    out = ROOT / "results"
    run_file = out / f"{prefix}_run.json"
    if run_file.exists():
        run_record = json.loads(run_file.read_text())
        # Correct only the reference metadata; the actual command is unchanged.
        run_record["protocol"]["pdb"] = "6ZKW"
        run_record["protocol"]["reference_url"] = info["reference_url"]
        run_record["protocol"]["reference_sha256"] = info["reference_sha256"]
        run_record["protocol"]["training_overlap"] = info["training_overlap"]
        run_record["protocol"]["upstream_example_job_id"] = "test_clsI_6kzw"
        run_file.write_text(json.dumps(run_record, indent=2) + "\n")
    shutil.copy2(models[0], out / f"{prefix}_6zkw_prediction.pdb")
    (out / f"{prefix}_structure_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    protocol_text = "服务器 unpaired MSA、无模板" if args.protocol == "msa" else "single-sequence、无模板"
    lines = ["# 本机新结构预测：AF2-Multimer 小试跑", "", f"PDB 6ZKW 的公开域序列；{protocol_text}。训练/近邻结构重叠未系统排除，不能当独立 OOD 验证。ipTM、pLDDT 和 PAE 不是结合概率。", "", "| 指标 | 本机实际结果 |", "| --- | --- |", f"| pTM（score JSON，保留两位） | {result['ptm']:.2f} |", f"| ipTM（score JSON，保留两位） | {result['iptm']:.2f} |", f"| 平均 pLDDT | {result['mean_plddt']:.2f} |", f"| 肽↔TCR 平均双向 PAE / Å | {peptide_tcr_pae:.2f} |"]
    for role, value in result["mhc_aligned_ca_rmsd_angstrom"].items():
        lines.append(f"| MHC 对齐后 {role} Cα RMSD / Å | {value:.2f} |")
    lines += ["", "输入域序列来自 TCRmodel2 官方示例，但推理程序是 ColabFold AF2-Multimer，并非复现 TCRmodel2 的专门化算法。上游 job_id 的 6kzw 是字母转置，真实参考为 [6ZKW](https://www.rcsb.org/structure/6ZKW)；原始任务文件名保留上游别名。参考 PDB 只在预测之后用于坐标比较，不输入为模板。", "", f"[预测结构]({prefix}_6zkw_prediction.pdb) · [坐标与界面指标]({prefix}_structure_metrics.json) · [实际命令和耗时]({prefix}_run.json)"]
    report = "AF2_MSA_PILOT_RESULTS.md" if args.protocol == "msa" else "AF2_PILOT_RESULTS.md"
    (out / report).write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "predict", "analyze"])
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--protocol", choices=["single", "msa"], default="single")
    args = parser.parse_args()
    {"prepare": prepare, "predict": lambda: predict(args), "analyze": lambda: analyze(args)}[args.action]()
