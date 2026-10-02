"""Run a prespecified multi-reference AF2 geometry pilot, not a binding benchmark."""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from run_af2_pilot import DEFAULT_DATA, DEFAULT_ENV


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inputs():
    manifest = json.loads((ROOT / "inputs/panel/panel.json").read_text())
    cases = [json.loads((ROOT / "inputs/panel" / c["input_json"]).read_text()) for c in manifest["cases"]]
    if [c["case_id"] for c in cases] != ["1oga", "5wkh", "6uon"]:
        raise ValueError("The geometry panel must remain fixed before inference")
    return cases


def validate_case(case):
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    reference = ROOT / f"cache/reference_panel/{case['pdb']}.pdb"
    if sha256(reference) != case["reference_sha256"]:
        raise ValueError("Reference PDB checksum mismatch")
    structure = PDBParser(QUIET=True).get_structure("reference", reference)[0]
    for chain, sequence in case["chain_sequences"].items():
        auth_chain = case["reference_chain_mapping"][chain]
        observed = "".join(seq1(r.resname) for r in structure[auth_chain]
                           if is_aa(r, standard=True) and "CA" in r)
        if sequence not in observed:
            raise ValueError(f"Input not a continuous observed sequence: {case['pdb']} {chain}/{auth_chain}")
    a3m = ROOT / f"inputs/panel/{case['case_id']}.a3m"
    msa_info = json.loads(a3m.with_suffix(".manifest.json").read_text())
    if sha256(a3m) != msa_info["a3m_sha256"]:
        raise ValueError("MSA checksum mismatch")
    with a3m.open() as handle:
        header, names, query = [next(handle).strip() for _ in range(3)]
    lengths = [len(s) for s in case["chain_sequences"].values()]
    if header != "#" + ",".join(map(str, lengths)) + "\t1,1,1,1" or query != "".join(case["chain_sequences"].values()):
        raise ValueError("MSA header/query mismatch")
    return a3m, reference


def completed_files(folder, job_id):
    models = list(folder.glob(f"{job_id}_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_042.pdb"))
    scores = list(folder.glob(f"{job_id}_scores_rank_001_alphafold2_multimer_v3_model_1_seed_042.json"))
    if len(models) != 1 or len(scores) != 1 or not (folder / f"{job_id}.done.txt").exists():
        raise ValueError(f"Incomplete prediction or unexpected model/seed: {job_id}")
    return models[0], scores[0]


def inference_log(folder, job_id):
    text = (folder / "log.txt").read_text()
    current = None
    result = None
    for line in text.splitlines():
        query = re.search(r"Query \d+/\d+: (\S+) \(length", line)
        if query:
            current = query.group(1)
        timing = re.search(r"took ([\d.]+)s \((\d+) recycles\)", line)
        if timing and current == job_id:
            result = {"seconds": float(timing.group(1)), "recycles": int(timing.group(2))}
    if result is None:
        raise ValueError(f"No actual inference/recycle log for {job_id}")
    return result


def predict(args):
    cases = inputs()
    raw = ROOT / "results/af2_raw" / args.run_name
    output = args.result_dir.resolve()
    if list(raw.glob("*.done.txt")) or list(raw.glob("*_scores_rank_001_*.json")):
        raise FileExistsError("Completed outputs exist; analyze instead. Original timing records are not overwritten.")
    source_dir = ROOT / "cache" / args.run_name
    source_dir.mkdir(parents=True, exist_ok=True)
    if any(p.suffix != ".a3m" or p.stem not in {c["case_id"] for c in cases} for p in source_dir.iterdir()):
        raise ValueError("Unexpected file in A3M-only batch directory")
    hashes = {}
    for case in cases:
        a3m, _ = validate_case(case)
        shutil.copy2(a3m, source_dir / a3m.name)
        hashes[case["case_id"]] = sha256(a3m)
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({"LD_LIBRARY_PATH": str(args.env / "lib") + ":" + environment.get("LD_LIBRARY_PATH", ""),
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.35",
                        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "PYTHONDONTWRITEBYTECODE": "1"})
    command = [str(args.env / "bin/python"), "-m", "colabfold.batch", str(source_dir), str(raw),
               "--data", str(args.data), "--model-type", "alphafold2_multimer_v3", "--model-order", "1",
               "--num-models", "1", "--num-seeds", "1", "--random-seed", "42", "--num-recycle", "3",
               "--msa-mode", "mmseqs2_uniref_env", "--pair-mode", "unpaired", "--max-msa", "64:128",
               "--num-relax", "0", "--recompile-padding", "10", "--disable-unified-memory"]
    started = time.monotonic()
    record_path = output / "run.json"
    if record_path.exists():
        raise FileExistsError("Run metadata already exists; choose a separate checkout rather than overwrite it")
    record = {"status": "running", "command": command, "input_sha256": hashes,
              "case_ids": [c["case_id"] for c in cases], "templates": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "three fixed geometry references, not binding classification"}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        with (raw / "launcher.log").open("a") as handle:
            subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
        config = json.loads((raw / "config.json").read_text())
        if config["num_queries"] != 3 or config["max_seq"] != 64 or config["max_extra_seq"] != 128:
            raise ValueError("Effective ColabFold config does not match the prespecified panel")
        for case in cases:
            completed_files(raw, case["case_id"])
        record.update(status="completed", elapsed_seconds=time.monotonic() - started,
                      effective_config=config,
                      model_weight_sha256=sha256(args.data / "params/params_model_1_multimer_v3.npz"))
    except Exception as error:
        record.update(status="failed", error=str(error), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"]}))


def analyze(args):
    import numpy as np
    from geometry import analyze_geometry

    raw = ROOT / "results/af2_raw" / args.run_name
    output = args.result_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((output / "run.json").read_text())
    if run["status"] != "completed":
        raise ValueError("Analyze requires a successfully completed panel run")
    for case in inputs():
        a3m, _ = validate_case(case)
        if run["input_sha256"][case["case_id"]] != sha256(a3m):
            raise ValueError("Current MSA differs from the input actually used for inference")
    records = []
    specifications = [(c, raw, c["case_id"]) for c in inputs()]
    old = json.loads((ROOT / "inputs/6kzw_domain_complex.json").read_text())
    old.update(case_id="6zkw", mhc_allele="HLA-E", release_date="2022-01-26",
               reference_chain_mapping={"A": "D", "B": "E", "C": "C", "D": "A"})
    specifications += [(old, ROOT / "results/af2_raw/msa_unpaired", "6zkw_domain_msa")]
    for case, folder, job_id in specifications:
        model, score_path = completed_files(folder, job_id)
        if case["case_id"] == "6zkw":
            reference = ROOT / "cache/tcrmodel2/6ZKW.pdb"
            if sha256(reference) != case["reference_sha256"]:
                raise ValueError("Previous 6ZKW reference checksum mismatch")
            previous = json.loads((ROOT / "results/af2_msa_run.json").read_text())
            if previous["status"] != "completed" or previous["protocol"]["chain_sequences"] != case["chain_sequences"]:
                raise ValueError("Previous 6ZKW run/input mismatch")
            if previous["effective_config"]["max_seq"] != 64 or previous["effective_config"]["max_extra_seq"] != 128:
                raise ValueError("Previous 6ZKW MSA depth mismatch")
            if previous["model_weight_sha256"] != run["model_weight_sha256"]:
                raise ValueError("Reference panel model weight differs from previous pilot")
        else:
            _, reference = validate_case(case)
        geometry = analyze_geometry(model, reference, case["chain_sequences"], case["reference_chain_mapping"], min_coverage=0.90)
        score = json.loads(score_path.read_text())
        lengths = [len(s) for s in case["chain_sequences"].values()]
        bounds = np.cumsum([0] + lengths)
        pae = np.asarray(score["pae"])
        if pae.shape != (sum(lengths), sum(lengths)):
            raise ValueError("PAE shape mismatch")
        if len(score["plddt"]) != sum(lengths) or not np.isfinite(pae).all() or not np.isfinite(score["plddt"]).all():
            raise ValueError("Non-finite or incomplete confidence output")
        if not np.isfinite([score["ptm"], score["iptm"]]).all():
            raise ValueError("Non-finite global confidence output")
        p_indices, t_indices = np.arange(bounds[2], bounds[3]), np.arange(bounds[2])
        p_t_pae = float(np.mean(np.concatenate([
            pae[np.ix_(p_indices, t_indices)].ravel(), pae[np.ix_(t_indices, p_indices)].ravel()])))
        record = {"pdb": case["pdb"], "case_id": case["case_id"], "mhc_allele": case["mhc_allele"],
                  "release_date": case["release_date"], "mean_plddt": float(np.mean(score["plddt"])),
                  "ptm": float(score["ptm"]), "iptm": float(score["iptm"]),
                  "peptide_tcr_full_variable_domain_bidirectional_pae": p_t_pae,
                  "geometry": geometry, "reference_sha256": sha256(reference),
                  "prediction_sha256": sha256(model), "training_overlap": case["training_overlap"],
                  "inference_log": inference_log(folder, job_id), "num_recycles_max_requested": 3}
        shutil.copy2(model, output / f"{case['case_id']}_prediction.pdb")
        records.append(record)
    payload = {"scope": "fixed geometry pilot; three new predictions plus previous 6ZKW; all have experimental complex structures, no labeled nonbinding controls",
               "protocol": "AF2 multimer_v3 model1 seed42 at most3recycles unpairedMSA64:128 no templates",
               "cases": records}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# 多参考结构验证：实际结果", "",
             "2026-10-03。三例在推理前固定；6ZKW 复用上一轮 model1 / seed42 / MSA64:128 / 无模板结果，没有重新预测。全部为已有实验复合物结构的配对，不是结合分类或独立 OOD 测试。", "",
             "| PDB | HLA | pLDDT | ipTM | 肽↔TCR PAE / Å | MHC 对齐后 α / β RMSD / Å | α / β 自对齐 RMSD / Å | 肽 RMSD / Å |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        g = r["geometry"]
        m, s = g["mhc_aligned_ca_rmsd"], g["per_chain_self_fit_ca_rmsd"]
        lines.append(f"| [{r['pdb']}](https://www.rcsb.org/structure/{r['pdb']}) | {r['mhc_allele']} | {r['mean_plddt']:.2f} | {r['iptm']:.2f} | {r['peptide_tcr_full_variable_domain_bidirectional_pae']:.2f} | {m['A']:.2f} / {m['B']:.2f} | {s['A']:.2f} / {s['B']:.2f} | {m['C']:.2f} |")
    lines += ["", "## 肽–TCR 接触恢复", "",
              "任意跨残基非氢原子距离 <5 Å 定义一个残基对接触，仅比较序列对应且有坐标的残基。", "",
              "| PDB | 参考接触数 | 预测接触数 | 正确接触数 | Precision | Fnat（Recall） | F1 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        c = r["geometry"]["pep_tcr_contacts"]["combined"]
        def f(value):
            return "" if value is None else f"{value:.3f}"
        lines.append(f"| {r['pdb']} | {c['native']} | {c['pred']} | {c['tp']} | {f(c['precision'])} | {f(c['recall'])} | {f(c['f1'])} |")
    actual_recycles = "、".join(f"{r['pdb']}={r['inference_log']['recycles']}" for r in records)
    lines += ["", f"请求最多 3 recycles，保留 ColabFold 默认提前收敛规则；日志中的实际次数：{actual_recycles}。没有依据参考 RMSD 或识别标签选择中间 recycle。", "",
              "高单体置信度不等于正确对接，更不等于结合概率。全部采用官方连续结构域裁剪，未输入 β2m、MHCα3 或 TCR constant 域；1OGA/6UON 裁剪还省略少量 N 端残基，没有补推 α 链或拼接缺失环。", "",
              "原始 PDB 的训练重叠与近邻结构影响未排除；无模板只表示本次推理未输入结构模板，不能声称模型从未见过这些结构。PAE 是肽与完整 TCR 可变域的双向均值，不是 AF3-PTI-PAE 的同协议指标。", "",
              "接触恢复及 native 界面指标使用参考坐标作事后几何评测，不能在未知结合推理时使用。本目录的 interface_CA_RMSD_10A 是自定义 Cα 指标，不是标准 DockQ。", "",
              "本轮说明单链折叠与复合物对接必须分开评估：6UON 的 ipTM 最高，但 β 链在 MHC 坐标系下偏差最大，肽–TCR 接触恢复并非最好。四例不足以估计结合识别准确率；下一阶段需专门化对接或结构集合，并在有实测阴性的配对 αβ 测试集上验证。", "",
              "前三例序列来自 [TCRdock 锁定清单](https://github.com/phbradley/TCRdock/blob/c5a7af42eeb0c2a4492a4d4fe803f1f9aafb6193/tcrdock/db/ternary_templates_v2.tsv)。原论文：[1OGA](https://doi.org/10.1038/ni942)、[5WKH](https://doi.org/10.1038/ni.3850)、[6UON](https://doi.org/10.1073/pnas.1921964117)。本轮仅使用其中输入序列，不复现 TCRdock 算法。", "",
              f"[全部指标及覆盖信息](metrics.json) · [实际运行配置](run.json) · [输入与原始论文来源]({os.path.relpath(ROOT / 'inputs/panel/panel.json', output).replace(os.sep, '/')})"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ["pdb", "mean_plddt", "iptm"]} for r in records], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["predict", "analyze"])
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-name", default="reference_panel_v1")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results/reference_panel")
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Run name must be an alphanumeric folder name")
    {"predict": predict, "analyze": analyze}[args.action](args)
