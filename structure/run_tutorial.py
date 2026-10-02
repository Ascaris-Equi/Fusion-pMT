"""Re-score published enFoldX features; this script does not generate structures."""

import argparse
import hashlib
import json
import pickle
import platform
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
COMMIT = "89c8a01f50a0e3daceef06fb63de9a5481328dde"
FILES = {
    "input": "examples/MEL_data/MEL_enfoldx_input.csv",
    "ensemble": "examples/enfoldx_extracted_features/ensemble_features.csv",
    "samples": "examples/enfoldx_extracted_features/all_structures_features.csv",
    "reference": "examples/enfoldx_predictions/enFoldX_predictions.csv",
    "human": "models/enFoldX_human_vFebSept_DecoyPerm.pkl",
    "human_decoy": "models/enFoldX_human_vFebSept_Decoy.pkl",
    "human_1seed": "models/enFoldX_human_vFebSept_DecoyPerm_1seed.pkl",
}
GIT_BLOBS = {
    "examples/MEL_data/MEL_enfoldx_input.csv": "4e5c78ec3e8370cb6baeab332b66c6f6f66a26f7",
    "examples/enfoldx_extracted_features/ensemble_features.csv": "f07b2fefeb590e1d8c97f3ea915ec09a35f17a8a",
    "examples/enfoldx_extracted_features/all_structures_features.csv": "7ec438fd1429383265108d3476d83c24854929f0",
    "examples/enfoldx_predictions/enFoldX_predictions.csv": "d6b1e6caeb3fe7e5102d91a91a41cef9a211b7d8",
    "models/enFoldX_human_vFebSept_DecoyPerm.pkl": "fc5bd3d2ec400efc93378211da5ae0914124ff56",
    "models/enFoldX_human_vFebSept_Decoy.pkl": "b671d1e4140acae025130e6837b4d2962623a40a",
    "models/enFoldX_human_vFebSept_DecoyPerm_1seed.pkl": "024dade0fb890813effddfd9273782b6635bbb2a",
}


def fetch(path, cache):
    url = f"https://raw.githubusercontent.com/jonlevi/enFoldX/{COMMIT}/{path}"
    target = cache / path
    if not target.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "structure-pilot"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(5_000_001)
        if len(content) > 5_000_000:
            raise ValueError(f"Unexpected source size: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    content = target.read_bytes()
    git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if git_hash != GIT_BLOBS[path]:
        raise ValueError(f"Source content does not match pinned Git blob: {path}")
    return target, {"url": url, "git_blob": git_hash, "sha256": hashlib.sha256(content).hexdigest()}


def read_features(path):
    data = pd.read_csv(path)
    return data.loc[:, ~data.columns.str.startswith("Unnamed:")]


def aggregate(samples):
    keys = ["complex_id", "af3_output_directory"]
    values = samples.groupby(keys, dropna=False).agg(["mean", "std"])
    values.columns = [f"{name}_std" if stat == "std" else name for name, stat in values.columns]
    values = values.drop(columns=[c for c in values if c.startswith(("af3_seed", "af3_sample", "af3_ranking"))])
    return values.reset_index()


def score(features, path):
    # Only the pinned official project model bundles downloaded above are loaded.
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if type(bundle["model"]).__name__ != "LogisticRegression":
        raise TypeError("Unexpected classifier type")
    names = bundle["features"]
    if len(names) != 212 or not set(names).issubset(features.columns):
        raise ValueError("Model feature schema mismatch")
    matrix = features[names]
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Non-finite model features")
    classes = list(bundle["model"].classes_)
    positive = classes.index(1)
    scaled = bundle["scaler"].transform(matrix)
    probabilities = bundle["model"].predict_proba(scaled)[:, positive]
    return pd.Series(probabilities, index=features["complex_id"])


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache" / "enfoldx"
    paths, sources = {}, {}
    for name, path in FILES.items():
        paths[name], sources[name] = fetch(path, cache)
    sequences = read_features(paths["input"])
    ensemble = read_features(paths["ensemble"])
    samples = read_features(paths["samples"])
    official = read_features(paths["reference"])
    if official.complex_id.duplicated().any() or set(official.complex_id) != set(sequences.complex_id):
        raise ValueError("Official reference has missing or duplicate pair IDs")
    labels = json.loads((ROOT / "pilot_labels.json").read_text(encoding="utf-8"))
    peptide_labels = labels["peptides"]
    if len(sequences) != 18 or len(ensemble) != 18 or len(samples) != 900:
        raise ValueError("Unexpected tutorial dimensions")
    if sequences.complex_id.duplicated().any() or set(sequences.complex_id) != set(ensemble.complex_id):
        raise ValueError("Missing or duplicate pair IDs")
    for row in sequences.itertuples(index=False):
        if row.TRA_CDR3 not in row.TRA_aa or row.TRB_CDR3 not in row.TRB_aa:
            raise ValueError(f"CDR3 not contained in supplied chain: {row.complex_id}")
    if set(sequences.peptide) != set(peptide_labels):
        raise ValueError("Tutorial labels do not exactly cover the nine peptides")
    if not samples.groupby("complex_id").size().eq(50).all():
        raise ValueError("Not fifty original structures per pair")
    if samples.duplicated(["complex_id", "af3_seed", "af3_sample"]).any():
        raise ValueError("Duplicate structure samples")
    original_aggregate = aggregate(samples).set_index("complex_id")
    numeric = [c for c in ensemble if c not in ["complex_id", "af3_output_directory"]]
    aggregation_error = float(np.max(np.abs(original_aggregate.loc[ensemble.complex_id, numeric].to_numpy() - ensemble[numeric].to_numpy())))
    if aggregation_error > 1e-10:
        raise ValueError(f"Cannot reproduce published aggregation: {aggregation_error}")
    first_seed = samples.loc[samples.af3_seed.eq(1)].copy()
    if not first_seed.groupby("complex_id").size().eq(5).all():
        raise ValueError("Not five samples at seed one")
    one_seed = aggregate(first_seed)
    result = sequences[["complex_id", "TRA_CDR3", "TRB_CDR3", "peptide"]].copy()
    result["peptide_name"] = result.peptide.map(lambda p: peptide_labels[p]["peptide_name"])
    result["recognition_label"] = result.peptide.map(lambda p: peptide_labels[p]["recognition_label"])
    result["TCR"] = result.complex_id.str.extract(r"(?i)(MEL[58])", expand=False).str.upper()
    if result.TCR.isna().any() or result.recognition_label.value_counts().to_dict() != {0: 12, 1: 6}:
        raise ValueError("Invalid TCR IDs or recognition labels")
    for name in ["human", "human_decoy", "human_1seed"]:
        frame = one_seed if name == "human_1seed" else ensemble
        result[name] = result.complex_id.map(score(frame, paths[name]))
    original = official.set_index("complex_id")["enFoldX_score"]
    mapped_reference = result.complex_id.map(original)
    if not np.isfinite(mapped_reference.to_numpy()).all():
        raise ValueError("Non-finite official reference predictions")
    reference_error = float((result.human - mapped_reference).abs().max())
    if reference_error > 1e-10:
        raise ValueError(f"Official prediction comparison failed: {reference_error}")
    metrics = {}
    for name in ["human", "human_decoy", "human_1seed"]:
        metrics[name] = {
            "roc_auc": float(roc_auc_score(result.recognition_label, result[name])),
            "average_precision": float(average_precision_score(result.recognition_label, result[name])),
            "standardized_pauc_fpr_0_1": float(roc_auc_score(result.recognition_label, result[name], max_fpr=0.1)),
            "per_tcr_auc": {tcr: float(roc_auc_score(part.recognition_label, part[name])) for tcr, part in result.groupby("TCR")},
        }
    payload = {
        "scope": "published tutorial features rescored locally; not newly predicted structures or independent validation",
        "upstream_commit": COMMIT,
        "n_pairs": 18, "n_functional_positive": 6, "n_functional_control": 12,
        "original_af3_structures": 900, "original_samples_per_pair": 50,
        "one_seed_samples_per_pair": 5, "aggregation_max_abs_error": aggregation_error,
        "official_prediction_max_abs_error": reference_error, "sources": sources,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "label_source": labels["source"], "metrics": metrics,
        "pairs": json.loads(result.to_json(orient="records", double_precision=15)),
    }
    (output / "tutorial_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# enFoldX 公开教程：本地评分复现", "", "18 对公开配对，标签为实验条件下识别/未响应。使用作者公开 AF3 特征，不是本机新生成结构，也不是独立 OOD 结果。", "", "| TCR | 肽 | 识别标签 | human（50样本） | human_decoy（50样本） | human_1seed（5样本） |", "| --- | --- | --- | --- | --- | --- |"]
    for row in result.sort_values(["TCR", "peptide_name"]).itertuples(index=False):
        lines.append(f"| {row.TCR} | {row.peptide_name} | {row.recognition_label} | {row.human:.6f} | {row.human_decoy:.6f} | {row.human_1seed:.6f} |")
    lines += ["", "| 模型 | ROC-AUC | Average precision | 标准化 pAUC(FPR≤0.1) |", "| --- | --- | --- | --- |"]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['roc_auc']:.6f} | {values['average_precision']:.6f} | {values['standardized_pauc_fpr_0_1']:.6f} |")
    lines += ["", f"与官方 human 逐对预测最大绝对差：`{reference_error:.3e}`。900 个结构特征重新聚合的最大绝对差：`{aggregation_error:.3e}`。", "", "这些小样本指标仅用于教程复现，不能当实际泛化性能。BST2-9 等对照标签不表示绝对零亲和力。", "", f"实验来源：[Dolton et al., Cell 2023]({labels['source']['url']})。输入、权重和特征的固定版本 URL 与 SHA256 见 [tutorial_results.json](tutorial_results.json)。"]
    (output / "TUTORIAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": 18, "reference_error": reference_error, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    main(parser.parse_args().output)
