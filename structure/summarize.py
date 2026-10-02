"""Build the compact pilot report from the saved machine-readable results."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main():
    tutorial = load("tutorial_results.json")
    single = load("af2_structure_metrics.json")
    msa = load("af2_msa_structure_metrics.json")
    run = load("af2_msa_run.json")
    assert run["status"] == "completed"
    assert run["effective_config"]["max_seq"] == 64
    assert run["effective_config"]["max_extra_seq"] == 128
    lines = [
        "# 结构路线首轮实际结果", "",
        "日期：2026-10-02。旧 Fusion-pMT 实验未修改。", "",
        "## 1. 公开教程评分复现", "",
        "18 对 MEL5/MEL8 配对，6 个功能性识别阳性与 12 个教程对照；重新聚合作者公开的 900 个 AF3 特征行，不是本机新生成 900 个结构。", "",
        "| 固定分类器 | ROC-AUC | Average precision | 标准化 pAUC(FPR≤0.1) |",
        "| --- | --- | --- | --- |",
    ]
    for model, scores in tutorial["metrics"].items():
        lines.append(f"| {model} | {scores['roc_auc']:.4f} | {scores['average_precision']:.4f} | {scores['standardized_pauc_fpr_0_1']:.4f} |")
    lines += [
        "",
        f"与作者逐对评分最大差值 `{tutorial['official_prediction_max_abs_error']:.2e}`，特征聚合最大差值 `{tutorial['aggregation_max_abs_error']:.2e}`。三个分类器均固定；1seed 使用预先固定的 seed=1 / 5 个样本，未按标签挑种子。小教程集不是独立泛化验证。",
        "",
        "[逐对评分表](TUTORIAL_RESULTS.md) · [来源、权重校验及全部数值](tutorial_results.json)",
        "",
        "## 2. 本机新结构预测", "",
        "本轮输入采用与 [PDB 6ZKW](https://www.rcsb.org/structure/6ZKW) 对应的四条裁剪域序列：TCRα / TCRβ / 肽 / MHC 结合域，长度 112 / 115 / 9 / 180；未输入 β2m（参考 PDB 本身含 β2m）。用已有 AF2-Multimer v3 model_1、seed=42、3 recycles、无模板；不是 TCRmodel2 专门化算法复现。", "",
        "| 指标 | 单序列 | unpaired MSA（有效上限 64:128） |",
        "| --- | --- | --- |",
    ]
    for label, key, digits in [
        ("平均 pLDDT", "mean_plddt", 2), ("pTM（JSON 两位精度）", "ptm", 2),
        ("ipTM（JSON 两位精度）", "iptm", 2),
        ("肽↔完整 TCR 可变域平均双向 PAE / Å", "peptide_tcr_mean_bidirectional_pae_angstrom", 2),
    ]:
        lines.append(f"| {label} | {single[key]:.{digits}f} | {msa[key]:.{digits}f} |")
    for role in ["tcra", "tcrb", "pep", "mhca"]:
        lines.append(f"| MHC 对齐后 {role} Cα RMSD / Å | {single['mhc_aligned_ca_rmsd_angstrom'][role]:.2f} | {msa['mhc_aligned_ca_rmsd_angstrom'][role]:.2f} |")
    lines += [
        f"| 总运行耗时 / s | {load('af2_run.json')['elapsed_seconds']:.1f} | {run['elapsed_seconds']:.1f} |",
        "",
        "MSA 输入及校验值已保存；上限 64:128 是模型采样限制，并非所有检索行同时参与推理。参考结构只用于事后比较，没有输入为模板。模型训练与近邻结构重叠未系统排除，因此不是独立 OOD 验证。",
        "",
        "MSA 提高结构置信度，并改善该参考的 pMHC 几何及 MHC 坐标系下的 TCR 偏差，但 TCR 对接仍有 5–7 Å 的整体偏差；当前结果支持继续做结构辅助评分，不支持宣称已获得可信结合分类。全局 ipTM、pLDDT 不是结合概率；这里的全可变域 PAE 也不是原文 AF3-PTI-PAE 的同协议指标。",
        "",
        "[MSA 结果与结构](AF2_MSA_PILOT_RESULTS.md) · [单序列对照](AF2_PILOT_RESULTS.md) · [运行依赖快照](af2_runtime.json) · [错误截断配置诊断](diagnostics/README.md)",
        "",
        "## 3. 当前边界", "",
        "已完成：公开评分复现、真实 GPU 结构试跑、MSA 输入审计及逐项几何比较。尚未完成：AF3 本地新结构生成、配对 αβ 的独立识别测试集、多参考结构验证、结构与序列模型融合。原仓库 β-only / 34 aa MHC 伪序列不能直接冒充本轮四链输入。",
    ]
    output = ROOT / "results/PILOT_SUMMARY.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
