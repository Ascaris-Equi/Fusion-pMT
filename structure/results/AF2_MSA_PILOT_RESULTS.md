# 本机新结构预测：AF2-Multimer 小试跑

PDB 6ZKW 的公开域序列；服务器 unpaired MSA、无模板。训练/近邻结构重叠未系统排除，不能当独立 OOD 验证。ipTM、pLDDT 和 PAE 不是结合概率。

| 指标 | 本机实际结果 |
| --- | --- |
| pTM（score JSON，保留两位） | 0.68 |
| ipTM（score JSON，保留两位） | 0.58 |
| 平均 pLDDT | 88.46 |
| 肽↔TCR 平均双向 PAE / Å | 14.31 |
| MHC 对齐后 tcra Cα RMSD / Å | 6.95 |
| MHC 对齐后 tcrb Cα RMSD / Å | 5.45 |
| MHC 对齐后 pep Cα RMSD / Å | 0.57 |
| MHC 对齐后 mhca Cα RMSD / Å | 0.78 |

输入域序列来自 TCRmodel2 官方示例，但推理程序是 ColabFold AF2-Multimer，并非复现 TCRmodel2 的专门化算法。上游 job_id 的 6kzw 是字母转置，真实参考为 [6ZKW](https://www.rcsb.org/structure/6ZKW)；原始任务文件名保留上游别名。参考 PDB 只在预测之后用于坐标比较，不输入为模板。

[预测结构](af2_msa_6zkw_prediction.pdb) · [坐标与界面指标](af2_msa_structure_metrics.json) · [实际命令和耗时](af2_msa_run.json)
