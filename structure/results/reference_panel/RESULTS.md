# 多参考结构验证：实际结果

2026-10-03。三例在推理前固定；6ZKW 复用上一轮 model1 / seed42 / MSA64:128 / 无模板结果，没有重新预测。全部为已有实验复合物结构的配对，不是结合分类或独立 OOD 测试。

| PDB | HLA | pLDDT | ipTM | 肽↔TCR PAE / Å | MHC 对齐后 α / β RMSD / Å | α / β 自对齐 RMSD / Å | 肽 RMSD / Å |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1OGA](https://www.rcsb.org/structure/1OGA) | HLA-A*02:01 | 88.54 | 0.63 | 11.88 | 6.69 / 6.55 | 0.68 / 1.50 | 0.61 |
| [5WKH](https://www.rcsb.org/structure/5WKH) | HLA-A*11:01 | 84.53 | 0.43 | 19.58 | 5.99 / 4.82 | 0.99 / 0.87 | 2.44 |
| [6UON](https://www.rcsb.org/structure/6UON) | HLA-C*08:02 | 92.22 | 0.81 | 9.81 | 5.84 / 12.92 | 1.26 / 0.80 | 1.73 |
| [6ZKW](https://www.rcsb.org/structure/6ZKW) | HLA-E | 88.46 | 0.58 | 14.31 | 6.95 / 5.45 | 0.87 / 0.91 | 0.57 |

## 肽–TCR 接触恢复

任意跨残基非氢原子距离 <5 Å 定义一个残基对接触，仅比较序列对应且有坐标的残基。

| PDB | 参考接触数 | 预测接触数 | 正确接触数 | Precision | Fnat（Recall） | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1OGA | 19 | 16 | 8 | 0.500 | 0.421 | 0.457 |
| 5WKH | 22 | 18 | 9 | 0.500 | 0.409 | 0.450 |
| 6UON | 11 | 19 | 5 | 0.263 | 0.455 | 0.333 |
| 6ZKW | 16 | 26 | 11 | 0.423 | 0.688 | 0.524 |

请求最多 3 recycles，保留 ColabFold 默认提前收敛规则；日志中的实际次数：1OGA=2、5WKH=3、6UON=3、6ZKW=3。没有依据参考 RMSD 或识别标签选择中间 recycle。

高单体置信度不等于正确对接，更不等于结合概率。全部采用官方连续结构域裁剪，未输入 β2m、MHCα3 或 TCR constant 域；1OGA/6UON 裁剪还省略少量 N 端残基，没有补推 α 链或拼接缺失环。

原始 PDB 的训练重叠与近邻结构影响未排除；无模板只表示本次推理未输入结构模板，不能声称模型从未见过这些结构。PAE 是肽与完整 TCR 可变域的双向均值，不是 AF3-PTI-PAE 的同协议指标。

接触恢复及 native 界面指标使用参考坐标作事后几何评测，不能在未知结合推理时使用。本目录的 interface_CA_RMSD_10A 是自定义 Cα 指标，不是标准 DockQ。

本轮说明单链折叠与复合物对接必须分开评估：6UON 的 ipTM 最高，但 β 链在 MHC 坐标系下偏差最大，肽–TCR 接触恢复并非最好。四例不足以估计结合识别准确率；下一阶段需专门化对接或结构集合，并在有实测阴性的配对 αβ 测试集上验证。

前三例序列来自 [TCRdock 锁定清单](https://github.com/phbradley/TCRdock/blob/c5a7af42eeb0c2a4492a4d4fe803f1f9aafb6193/tcrdock/db/ternary_templates_v2.tsv)。原论文：[1OGA](https://doi.org/10.1038/ni942)、[5WKH](https://doi.org/10.1038/ni.3850)、[6UON](https://doi.org/10.1073/pnas.1921964117)。本轮仅使用其中输入序列，不复现 TCRdock 算法。

[全部指标及覆盖信息](metrics.json) · [实际运行配置](run.json) · [输入与原始论文来源](../../inputs/panel/panel.json)
