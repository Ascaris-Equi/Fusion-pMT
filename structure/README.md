# 利用结构预测 TCR–pMHC 结合：可行性调研

更新：2026-10-02。范围：`Ascaris-Equi/Fusion-pMT` 的新研究方向；本目录不改已有序列模型。以下为文献与代码调研，不是本地实验结果。

## 1. 结论

值得尝试，但目标应是**结构辅助的结合候选排序/分类**，不能把“预测出了复合物结构”直接解释成“发生结合”。优先考虑 **AlphaFold 3（AF3）生成结构集合，再用界面特征和集合统计评分**；已有 enFoldX 代码及分类器可作为 MHCI 任务的起点，其当前实现不能直接推广到 MHCII。[enFoldX 官方代码](https://github.com/jonlevi/enFoldX)

需要区分三件事：结构姿态是否正确、TCR 是否识别该 pMHC、实验测得的亲和力或细胞激活。前者的置信度不是后两者的直接测量。2026 年不同独立评测的判别性能差异很大，不能将一个数据集的 AUROC 当成我们数据上的预期准确率。[AF3 对比评测代码](https://github.com/benjiemc/tcr-pmhc-predicted-structures)

当前最先要解决的是输入：本仓库样例只有 CDR3β-like 短序列和 34 aa MHC 伪序列，不能据此构造唯一、真实的 αβ TCR–pMHC 复合物。

## 2. 本仓库能直接提供什么

审计基准：main 的 `a7ab444c49d259d8e342efdfffafc76054578825`，仅针对该版本的样例和推理接口，不代表此前所有训练/OOD 数据。

| 项目 | 当前代码/样例 | 结构建模还需要什么 |
| --- | --- | --- |
| 肽 | `input.csv` 的 4 条样例均为 9 aa 肽 | 可直接作为肽链；保留来源及实验标签 |
| TCR | `TCR_sequence` 为 13 或 15 aa、CDR3β-like 序列；代码配置 `tcr_max_len=26` | 真实配对的 TCRα/β 可变域，或双链可靠的 V/J/CDR3 信息以重建 |
| MHC | `MHC_sequence` 为 34 aa 伪序列；样例在本仓库映射表中仅匹配 `HLA-A*02:01` | 明确 HLA 等位基因及连续的结构域序列；按工具协议决定是否加入 β2m |
| 监督信息 | 样例三列没有标签、αβ 配对或实验元数据 | 结合/非结合或功能实验标签、测定类型、数据来源、划分标识 |

依据：[input.csv](../input.csv)、[infer.py](../infer.py)、[common_hla_sequence.csv](../common_hla_sequence.csv)。伪序列是从不同位置选出的残基，不是可折叠的连续 MHC 链；不能由 β 链 CDR3 唯一推出 α 链。

完整输入不一定指全长膜蛋白。TCRmodel2 使用可变域及 MHC 结合结构域；enFoldX 的预训练分类器则依赖其特定的完整序列重建与裁剪流程，不能随意换成另一套域定义后仍声称同协议复现。[TCRmodel2 输入说明](https://github.com/piercelab/tcrmodel2#generate-tcr-pmhc-complex-predictions)、[enFoldX 输入与预处理说明](https://github.com/jonlevi/enFoldX#step-1-prepare-sequence-inputs)

若原始数据只能提供 β 链，可研究“假设 α 链条件下”的敏感性分析，但结果只能视为条件性模型评分，不能当作该 TCR 的真实结合预测。

## 3. 直接相关的研究证据

下列数值属于各自原文或官方代码的测试，数据、阴性定义和指标不同，不作横向排行榜。

| 研究 | 直接证据 | 对本项目的意义与限制 |
| --- | --- | --- |
| [TCRdock，eLife 2023](https://elifesciences.org/articles/82813) | 在有限候选肽集合中，用结构建模与界面置信度识别认知肽 | 支持结构辅助排序；不是对任意未知肽普遍有效的证明 |
| [AF3-PTI-PAE，2026-06 预印本](https://pubmed.ncbi.nlm.nih.gov/42282661/) | 用肽–TCR 界面 PAE；抗原中心测试的中位 ROC-AUC：MHCI 0.92、MHCII 0.81 | 有积极信号，但独立 MHCI 验证仅 16 个认知对与 181 个置换对；置换阴性并非逐对实测非结合 |
| [AF3 特异性对比评测，2026-07 预印本及代码](https://github.com/benjiemc/tcr-pmhc-predicted-structures) | 官方 notebook 的 1,500 对评测：AF3 CDR:pMHC PAE ROC-AUC 约 0.601，标准化 pAUC(FPR≤0.1) 约 0.502 | 提醒我们高结构质量不保证低假阳性区间的判别能力；阴性来自打乱配对。[固定版本 notebook](https://github.com/benjiemc/tcr-pmhc-predicted-structures/blob/b1e969a460e1ceab8dd95a081e42241819843187/notebooks/benchmark_structure_prediction_methods.ipynb) |
| [enFoldX，2026-07-20 预印本](https://pmc.ncbi.nlm.nih.gov/articles/PMC13419715/) | 汇总多次 AF3 预测的特征分布；二月训练、九月新增数据上的新 TCR / 新表位 ROC-AUC 约 0.86 / 0.77 | 可实施的“结构集合→结合评分”路线；该测试仍用 decoy/置换阴性，不是逐对实测非结合，也不是严格远同源隔离测试 |
| [Culka 等，Cell Systems 2026](https://doi.org/10.1016/j.cels.2026.101700) | 比较序列学习与结构/物理模型；未见表位及肿瘤新抗原泛化仍有明显困难 | 可加入界面能量等特征，但不能假定物理评分天然解决 OOD，也不能将其直接当 Kd |

更严格的外部参照是 [IMMREP25，2026 年预印本](https://www.biorxiv.org/content/10.64898/2026.03.30.715276v1)：1,000 个 TCR、20 个此前未见 pMHC，最好方法的宏平均标准化 pAUC(FPR≤0.1) 约 0.60，随机基线为 0.50。这是部分 AUC，不是普通 ROC-AUC，也不是 60% 准确率；其功能性识别终点也不同于亲和力测定。[官方评测规则](https://www.kaggle.com/competitions/immrep25)

## 4. 可用代码与优先级

| 工具/路线 | 官方地址 | 在本项目中的角色 | 主要限制 |
| --- | --- | --- | --- |
| AF3 + enFoldX | [AF3](https://github.com/google-deepmind/alphafold3)、[enFoldX](https://github.com/jonlevi/enFoldX) | 优先 MHCI 路线：结构集合及现成特异性分类器 | enFoldX 现成模型对应本地 AF3、固定预处理；不能直接混用服务器输出或不同集合大小 |
| AF3-PTI-PAE | [原始实验代码](https://github.com/AltinLab/tcrtrifold-experiments) | 肽–TCR 界面 PAE 的单一评分基线 | 是特征评分而非发布了权重的二分类器；原工作流含实验室/集群依赖，需要适配 |
| TCRmodel2 | [代码](https://github.com/piercelab/tcrmodel2)、[服务器](https://tcrmodel.ibbr.umd.edu/) | 专门化复合物几何基线 | 需要 αβ 输入；生成高质量姿态本身不是结合分类器 |
| TCRdock | [代码](https://github.com/phbradley/TCRdock) | TCR 专门化 AF2、模板排除与几何分析 | 官方输入包括双链 V/J/CDR3；近邻模板必须控制 |
| Boltz-2 | [代码与输入说明](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md) | 通用复合物结构预测对照 | 其 affinity 模块仅支持小分子–蛋白，不支持把 TCR–pMHC 当该模块的亲和力任务 |
| Chai-1 | [代码](https://github.com/chaidiscovery/chai-lab) | 另一种结构预测对照 | 仍需独立的 TCR 特异性评分与验证，不能仅比较全局置信度 |
| NetTCR-struc | [代码](https://github.com/mnielLab/NetTCR-struc)、[论文](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1616328/full) | 几何 GNN/ESM-IF1 特征与姿态重评分 | 姿态重评分不等同于跨表位结合分类；旧 PyTorch/CUDA/PyG 依赖需要单独兼容性验证 |

enFoldX 官方提供 `human_1seed`（1 seed、5 个 AF3 样本）与 10-seed 人类模型（50 个样本）；使用对应分类器，不能用 5 个样本的统计特征冒充 50 个样本协议。模型随附 scaler，应保持特征及序列处理一致。[模型使用说明](https://github.com/jonlevi/enFoldX#step-3-predict-binding)

Boltz 的 `affinity_pred_value` / `affinity_probability_binary` 不应用作 TCR–pMHC 结合输出。[官方 affinity 适用范围](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md#binding-affinity-prediction)

部署时分别核对代码和权重的使用条款。AF3 与 enFoldX 有各自条款；NetTCR-struc 官方限定学术非商业用途。本目录只放调研，不重新分发第三方模型或数据。[AF3](https://github.com/google-deepmind/alphafold3)、[enFoldX 条款](https://github.com/jonlevi/enFoldX/blob/main/TERMS_OF_USE.md)、[NetTCR-struc 许可](https://github.com/mnielLab/NetTCR-struc#license)

## 5. 本机资源可行性

按现有 RTX 5090 32 GB 工作站规划，首先做小规模兼容性试跑；尚未验证这台机器的 AF3/JAX 或上述工具能否正常运行。AF3 官方充分测试的是 A100/H100 80 GB，不能直接将其速度表换算成本机吞吐。[AF3 硬件与性能说明](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)

资源不仅是显存：enFoldX 官方给完整 AF3 数据库预留约 1 TB；TCRmodel2 的 `pdb_mmcif + params` 解压后约 120 GB。磁盘、CPU/RAM、驱动及框架兼容性应在下载之前核对。[enFoldX requirements](https://github.com/jonlevi/enFoldX#requirements)、[TCRmodel2 数据库说明](https://github.com/piercelab/tcrmodel2#download-database)

可缓存并复用每条链的 MSA/模板结果，分开 CPU 搜索与 GPU 推理，避免反复搜索同一 HLA/TCR。完全无 MSA/无模板属于另一种协议，不能默认为原文复现。[AF3 分阶段运行](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#running-the-pipeline-in-stages)

GPU 先串行测试；总生成量按 `候选对数 × seeds × 每 seed 样本数` 计算。先测本机实际耗时、峰值显存、输出体积及失败率，再决定规模。本次没有安装环境、下载模型/数据库或提交结构预测任务。

## 6. 建议怎样一步步验证

以下是拟议实验，不是已经完成的结果；旧 Fusion-pMT 复现实验保持停止。

| 阶段 | 工作 | 完成后应该看到什么 |
| --- | --- | --- |
| S0 输入审计 | 从原始来源筛选真正配对的 αβ TCR，保留双链 V/J/CDR3、HLA、肽、测定类型及来源；统计能完整还原的数量 | 可建模子集的清单及缺失比例，不凭空补 α 链 |
| S1 小规模结构试跑 | 用约 10–20 个有实验结构的复合物检查输入、链对应、运行与输出；用模板截止日期及相似性排除控制近邻信息，并区分模型结构训练截止日前/后的样本 | 本机资源记录；单体/界面几何指标；明确区分结构复现与结合识别 |
| S2 固定判别试验 | 建立约 100–200 对的探索集，包含认知对及同 HLA/相近肽长度的对照；实测非结合与合成错配分别标记 | AF3 界面分数、enFoldX 对应版本及序列基线在同一批样本上的逐对结果 |
| S3 严格泛化评估 | 扩大数据，按 TCR 克隆型/相似性、表位、来源或时间做预先固定的留出；分别报告已见及未见表位 | ROC-AUC、AUPRC、标准化 pAUC(FPR≤0.1)、Top-k 召回及不确定性；同时报告模型失败的样本 |
| S4 再考虑融合 | 只在结构路线有可复验增益后，对比序列、结构、序列+结构；保持样本、划分与调参预算一致 | 结构特征是否提供独立增益，而不只是增加参数或利用数据来源差异 |

评分的起点是肽–TCR/CDR 界面 PAE、接触、埋藏面积、碰撞及多结构间的统计变化，而非只有全局 ipTM。AF3 的 `ranking_score` 用于结构排序，不是结合概率。[AF3 输出定义](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md) 若使用 enFoldX 现成分类器，应先完整复现其特征流程；新加物理特征属于另一个模型，需要独立训练与验证。

负样本不能只用背景健康 TCR：这可能使模型学到来源/序列分布差异，而不是配对特异性。优先使用可核验的实验阴性；合成错配作为单独对照，并排除已知阳性。为避免污染留出，先划分正例与 TCR/表位组，再在各划分内部生成错配；仍不能把“未被记录为阳性”等同于真实不结合。[阴性与泛化问题的原始研究](https://doi.org/10.1016/j.cels.2026.101700)

测试标签不用于挑选 seed、结构姿态、阈值或超参数；姿态选择规则在评测前固定，仅依赖推理可用信息。多种子是结构采样，不是多个独立实验重复；置信区间应按独立 TCR/表位组重采样。S2 的小集只用于判断是否值得扩大，不能据此宣称普适泛化。

现成分类器的训练数据也要对账：enFoldX 当前人类模型说明列出 VDJdb 2025-02/2025-09 数据，因此不能直接拿这些数据随机抽样作独立评测，也不能用已包含九月数据的权重复现论文“二月训练→九月验证”的 0.86/0.77。应核对具体权重对应的训练版本及正/负配对，排除重叠，并按所声称的“新 TCR / 新表位”任务控制相似性。无法核验训练重叠的测试只能称流程演示，不能称独立泛化验证。[enFoldX 模型说明](https://github.com/jonlevi/enFoldX/blob/main/models/MODELS.md)

建议先做 S0，再做 S1 和 MHCI 的 `AF3 + enFoldX human_1seed` 小规模 S2；若输入或本机兼容性不满足，就先停在可建模数据审计，而不是将当前 β-only 样例强行转换成全复合物输入。
