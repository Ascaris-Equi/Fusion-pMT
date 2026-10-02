# MSA 深度截断诊断

此目录保留修正前的实际运行记录，不纳入有效的 MSA 对照。

输入 A3M 已解析，但错误的 `--msa-mode single_sequence` 使 ColabFold 1.5.5 将有效 `max_seq` 和 `max_extra_seq` 强制改为 1。命令中的 `64:128` 未实际生效，详见 `truncated_msa_config.json`。这里的 ipTM 0.08 / 平均 pLDDT 31.93 不能用于判断充分使用 MSA 后的效果。

修正后使用非 single-sequence 模式及既有 A3M，并检查有效配置。最终对照放在上一级 `AF2_MSA_PILOT_RESULTS.md`。
