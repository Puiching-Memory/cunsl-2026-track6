# 验证证据

2026-10-01-env-probe.json来自用户提供的AutoDL 5090实例，原始文件通过SSH文件传输取回。运行工具版本：ffb651e65ab9e21a3936a94444da9a03067e4875。

远程日志位于/root/autodl-tmp/cunsl-2026/track6/local/runs/20261001T055126691304Z-env-smoke/。manifest.json记录进程成功退出，metrics_validated为false；该检查不涉及比赛数据或模型效果。

本地基础工具检查通过：6个Python语法检查、2个TOML配置、12个文档本地链接、8个正常/不安全路径/损坏ZIP案例和4个运行记录成功/失败案例（两仓库合计）。

本题接手后的证据：

- [2026-10-01-archive.json](2026-10-01-archive.json)：正式包SHA-256、全包CRC和结构汇总；两端传输摘要一致。
- [2026-10-01-data-audit.json](2026-10-01-data-audit.json)：9,489张TIFF全量解码、真实元数据字段与匿名统计，没有逐图路径或身份名单。
- [2026-10-01-split.json](2026-10-01-split.json)：固定划分摘要、身份/样本数、异常排除和三个方向规模；完整身份名单仅在local/splits及算力机。
- [2026-10-01-data-audit-remote.json](2026-10-01-data-audit-remote.json)：算力机加锁修复后的全量通过摘要，统计与Windows一致。
- [2026-10-01-baselines.json](2026-10-01-baselines.json)：三组干净版本的完整报告，包括三方向指标、种子、摘要、用时、显存与训练配置。
- [2026-10-01-metric-verification.json](2026-10-01-metric-verification.json)：保存排序的独立逐位AP复算、预测校验和检查点重放摘要；不含排序ID。
- [2026-10-01-runs.json](2026-10-01-runs.json)：命令、版本、工作区状态、起止时间、退出码、GPU锁与日志位置；包括失败的并发审查。
- [2026-10-01-weight-source.json](2026-10-01-weight-source.json)：原始ViT-S/14权重的公开来源、下载时间、字节数与SHA-256。下载前资源清单尚未填入权重摘要，其清单摘要与下载后版本不同。
- [2026-10-01-validation-checks.json](2026-10-01-validation-checks.json)：两端检查数、全量审查、跨平台划分复现和指标核对汇总。

新增23项回归检查在算力机全部通过；Windows通过20项，3项Linux共享锁检查按平台跳过。三个基准报告已取回，独立标量复算误差≤1e-12，适配器检查点推理输出逐字节复现。AP@10正式程序尚未取得，指标核对只证明本地实现符合明确写出的口径，不能称为官方评分一致性。
