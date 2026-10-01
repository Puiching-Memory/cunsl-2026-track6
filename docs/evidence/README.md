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

识别效果升级轮次新增证据：

- [2026-10-01-dinov3-resources.json](2026-10-01-dinov3-resources.json)：公开维护者的两项DINOv3具体版本、发布时间、完整摘要、许可及下载/传输验证。
- [2026-10-01-modern-models.json](2026-10-01-modern-models.json)：两组划分的20份报告，含18个实验设置和2次恢复推理，完整三方向指标与训练用时/显存。
- [2026-10-01-modern-verification.json](2026-10-01-modern-verification.json)：所有20份保存排名独立复算、预测校验、两组微调检查点摘要和排名/预测逐字节恢复。
- [2026-10-01-modern-runs.json](2026-10-01-modern-runs.json)：16次持GPU锁的成功模型运行，以及保留的算力机网络下载失败记录。
- [2026-10-01-split2.json](2026-10-01-split2.json)：seed=20261002的身份隔离划分汇总，完整名单仅留在local/splits。

新代码4c2394b在算力机27项检查通过，Windows 21项通过、6项按Linux平台或无PyTorch跳过。两组B/16末4块监督微调加权分0.33659/0.31150，仍是本地验证结果，没有排行榜成绩。源代码版本与清单的历史摘要保留，后续说明更新不改写原始报告；各项身份、预测和权重均不入Git。

领域研究与消融轮次新增证据：

- [2026-10-01-domain-sources.json](2026-10-01-domain-sources.json)：3份完整论文摘要、3个截止日前固定代码快照与141个文件指纹、许可边界及待审查候选；未导入外部数据/权重。
- [2026-10-01-retrieval-errors.json](2026-10-01-retrieval-errors.json)：两组旧微调验证集完整Gallery错配分析、像素占比分组及训练身份样本数；仅匿名观察，未使用测试身份。
- [2026-10-01-domain-ablations.json](2026-10-01-domain-ablations.json)：7次训练与6次恢复的完整报告、历史第二组固定入口控制、三方向改变量及预算边界。
- [2026-10-01-domain-verification.json](2026-10-01-domain-verification.json)：13份独立指标复算和10候选校验、6次检查点恢复、初始化前向一致性及历史控制逐字节对照。
- [2026-10-01-domain-runs.json](2026-10-01-domain-runs.json)：共同GPU锁下15次调用，含保留的NHWC广播失败及修复后的14次成功。

实验代码ec180f5在算力机29项检查通过，Windows 21项通过、8项按平台或无Torch跳过。独立光学/SAR入口加权分0.35548/0.32832，结构损失效果不一致；没有排行榜成绩。文献原件、逐图诊断、身份、排名、预测、特征和检查点全部留在Git外。
