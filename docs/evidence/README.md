# 验证证据

2026-10-01-env-probe.json来自用户提供的AutoDL 5090实例，原始文件通过SSH文件传输取回。运行工具版本：ffb651e65ab9e21a3936a94444da9a03067e4875。

远程日志位于/root/autodl-tmp/cunsl-2026/track6/local/runs/20261001T055126691304Z-env-smoke/。manifest.json记录进程成功退出，metrics_validated为false；该检查不涉及比赛数据或模型效果。

本地基础工具检查通过：6个Python语法检查、2个TOML配置、12个文档本地链接、8个正常/不安全路径/损坏ZIP案例和4个运行记录成功/失败案例（两仓库合计）。

本题接手后的证据：

- [2026-10-01-archive.json](2026-10-01-archive.json)：正式包SHA-256、全包CRC和结构汇总；两端传输摘要一致。
- [2026-10-01-data-audit.json](2026-10-01-data-audit.json)：9,489张TIFF全量解码、真实元数据字段与匿名统计，没有逐图路径或身份名单。
- [2026-10-01-split.json](2026-10-01-split.json)：固定划分摘要、身份/样本数、异常排除和三个方向规模；完整身份名单仅在local/splits及算力机。

新增18项数据/指标合成回归测试已在Windows通过，3项Linux共享锁测试待算力机执行；模型报告待运行后取回。AP@10正式程序尚未取得，指标核对只证明本地实现符合明确写出的口径。
