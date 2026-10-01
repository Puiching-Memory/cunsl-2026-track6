# 验证证据

2026-10-01-env-probe.json来自用户提供的AutoDL 5090实例，原始文件通过SSH文件传输取回。运行工具版本：ffb651e65ab9e21a3936a94444da9a03067e4875。

远程日志位于/root/autodl-tmp/cunsl-2026/track6/local/runs/20261001T055126691304Z-env-smoke/。manifest.json记录进程成功退出，metrics_validated为false；该检查不涉及比赛数据或模型效果。

本地基础工具检查通过：6个Python语法检查、2个TOML配置、12个文档本地链接、8个正常/不安全路径/损坏ZIP案例和4个运行记录成功/失败案例（两仓库合计）。
