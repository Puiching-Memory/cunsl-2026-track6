# 通用实验工具

工具包括正式数据读取与审查、身份隔离协议、显式AP@10指标、严格提交校验以及基准运行。inspect_archive不解压或执行包内程序；fetch_weight只下载逐项审查过的精确公开权重。

## 环境

Python 3.11及以上。inspect_archive.py和run_job.py只用标准库；env_probe.py需要已有PyTorch。实际5090环境已经有可用PyTorch，不在系统环境覆盖安装旧版本。

## 命令

```bash
python tools/inspect_archive.py data/source.zip --verify-crc --output local/archive-audit.json
python tools/run_job.py --name env-smoke -- python tools/env_probe.py --output local/env-probe.json
```

run_job.py在local/runs中建立独立运行目录，记录命令、代码提交、开始/结束时间、退出码和日志。它不记录环境变量或登录凭据。命令自身不得包含密码或令牌。

新增记录git_dirty_files；正式实验要求干净的已提交代码。GPU命令必须加--gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock；Linux flock取得后检查现有GPU进程，占用时拒绝启动。--gpu-wait-seconds可限定等锁时间，默认立即退出。不要凭锁文件中的旧PID判断锁是否仍占用。

state=completed只表示进程成功退出；metrics_validated始终先为false。赛题成绩需要单独的固定验证证据，不能根据任务进程退出码推断。

## 数据检查边界

ZIP审查报告涵盖摘要、CRC（显式启用）、文件目录、重名、空文件、加密和不安全路径。audit_data进一步全量解码TIFF并校验标签/任务覆盖、尺寸、dtype、SAR复制通道、文件/像素重复与近重复候选。通过结构检查不等于标签真值已确认。

```bash
python tools/audit_data.py data/stage1.zip --output-dir local/data-audit --workers 4
python tools/make_split.py data/stage1.zip --audit-dir local/data-audit --output local/splits/identity-v1.json
python tools/run_baseline.py data/stage1.zip --split local/splits/identity-v1.json --output-dir local/benchmarks/gray32 --predict-test
python tools/validate_submission.py data/stage1.zip local/benchmarks/gray32/prediction.json
python tools/check_metrics.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/benchmarks/gray32 --output local/benchmarks/gray32/verification.json
python -m unittest discover -s tests -v
```

check_metrics.py以独立标量逐位累加核对R@1、两种AP分母及三方向加权，不调用track6.metrics；同时核对保存产物摘要与预测结构。predict_adapter.py重新加载检查点和绑定的特征缓存，复现验证排序与prediction.json，无需重新训练。

工具实现和真实结构位于track6/，正式模型命令见[基准章](../docs/08-baselines.md)。身份标签、划分、逐图审查、特征、权重与预测只保存在Git忽略目录。验收使用合成数据覆盖安全路径、CSV/JSON结构、解码类型、并发ZIP字节一致性、跨进程读取、身份泄漏、Query自匹配、AP分母、重复排序、非法预测、三方向加权与相同种子重现；Linux另核对共享锁保护。
