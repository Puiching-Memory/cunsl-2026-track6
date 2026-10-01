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

## 现代骨干与图像级监督

run_dinov3.py严格加载逐项审查的本地safetensors，在单次编码中比较CLS、均值和前景池化，保存各自特征与三方向报告；不会自动从hub下载模型。train_adapter.py支持这些审查过的DINOv3缓存，输出适配后特征供后续对照。train_reid.py使用训练身份的实际图像与配对监督微调编码器末端，predict_reid.py从绑定的公开初始化恢复部分参数检查点。资源与方法边界见[新模型章](../docs/09-modern-models.md)。

依赖可隔离安装：`python -m pip install --no-deps --target local/dinov3-deps -r requirements-models.txt`，运行前设置PYTHONPATH指向该目录。所有GPU运行继续使用共同gpu.lock，模型输出和检查点留在Git之外。

上述新路线已在真实5090上运行：两组身份划分分别重训，20份报告独立复算，图像微调检查点恢复后排名/预测逐字节一致。check_metrics自动核对特征缓存、线性适配器及预测摘要；图像级reid.pt摘要另外与训练报告比较，再执行predict_reid覆盖恢复检查，不能把格式校验等同于效果证明。完整命令和配置记录见第9章。
# 领域方法对照与错配分析

`train_reid.py` 可用 `--stem-mode frozen/shared/separate` 比较固定图像入口、可训练共享入口和可训练光学/SAR入口；`--structure-weight 1 --structure-layer 6` 添加中间层 patch 梯度能量一致性约束。默认值仍为 `frozen` 和结构权重 `0`，旧检查点可继续恢复。等预算配置见[领域对照配置](../configs/domain-ablation.example.toml)。这些是自行实现的 DINOv3 改造，不是 TransOSS、MOS 或 SDF-Net 的完整复现。

```bash
python tools/analyze_retrieval.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/benchmarks/dinov3-b16-reid12 --output-dir local/error-analysis/split1
```

错配工具核对特征与划分指纹，重新计算全 Gallery 排名并验证原有前十顺序，然后统计各方向 R@1/5/10/50、正确候选排名、相似度边际及像素占用率。仅分析有训练标签的身份隔离验证集，不能用于猜测初赛身份。`report.json` 是匿名汇总；`private-errors.json`、可选 `--panel` 生成的原始像素图和案例清单均须留在 Git 之外。大于 8 的非黑区域是观察性统计，不是舰船分割，也不能换算物理船长。

