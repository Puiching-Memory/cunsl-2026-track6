# 基准实验与复现

全部本地指标使用[固定身份协议](07-validation.md)，指标口径与排行榜结果分别记录。输出留在local/或算力机数据盘。GPU任务使用共同文件锁，运行前检查现有进程。

## 环境与命令

算力机使用/root/miniconda3/bin/python和已有PyTorch 2.12.1+cu130、torchvision 0.27.1+cu130、numpy 2.4.6、Pillow 12.2.0；不覆盖可用CUDA环境。Windows数据工具另实测Python 3.12.14、numpy 2.3.5、Pillow 12.3.0。工具测试只需numpy/Pillow和Python标准库。

以下从算力机track6/project执行。共享锁由run_job.py持有，直到子进程结束。

```bash
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/cunsl-2026/track6/data/stage1.zip
LOCAL=/root/autodl-tmp/cunsl-2026/track6/local
LOCK=/root/autodl-tmp/cunsl-2026/gpu.lock
VENDOR=$LOCAL/vendor/dinov2
WEIGHT=/root/autodl-tmp/cunsl-2026/track6/data/weights/dinov2_vits14_pretrain.pth

$PY tools/inspect_archive.py "$DATA" --verify-crc --output "$LOCAL/archive-audit.json"
$PY tools/audit_data.py "$DATA" --output-dir "$LOCAL/data-audit" --workers 4
$PY tools/make_split.py "$DATA" --audit-dir "$LOCAL/data-audit" --output "$LOCAL/splits/identity-v1.json"
$PY -m unittest discover -s tests -v

$PY tools/run_job.py --name gray32-holdout --output-root "$LOCAL/runs" -- \
  "$PY" tools/run_baseline.py "$DATA" --split "$LOCAL/splits/identity-v1.json" \
  --output-dir "$LOCAL/benchmarks/gray32" --predict-test

git clone https://github.com/facebookresearch/dinov2.git "$VENDOR"
git -C "$VENDOR" checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
$PY tools/fetch_weight.py resources/dinov2-vits14.json "$WEIGHT"

$PY tools/run_job.py --name dino-frozen --output-root "$LOCAL/runs" --gpu-lock "$LOCK" -- \
  "$PY" tools/run_baseline.py "$DATA" --split "$LOCAL/splits/identity-v1.json" \
  --output-dir "$LOCAL/benchmarks/dino-frozen" --model dinov2_vits14 \
  --vendor "$VENDOR" --weights "$WEIGHT" --resource resources/dinov2-vits14.json \
  --batch-size 32 --cache-train --predict-test

$PY tools/run_job.py --name dino-linear20 --output-root "$LOCAL/runs" --gpu-lock "$LOCK" -- \
  "$PY" tools/train_adapter.py "$DATA" --split "$LOCAL/splits/identity-v1.json" \
  --features-dir "$LOCAL/benchmarks/dino-frozen" --output-dir "$LOCAL/benchmarks/dino-linear20" \
  --seed 20261001 --epochs 20 --identities-per-batch 128 --lr 0.001 \
  --temperature 0.07 --identity-regularization 0.01 --predict-test

$PY tools/validate_submission.py "$DATA" "$LOCAL/benchmarks/dino-linear20/prediction.json"
$PY tools/check_metrics.py "$DATA" --split "$LOCAL/splits/identity-v1.json" \
  --run-dir "$LOCAL/benchmarks/dino-linear20" --output "$LOCAL/benchmarks/dino-linear20/verification.json"

$PY tools/run_job.py --name adapter-replay --output-root "$LOCAL/runs" --gpu-lock "$LOCK" -- \
  "$PY" tools/predict_adapter.py "$DATA" --split "$LOCAL/splits/identity-v1.json" \
  --features-dir "$LOCAL/benchmarks/dino-frozen" \
  --checkpoint "$LOCAL/benchmarks/dino-linear20/adapter.pt" --output-dir "$LOCAL/benchmarks/adapter-replay"
```

资源下载前审查，下载后将实际SHA-256登记到resources/dinov2-vits14.json；没有具体摘要时GPU工具拒绝使用权重。vendor提交也必须与清单一致。原始ViT-S权重许可单独核对，不覆盖其他DINOv2变体。

## 比较对象

gray32将浮点灰度缩为32×32，对每图标准化，附加常数维后进行余弦排序，不用外部模型或训练标签拟合。常量测试图仍有可排序特征。

DINOv2冻结基准将完整256×256图以bicubic缩为224×224，按ImageNet均值/标准差处理；原始无registers ViT-S/14输出384维CLS，BF16前向、FP32特征和排序。模态保留公开元数据。冻结基准与适配器共享同一输入和特征缓存。

线性适配器为光学/SAR各一个384×384无偏置线性层，初始化单位矩阵。每批128个不同训练身份，各随机选1张光学与1张SAR；归一化特征计算双向InfoNCE，加入单位矩阵偏离正则。AdamW、FP32、固定20轮最终模型，骨干冻结。测试和验证均不参与优化。

## 结果与证据

2026-10-01 三组正式基准均在干净的 `26f73f30adec11eefbae49ae3789a2e41e40c598` 上运行，种子20261001、同一身份划分。每格为 R@1 / mAP@10，以下采用0–1量纲；两种分母在当前正样本数≤3的协议上相同。

| 基准 | O2S | S2O | O2O | 加权分 |
|---|---|---|---|---:|
| gray32 | 0.09804 / 0.16148 | 0.10924 / 0.16186 | 0.77273 / 0.82367 | 0.19921 |
| 冻结DINOv2-S/14 | 0.00915 / 0.02400 | 0.00840 / 0.01735 | 0.68182 / 0.75246 | 0.08497 |
| 冻结DINOv2 + 双线性适配器20轮 | 0.13725 / 0.24662 | 0.14286 / 0.24141 | 0.75000 / 0.80208 | 0.25044 |

适配器在当前划分提升两个跨模态方向，O2O仍低于gray32；其加权分比gray32高0.05123。冻结DINOv2直接检索明显较弱，因此不能把候选通用模型视作本题有效框架。这里只证明这组具体设置的本地结果，未证明多种划分下稳定改善或算法创新。

| 运行 | 实际阶段用时 | 报告总用时 | PyTorch峰值 allocated / reserved |
|---|---|---:|---:|
| gray32 | 3,550张特征5.55秒 | 7.04秒 | CPU，无CUDA分配 |
| 冻结DINOv2 | 9,476张特征25.16秒（包括训练缓存；模型内前向22.78秒） | 26.86秒 | 249.27 / 334.00 MiB |
| 适配器 | 20轮训练4.59秒；缓存特征推理/评分0.71秒 | 7.28秒 | 88.39 / 102.00 MiB |

适配器处理2,819个仍成对的训练身份，每轮23步，共460步，294,912个可训练参数。总用时未包含上游DINO特征提取，完整路线需另计25.16秒及首次权重下载；灰度与DINO本次提取图像数不同，不能按上表直接比较每图速度。显存数值是PyTorch分配器峰值，不包含驱动、CUDA上下文或其他进程占用。

证据见 [基准原始报告](evidence/2026-10-01-baselines.json)、[独立复算](evidence/2026-10-01-metric-verification.json) 和 [运行记录](evidence/2026-10-01-runs.json)。三个模型的保存排序均由独立标量逐位AP程序复算，误差≤1e-12；三个prediction.json均通过267个Query、10个唯一候选、完整覆盖及目标模态校验。适配器检查点重新加载后，验证排序和测试预测均逐字节一致。

首轮候选文件为算力机 `local/benchmarks/dino-linear20/prediction.json`，Windows副本为仓库忽略目录 `local/remote-results/benchmarks/dino-linear20/prediction.json`；它使用本轮未见身份验证的固定最终模型。随后完成的DINOv3升级、第二组划分对照与较优候选见[第9章](09-modern-models.md)。尚未使用所有可用训练身份重训，也未向比赛平台提交。

失败记录：初次算力机并发ZIP审查在c423a7b上退出码1，8个间歇读取错误。26f73f3为共享读取加锁后，9,489张重新审查全部通过；身份划分跨Windows/Linux逐字节一致。外部库xFormers可选加速被显式关闭，DINO以现有PyTorch运行，未覆盖CUDA环境。失败原始日志保留，见研究日志和运行记录。

report.json记录三个方向、两种分母、代码版本、数据与划分摘要、种子、预处理、用时、显存和提交格式校验。run_job的completed只表示退出码0，其metrics_validated保持false；模型效果以report.json及独立指标复算为准。AP@10正式程序仍待取得，O2O仅44个Query，下一阶段先做其他身份划分、失败样本审查和同预算对照，再安排全训练重训和有限提交。
