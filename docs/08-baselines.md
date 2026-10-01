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

结果将在完整模型运行后更新。report.json记录三个方向、两种分母、代码版本、数据与划分摘要、种子、预处理、用时、显存和提交格式校验。run_job的completed只表示退出码0，其metrics_validated保持false；模型效果以report.json及指标复算为准。

本地验证规模和构造不等于排行榜；当前未向比赛平台提交。
