# 分模态入口与结构约束：本题实测

2026-10-07起本页保留为历史研究记录。DINO实现已删除；当前TransOSS路线与入口见[项目结构](14-project-structure.md)和[工具说明](../tools/README.md)。

本轮把[领域研究](10-domain-research.md)落到可比较的实验：保留同一个已审查DINOv3-B/16初始化，只改变patch输入层是否训练、是否按模态分开，以及是否添加中间层结构损失。**独立光学/SAR入口在两组身份隔离验证中均超过原固定入口；结构损失效果不一致，暂不保留。**这些是本地验证结果，未提交排行榜，正式AP@10程序仍待确认。

## 控制条件与实现边界

训练代码为干净的`ec180f5f9bb7305ac4d8b25ba1b762a63c115dde`。数据、异常排除与两组划分沿用[第7章](07-validation.md)和[第二组摘要](evidence/2026-10-01-split2.json)，未增加外部数据或权重。各变体均从同一公开初始化重新训练，未从另一划分训练模型继续微调。

- 256×256完整图像；BF16编码器、FP32池化/损失；CLS加阈值区域池化。阈值区域不是人工舰船分割。
- 共享主干末4块及最终norm、两模态投影头、共享身份分类器沿用第9章。
- 每批32个身份，每身份1光学+1SAR；首/次划分2,819/2,816个可用成对训练身份。
- 固定12轮最终模型、种子20261001/20261002；未用验证选训练检查点。每轮89/88步，训练无增强。
- AdamW，骨干学习率2e-5、头1e-3、weight decay 0.01；2轮warmup和cosine衰减。InfoNCE温度0.07，身份分类权重0.5、triplet权重0.2，其余与原基线一致。

四种设置：固定入口控制；可训练共享入口；可训练独立光学/SAR入口；独立入口加第6块结构损失。独立入口复制相同已审查权重，不随机重置；新增SAR入口参数纳入骨干学习率和保存/恢复。

结构损失取第6块之后、最终norm之前的16×16 patch token，排除CLS与4个register；将横纵相邻差分的绝对值空间均值相加，在每张图的通道维标准化，配对MSE权重1。它是SDF启发的独立适配，不是完整SDF-Net复现，不含作者解耦token、正交损失或领域预训练权重。入口可训练提供了第6层之前的梯度路径；后续被冻结的块仍会对输入反向传播，因此显存和耗时增加。

配置见[domain-ablation.example.toml](archive/dino-settings.md#domain-ablation-example-toml)。本轮为等样本/等轮数比较，不是等参数/等算力；第二组固定入口复用已独立验证的第9章原结果，其代码版本为4c2394b，表中明确标记。

## 三方向指标

各方向单元格为**R@1 / mAP@10，单位%**；加权分仍为0–1。两种已实现AP分母在这些验证Gallery上相同，因为每Query最多3个相关图，不代表正式评分程序已经确认。

| 划分/种子 | 设置 | O2S | S2O | O2O | 加权分 |
|---|---|---:|---:|---:|---:|
| 1 / 20261001 | 固定入口，重算控制 | 20.78 / 34.61 | 21.57 / 32.99 | 86.36 / 92.05 | 0.33659 |
| 1 / 20261001 | 可训练共享入口 | 21.18 / 33.84 | 23.67 / 33.76 | 88.64 / 92.61 | 0.34363 |
| 1 / 20261001 | 可训练独立入口 | **22.88 / 35.12** | **23.11 / 34.25** | **95.45 / 96.40** | **0.35548** |
| 1 / 20261001 | 独立入口+结构损失 | 22.09 / 34.47 | 20.87 / 32.63 | 86.36 / 91.86 | 0.33674 |
| 2 / 20261002 | 固定入口，第9章历史控制 | 20.21 / 32.26 | 20.03 / 31.10 | 75.00 / 81.82 | 0.31150 |
| 2 / 20261002 | 可训练共享入口 | 21.39 / 33.60 | 21.00 / 32.59 | 70.00 / 80.77 | 0.31970 |
| 2 / 20261002 | 可训练独立入口 | **21.52 / 34.54** | **21.42 / 33.00** | **75.00 / 84.46** | **0.32832** |
| 2 / 20261002 | 独立入口+结构损失 | 23.10 / 35.57 | 21.00 / 33.60 | 72.50 / 83.94 | 0.33307 |

独立入口相对固定入口加权分增加0.01888/0.01682。排除O2O后，两跨模态方向得分的均值仍增加0.01351/0.01723；O2S和S2O的R@1及mAP@10在两组均提高。相对可训练共享入口，加权分再增加0.01185/0.00862，但首组S2O单方向得分略低，不能说每项指标都领先共享入口。

结构损失相对独立入口，在首组下降0.01874、次组上升0.00475。其训练损失从约0.13–0.14下降到约0.08，说明约束被优化，不能说明检索泛化改善。两组加权分平均值分别为独立入口0.34190、结构版0.33491；仅为同源两组的描述性均值，没有独立外部测试、多训练seed重复或统计显著性结论。

## 时间、参数与显存

下表显存为PyTorch峰值allocated GiB，不是整张显卡占用。时间为训练阶段，不含约6–7秒数据预载和约6–8秒推理；完整端到端时间、reserved显存、逐轮损失均保存在[完整报告](evidence/2026-10-01-domain-ablations.json)。

| 划分 | 设置 | 可训练参数 | 训练秒数 | 峰值GiB |
|---|---|---:|---:|---:|
| 1 | 固定入口控制 | 37,398,528 | 108.86 | 3.303 |
| 1 | 共享入口 | 37,989,120 | 135.48 | 6.134 |
| 1 | 独立入口 | 38,579,712 | 138.33 | 6.167 |
| 1 | 独立入口+结构 | 38,579,712 | 138.81 | 6.258 |
| 2 | 固定入口历史控制 | 37,393,920 | 110.43 | 3.302 |
| 2 | 共享入口 | 37,984,512 | 133.77 | 6.132 |
| 2 | 独立入口 | 38,575,104 | 135.39 | 6.165 |
| 2 | 独立入口+结构 | 38,575,104 | 138.35 | 6.258 |

新增参数较少，但训练入口需保留冻结前段的输入梯度，显存约从3.3增至6.2GiB，训练时间增加约25%–27%。当前32GiB单卡实际运行通过；不据此估计更大模型或全解冻的性能。

## 验证与失败记录

实际EVA-BF16前向检查证实：尚未训练时，共享/独立入口输出与原生路径逐元素一致，最大绝对差0；捕获中间结构不改变最终输出。首组固定入口重算的验证排序和测试预测与第9章旧结果逐字节一致。

7次训练和6次恢复推理，共13份保存排序，全部经独立标量AP/R@1复算、摘要绑定及267个Query的10候选格式检查。6个新模型检查点完整SHA核验通过，重新载入后预测和排序逐字节一致。结构版保存的是训练所得参数，恢复推理不计算训练结构损失。算力机29项检查全通过，Windows 21项通过、8项因平台或无Torch跳过。

训练前的一次真实编码入口检查失败：fd19版本假设patch输出为NLC，但已审查EVA实际返回NHWC，模态掩码广播维度不符。ec180修复后，通过真实初始化一致性检查，再启动全部训练。失败退出码和原始日志保留，未把失败运行计为模型成绩。

所有15次GPU调用（1失败、14成功）均持共同文件锁，并检查已有GPU进程；未终止第5题。匿名证据分别见[评分与恢复核验](evidence/2026-10-01-domain-verification.json)、[运行清单](evidence/2026-10-01-domain-runs.json)和[阅读源摘要](evidence/2026-10-01-domain-sources.json)。原始图像、身份、排序、预测、权重和检查点留在Git之外。

## 复现与下一轮

使用新输出目录，先提交源代码且保持干净；TOML记录设置，实际CLI如下。第二组同时换split和seed，不能复用首组训练检查点。

```bash
cd /root/autodl-tmp/cunsl-2026/track6/project
TRACK6_PY=/root/miniconda3/bin/python
TRACK6_LOCAL=/root/autodl-tmp/cunsl-2026/track6/local
TRACK6_DATA=/root/autodl-tmp/cunsl-2026/track6/data
TRACK6_OUT="$TRACK6_LOCAL/domain-reproduce/separate"
export PYTHONPATH="$TRACK6_LOCAL/dinov3-deps:$PWD"
"$TRACK6_PY" tools/run_job.py --name reproduce-sensor-input --output-root "$TRACK6_LOCAL/runs" \
  --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- \
  "$TRACK6_PY" tools/train_reid.py "$TRACK6_DATA/stage1.zip" \
  --split "$TRACK6_LOCAL/splits/identity-v1.json" --seed 20261001 \
  --resource resources/dinov3-vitb16-timm.json --weights "$TRACK6_DATA/weights/dinov3-vitb16-timm.safetensors" \
  --output-dir "$TRACK6_OUT" --epochs 12 --identities-per-batch 32 --image-size 256 \
  --pooling cls_foreground --train-last-blocks 4 --backbone-lr .00002 --head-lr .001 \
  --stem-mode separate --structure-weight 0 --predict-test
"$TRACK6_PY" tools/check_metrics.py "$TRACK6_DATA/stage1.zip" \
  --split "$TRACK6_LOCAL/splits/identity-v1.json" --run-dir "$TRACK6_OUT" \
  --output "$TRACK6_OUT/verification.json"
"$TRACK6_PY" tools/run_job.py --name reproduce-sensor-replay --output-root "$TRACK6_LOCAL/runs" \
  --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- \
  "$TRACK6_PY" tools/predict_reid.py "$TRACK6_DATA/stage1.zip" --split "$TRACK6_LOCAL/splits/identity-v1.json" \
  --resource resources/dinov3-vitb16-timm.json --weights "$TRACK6_DATA/weights/dinov3-vitb16-timm.safetensors" \
  --checkpoint "$TRACK6_OUT/reid.pt" --output-dir "$TRACK6_OUT-replay" --predict-test
```

下一基线固定为独立入口、不加结构损失，优先研究光学背景干扰与局部表示，再单测增强和已审查的配对预训练。首组预测存于算力机`local/domain-ablation/split1/separate/prediction.json`，本地私有副本为`local/domain-results/domain-ablation/split1/separate/prediction.json`。这仍只训练首组训练身份，尚未全身份重训，不把当前文件当作已确认的最佳正式提交。
