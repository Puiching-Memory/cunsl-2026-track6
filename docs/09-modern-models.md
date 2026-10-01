# 新骨干与识别效果实验

## 研究重心

首轮gray32、冻结DINOv2-S/14及线性适配器用于建立正确、可复现的流程。0.25044只是一次本地身份隔离划分上的起点，不能代表先进算法或最终方案。后续优先提升两个跨模态方向的识别效果，以三方向完整指标和独立复算决定取舍。

## DINOv3资源审查

Meta在2025年8月发布[DINOv3](https://github.com/facebookresearch/dinov3)。官方包含通用LVD-1689M和卫星SAT-493M系列；卫星版是光学RGB预训练，不能称为SAR预训练。原始官方分发页面要求账号持有人提交联系信息、同意许可。按用户授权寻找公开转载，已接入维护者Ross Wightman在`timm`组织发布的转换版本。

| 具体资源 | 数据来源 | 文件字节数 | 首次公开 |
|---|---|---:|---|
| [ViT-B/16](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m) | LVD-1689M | 342,579,728 | 2025-09-17 |
| [ViT-L/16卫星版](https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m) | SAT-493M | 1,212,347,640 | 2025-09-17 |

维护者的初始提交已含精确safetensors文件；当日当前版本的LFS完整SHA-256与初始版本相同。两个仓库均为公开非gated资源。本地完整下载和传入算力机后的摘要均核验通过，实际权重保存在data/weights，原始模型卡、许可及API历史快照保存在local/research。

精确来源、权重摘要与许可分别记录在[通用版资源清单](../resources/dinov3-vitb16-timm.json)和[卫星版资源清单](../resources/dinov3-vitl16-sat-timm.json)。timm代码固定1.0.20/tag 234907e，2025-09-21公开，代码Apache-2.0；已核验PyPI原始wheel摘要及EVA实现文件。权重继续受[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)约束，不能随代码称为Apache-2.0。发布研究需致谢；分发权重或衍生权重需随附该许可并遵守其中用途限制。

适用边界：按2026-10-01已核对的赛题规则，这两份具体权重满足2026-09-01之前公开与资源登记条件；这是资源条件审查，不是主办方对DINOv3的专项审核结论。当前只用于公开算法竞赛研究，在自有环境上使用正式训练标签优化、测试图像推理，未导入AIS、位置、时间等未公开辅助信息。

许可还必须单独满足。[Meta官网许可](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)第1.b.v限制受ITAR约束的活动及贸易管制禁止的最终用途，列举了军事/战争、核工业、间谍等用途。不能把“舰船”题名自动等同于军用，也不能由公开下载推断任何最终用途均获授权；如果涉及这类交付用途，需另核对，当前没有此类用途的授权结论。timm转换版本继续受这些条款约束。

2026-10-01再次读取Meta官网许可发现：官网标记日期为2025-08-14，并包含第三方分发时显著展示“Built with DINOv3”的要求；timm仓库内已存档的LICENSE标记2025-08-19，文字有所不同。两份来源分别保留，不将版本视为完全相同。实际发布或交付时复核适用许可，并准备随附许可、明确致谢和DINOv3使用标识。本次重新访问授权赛题页面的浏览器连接超时，竞赛条款依据当天已有核对记录；权重初始提交日期与官网许可已重新读取。

timm转换去掉原权重中的零QKV偏置，使用FP32生成RoPE周期；模型卡明确描述这种数值差异。采用timm默认行为，记录实现与具体checkpoint，不声称与Meta原生权重逐字节或输出完全相同。卫星版归一化均值/标准差也不同于通用版，读取固定版本的模型配置，写入每次报告。

环境准备中算力机直接访问Hugging Face返回Network is unreachable，run_job保留失败记录。随后在本地下载并上传，两个端点均通过发布方完整SHA-256。依赖通过--no-deps安装到local/dinov3-deps，现有PyTorch/CUDA环境保留。

## 对照与训练路线

所有首轮对照保持既有identity-v1划分与种子20261001，输入使用完整256×256图像，明确与首轮DINOv2的224×224输入差别。冻结提取一次同时比较CLS、patch均值，以及CLS+前景patch均值。前景权重使用原图max(RGB)>8/255的软patch覆盖率；只依据已提供像素，空mask回退均值；该阈值是工程假设，尚不等于可靠分割或物理散射模型。

先在各冻结特征上训练相同20轮双模态线性适配器，避免把换骨干与新损失的影响混在一起。再使用train_reid微调编码器末端4个Transformer块和最终归一化，训练两个模态投影及共享身份分类器。使用成对身份采样、对称InfoNCE、身份交叉熵与软batch-hard triplet；可选独立水平/垂直翻转，完整保留船体，不随机裁掉目标。该实现是本题监督ReID对照，不是下列论文的忠实复现。

验证与测试身份不进入优化。训练预算事先固定12轮、每批32个身份，保存最终轮，暂不按验证选择epoch；推理后保存特征、排序、实际用时与显存。部分训练参数检查点绑定精确公开初始化、划分与数据摘要，predict_reid严格覆盖检查后恢复，用于验证保存模型的可复现推理。

## 与本题直接相关的研究

- [TransOSS，ICCV 2025](https://arxiv.org/abs/2506.22027)及[官方实现](https://github.com/Alioth2000/Hoss-ReID)：提出光学/SAR舰船重识别、模态独立patch输入和共享Transformer。其公开协议、数据规模与本题不同；涉及实物尺寸的附加输入不能在本题未经公开提供时使用。
- [MOS，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_MOS_Mitigating_Optical-SAR_Modality_Gap_for_Cross-Modal_Ship_Re-Identification_CVPR_2026_paper.pdf)及[官方实现](https://github.com/yjzhao1019/MOS)：研究SAR去噪、身份内模态对齐及生成模态的特征融合。作者标注代码2026年3月公开，但伪光学转SAR样本于2026年9月发布，晚于本题外部资源截止，不使用该批样本。README更正融合系数τ为0.02，不能照抄论文中的笔误。生成细节可靠性和推理成本仍需本题验证。
- [SDF-Net（2026）](https://arxiv.org/abs/2603.12588)及[官方实现](https://github.com/cfrfree/SDF-Net)：使用中间特征结构约束、共享和模态特定表示。结构池化和监督对齐值得做独立消融，其论文效果尚未在本题复现。

当前不导入外部HOSS训练数据或基于其身份训练的ReID权重，需先核对身份重叠、资源日期、权重许可和本题允许信息。只把上述论文作为设计参考，论文在HOSS上的分数不能称为本题成绩。

## 第一组实测：seed=20261001

全部新实验来自干净提交`4c2394bdc35b340ab6f53de25acbb8a5a2e82990`，沿用第7章的身份隔离划分；旧DINOv2对照来自已归档的26f73f3。下表每方向为**R@1% / mAP@10%**，加权分保持0–1尺度。全部保存排名独立复算，两种显式AP分母在此数据上相同；尚未拿到官方评分程序。

| 模型与设置 | O2S | S2O | O2O | 加权分 |
|---|---:|---:|---:|---:|
| 旧DINOv2-S/14 + 20轮适配器 | 13.73 / 24.66 | 14.29 / 24.14 | 75.00 / 80.21 | 0.25044 |
| DINOv3-B 冻结CLS | 1.31 / 3.02 | 0.70 / 1.38 | 68.18 / 75.34 | 0.08618 |
| DINOv3-B 冻结patch均值 | 2.48 / 3.90 | 1.26 / 2.15 | 63.64 / 69.24 | 0.08847 |
| DINOv3-B 冻结CLS+前景 | 1.83 / 3.30 | 0.70 / 1.31 | 63.64 / 71.24 | 0.08349 |
| DINOv3-B CLS + 20轮适配器 | 16.34 / 26.48 | 16.25 / 24.80 | 77.27 / 86.52 | 0.27059 |
| DINOv3-B patch均值 + 20轮适配器 | 16.34 / 28.55 | 17.79 / 28.28 | 75.00 / 81.01 | 0.28267 |
| DINOv3-B CLS+前景 + 20轮适配器 | 17.78 / 29.12 | 17.65 / 27.69 | 79.55 / 86.08 | 0.29035 |
| DINOv3-L SAT 冻结CLS | 0.65 / 1.50 | 0.28 / 1.20 | 43.18 / 46.66 | 0.05308 |
| DINOv3-L SAT 冻结patch均值 | 0.26 / 1.16 | 0.28 / 1.09 | 45.45 / 48.81 | 0.05340 |
| DINOv3-L SAT 冻结CLS+前景 | 0.52 / 1.30 | 0.28 / 0.64 | 52.27 / 55.42 | 0.06001 |
| DINOv3-L SAT CLS + 20轮适配器 | 16.47 / 24.95 | 13.31 / 21.63 | 59.09 / 69.44 | 0.23605 |
| DINOv3-L SAT patch均值 + 20轮适配器 | 11.63 / 21.00 | 10.64 / 18.33 | 52.27 / 58.18 | 0.19383 |
| DINOv3-L SAT CLS+前景 + 20轮适配器 | 16.34 / 26.77 | 14.71 / 24.76 | 65.91 / 73.91 | 0.25571 |
| **DINOv3-B CLS+前景，末4块微调12轮** | **20.78 / 34.61** | **21.57 / 32.99** | **86.36 / 92.05** | **0.33659** |

在这组划分中，监督对齐比直接冻结检索有效，通用B版适配后优于SAT-L版；选择B版继续微调。该比较尚不能说明卫星版的充分微调上限，也不能单独归因于DINO版本：旧模型输入224、新模型256，特征384/768/1536维，对应头部参数量不同。CLS+前景的改善也需固定表示维度的后续消融。

实际微调使用2,819个成对训练身份，5,918张可用配对图，37,398,528个可训练参数。每轮89步，共1,068步；两个模态投影为1536维，编码器其余部分冻结。没有图像增强，采用两轮warmup、余弦衰减、梯度裁剪1、InfoNCE温度0.07、身份分类权重0.5和triplet权重0.2，AdamW权重衰减0.01。训练设定与逐轮损失完整保存在匿名报告。

| 路线 | 实际阶段用时 | PyTorch allocated / reserved峰值 |
|---|---|---:|
| B/16冻结编码，共享3种池化 | 9,476张，18.84秒，包含解码 | 587.32 / 778.00 MiB |
| SAT-L/16冻结编码，共享3种池化 | 9,476张，27.95秒，包含解码 | 1484.53 / 1734.00 MiB |
| B/16 CLS+前景线性适配器 | 缓存训练3.71秒；上游编码另计 | 见完整报告 |
| B/16末4块图像微调 | 12轮训练109.35秒；数据预载、推理另计 | 3381.93 / 3982.00 MiB |

显存是PyTorch分配器峰值，不包括CUDA上下文或驱动占用；阶段用时不等于从下载到预测的全流程耗时。

## 第二组实测：seed=20261002

依据第一组结果固定B/16 CLS+前景和12轮设置，再生成另一组身份隔离划分。训练2,827个身份/5,927张图，验证707个身份/1,482张图；方向Query/Gallery为O2S 762/720、S2O 719/762、O2O 40/722。[划分汇总](evidence/2026-10-01-split2.json)记录完整摘要`aa7b5835cbbf9699b4b0d1ece79717d36cd9a169ea5d273f6e7c3bb0a39304e0`，身份、像素和所用近重复边跨区重叠均为0。

旧、新方案各自从原始公开初始化重新训练，只用第二组训练身份。没有复用第一组训练模型评估第二组验证身份。下表单位同上。

| 模型与设置 | O2S | S2O | O2O | 加权分 |
|---|---:|---:|---:|---:|
| 冻结DINOv2-S/14 | 1.84 / 3.16 | 0.70 / 1.58 | 55.00 / 66.51 | 0.07712 |
| DINOv2 + 20轮适配器 | 13.78 / 24.61 | 13.21 / 23.49 | 57.50 / 68.69 | 0.23205 |
| DINOv3-B 冻结CLS+前景 | 1.31 / 2.53 | 0.56 / 1.03 | 62.50 / 65.95 | 0.07645 |
| DINOv3-B CLS+前景 + 20轮适配器 | 17.19 / 27.63 | 16.69 / 26.66 | 72.50 / 81.29 | 0.27527 |
| **DINOv3-B CLS+前景，末4块微调12轮** | **20.21 / 32.26** | **20.03 / 31.10** | **75.00 / 81.82** | **0.31150** |

第二组微调2,816个成对身份，训练110.43秒。相对旧适配器，两组加权分分别提升0.08616和0.07945，两个跨模态方向的R@1与mAP@10均改善。因此当前主线选B/16监督微调。两组来自同一训练包，验证身份可能部分重合，且同时更换了划分与训练随机种子；不是独立外部测试，也不是充分的统计稳定性证明。O2O仅44/40个Query，不根据其单次波动宣称普遍领先。

## 独立核对与复现

[完整实测报告](evidence/2026-10-01-modern-models.json)包含20份报告，其中18个实验设置和2次检查点重放；[核对证据](evidence/2026-10-01-modern-verification.json)独立逐位重算所有排名，误差≤1e-12。所有预测恰好覆盖267个Query、各10个有效唯一目标模态候选。两组图像微调检查点重新加载后，验证排名和测试预测均逐字节一致；检查点本体摘要也已单独核验。

[运行清单](evidence/2026-10-01-modern-runs.json)记录16次持GPU锁的成功模型运行及1次CPU下载失败；每次模型运行先检查已有GPU进程，源代码干净，未终止第5题。图像微调、损失、部分参数保存覆盖与恢复的真实测试已通过；算力机27项检查全通过，Windows 21项通过、6项按平台或无PyTorch跳过。[外部资源证据](evidence/2026-10-01-dinov3-resources.json)保存原始发布时间、精确权重、下载/传输摘要与代码许可。实验使用的是4c2394b中的资源清单摘要，后续仅完善清单说明，不改变具体权重。

复现需使用已核验正式ZIP、审查生成的私有划分和指定本地权重。新配置记录见[dinov3-reid.example.toml](../configs/dinov3-reid.example.toml)，TOML用于记录设置，下列CLI是实际执行入口；输出使用新的复现目录：

```bash
cd /root/autodl-tmp/cunsl-2026/track6/project
TRACK6_PY=/root/miniconda3/bin/python
TRACK6_LOCAL=/root/autodl-tmp/cunsl-2026/track6/local
TRACK6_DATA=/root/autodl-tmp/cunsl-2026/track6/data
TRACK6_OUT="$TRACK6_LOCAL/benchmarks/reproduce-b16"
export PYTHONPATH="$TRACK6_LOCAL/dinov3-deps"
"$TRACK6_PY" tools/run_job.py --name reproduce-b16-frozen --output-root "$TRACK6_LOCAL/runs" \
  --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- \
  "$TRACK6_PY" tools/run_dinov3.py "$TRACK6_DATA/stage1.zip" --split "$TRACK6_LOCAL/splits/identity-v1.json" \
  --resource resources/dinov3-vitb16-timm.json --weights "$TRACK6_DATA/weights/dinov3-vitb16-timm.safetensors" \
  --output-root "$TRACK6_OUT/frozen" --poolings cls avg cls_foreground \
  --image-size 256 --batch-size 32 --cache-train --predict-test
"$TRACK6_PY" tools/run_job.py --name reproduce-b16-reid12 --output-root "$TRACK6_LOCAL/runs" \
  --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- \
  "$TRACK6_PY" tools/train_reid.py "$TRACK6_DATA/stage1.zip" --split "$TRACK6_LOCAL/splits/identity-v1.json" \
  --resource resources/dinov3-vitb16-timm.json --weights "$TRACK6_DATA/weights/dinov3-vitb16-timm.safetensors" \
  --output-dir "$TRACK6_OUT/reid" --seed 20261001 --epochs 12 \
  --identities-per-batch 32 --image-size 256 --pooling cls_foreground --train-last-blocks 4 \
  --backbone-lr .00002 --head-lr .001 --predict-test
"$TRACK6_PY" tools/check_metrics.py "$TRACK6_DATA/stage1.zip" --split "$TRACK6_LOCAL/splits/identity-v1.json" \
  --run-dir "$TRACK6_OUT/reid" --output "$TRACK6_OUT/reid/verification.json"
"$TRACK6_PY" tools/run_job.py --name reproduce-b16-replay --output-root "$TRACK6_LOCAL/runs" \
  --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- \
  "$TRACK6_PY" tools/predict_reid.py "$TRACK6_DATA/stage1.zip" --split "$TRACK6_LOCAL/splits/identity-v1.json" \
  --resource resources/dinov3-vitb16-timm.json --weights "$TRACK6_DATA/weights/dinov3-vitb16-timm.safetensors" \
  --checkpoint "$TRACK6_OUT/reid/reid.pt" --output-dir "$TRACK6_OUT/replay" --predict-test
"$TRACK6_PY" tools/check_metrics.py "$TRACK6_DATA/stage1.zip" --split "$TRACK6_LOCAL/splits/identity-v1.json" \
  --run-dir "$TRACK6_OUT/replay" --output "$TRACK6_OUT/replay/verification.json"
cmp "$TRACK6_OUT/reid/prediction.json" "$TRACK6_OUT/replay/prediction.json"
cmp "$TRACK6_OUT/reid/validation-rankings.json" "$TRACK6_OUT/replay/validation-rankings.json"
```

实际运行的绝对命令已在运行清单保存，包括全部线性适配器对照。第二组先运行make_split并指定`--seed 20261002`，所有训练命令的split与seed都对应更新；输出使用独立目录，不能覆盖首组产物。正式可复现运行要求源代码先提交且工作区干净。

当前较优候选为算力机`local/benchmarks/dinov3-b16-reid12/prediction.json`，本地副本为`local/modern-results/benchmarks/dinov3-b16-reid12/prediction.json`。模型仍只在首组训练身份上训练，尚未全训练身份重训或提交排行榜。

## 下一步识别研究

跨模态R@1约20%，仍是主要瓶颈。保持这两组划分，优先审查错配样本，再做同预算的损失、翻转增强和末端解冻预算消融，避免把更长训练误当作结构收益。随后评估模态独立patch输入与共享主干、局部船体结构对齐和SAR像素预处理；DINOv3升级已实测，这些改进还未运行。灰度与深层特征融合可作为低成本对照。外部ReID权重、生成图像、全量重训及排行榜提交另行记录，不以论文名称或代码就绪代替本题识别证据。
