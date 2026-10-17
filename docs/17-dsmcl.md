# DSMCL：论文公式适配与 CPU 验证

2026-10-07。用户选择本轮只实现和本地验证，不启动 GPU。没有新的训练后验证成绩、测试预测、正式提交或算力费用。现有官方最好仍为第三版 **0.5144077657**；本页的 CPU 检查不能代替提分实验。

## 原文与复现边界

[Dual-Space Modality Consistency Learning for Universal Cross-Modal Re-Identification](https://arxiv.org/abs/2608.06943)于2026-08-07提交；本次核对固定 [v1 原文](https://arxiv.org/html/2608.06943v1) III-A–D、IV-B及Table III。作者在TransOSS上增加空间身份分布一致性SMCL和频域身份对比FDCL。HOSS上的O2S R@1为35.4%→40.0%，S2O为31.3%→38.8%，这不是本题分数。

按论文标题、DSMCL和GitHub检索，并查看论文页面与第一作者公开仓库，尚未找到可核实的作者DSMCL实现。这里是**论文公式的比赛适配**，不声称官方代码复现或完整复现论文结果。未导入新权重、外部数据或HOSS身份训练权重。原文HTML和其摘要留存于local/dsmcl；匿名证据见[evidence/2026-10-07-dsmcl-implementation.json](evidence/2026-10-07-dsmcl-implementation.json)。

## 实际实现

`track6/dsmcl.py`实现两个训练目标，`training.py`负责调度、优化和保存；固定vendor源码未修改。

**SMCL**依照Eq.10，对每个同时有光学/SAR观测的训练身份，计算原始全局CLS的均值和总体标准差，损失为两种模态均值差与标准差差的平方L2之和，再对身份取平均。未配对身份跳过并计数。

当前采样始终每身份一光学、一SAR。两侧总体标准差均为零，所以损失严格退化为 **配对CLS平方距离**；当前训练中单对比例为100%，并写入逐轮报告。数据约94%原始身份只有单对图像，不能把翻转视图当独立观测，也不能宣称已检验完整分布方差对齐。代码支持多观测公式，但本轮未改变采样。

**FDCL**依照Eq.12–21，从第6个block输出获取无CLS的16×16空间patch网格。每个通道在H/W上做二维FFT、`log(abs(FFT)+epsilon)`、空间fftshift、标准化，再经可学习sigmoid掩码，展平并用于余弦对比。正例是同身份异模态，负例是所有不同身份观测，包含同模态和异模态；排除自身与同身份同模态。分子求正例指数和，分母求正例和全部负例指数和，不采用top4挖掘。温度0.2。缺正例或负例的anchor不参与均值，但仍可作为其他anchor的负例；无有效anchor时报错。

这处理的是中间特征空间频谱，不是SAR物理复相位。掩码覆盖整个频谱，没有强制径向高频分区；学到什么频带需要后续分析，不能预先认定其关注高频。

论文未明确的细节采用以下固定选择，均未用验证分数调参：

| 细节 | 本次选择 |
|---|---|
| 中层位置 | 第6个block的patch输出 |
| 标准化范围 | 每张图的C/H/W一起求均值和总体标准差 |
| G掩码网络 | 先对通道均值，Conv3×3(1→8)、GELU、Conv1×1(8→1)、sigmoid；跨通道共享 |
| 数值保护 | epsilon=1e-6；FFT和损失FP32，CUDA骨干仍按原入口BF16 |
| 损失最终权重 | SMCL=0.001，FDCL=0.1，初始候选值而非作者给定值 |
| SMCL渐进长度 | 第20轮后线性20轮增至最终权重，自行明确的选择 |
| FDCL渐进长度 | 第40轮后线性40轮，按论文描述 |

权重公式为`target * clamp((epoch-start)/ramp, 0, 1)`。SMCL第20轮为0、第21轮首次启用、第40轮满权重；FDCL第40轮为0、第41轮首次启用、第80轮满权重。

掩码参数纳入原SGD，和骨干一起检查梯度是否有限；初始化有独立CPU随机流，避免新网络初始化消耗既有骨干/增强随机流。中层hook在成功或失败后都清理，可与现有局部匹配取不同层或同一层。训练辅助分支按同一个最佳轮次原子保存在`reid.pt`的DSMCL元数据中；原native模型state_dict仍严格兼容。推理只恢复原骨干、BN和分类头，不建立频域网络、不优化参数，依然使用原始全局CLS排序。没有实现精确恢复优化器训练轨迹。

## 下一次训练的四组固定对照

| 配置 | SMCL | FDCL |
|---|---:|---:|
| configs/transoss-dsmcl-control.toml | 0 | 0 |
| configs/transoss-dsmcl-spatial.toml | 0.001 | 0 |
| configs/transoss-dsmcl-frequency.toml | 0 | 0.1 |
| configs/transoss-dsmcl-joint.toml | 0.001 | 0.1 |

四组都维持相同公开场景配对初始化、seed20261001、200轮余弦周期、16身份/批、学习率0.0005、耐心20、min_delta0.001，**min_epochs统一80**，让FDCL完整渐进后才允许早停。原始min_epochs30基线不适合作为这一延迟调度的唯一对照，因此保留匹配的80轮控制组。仍逐轮验证和保存严格最佳，第0轮也可胜出，最佳轮次不必达到80。延长最低轮数增加训练成本；这些配置尚未执行。

四组均不叠加前一轮困难负样本、局部损失或局部重排序，以便区分DSMCL两个模块的作用。主入口默认DSMCL权重为0，继续使用原transoss.toml，未改变已提交模型默认训练设置。

未来GPU预算获批后，沿用共同GPU锁，用新空目录运行；下面只是命令示例，本轮未执行：

```bash
python tools/run_job.py --name dsmcl-control --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- python tools/train_transoss.py data/stage1.zip --split local/splits/identity-v1.json --weights data/weights/transoss-vit-b512-pre.pth --config configs/transoss-dsmcl-control.toml --output-dir local/transoss/dsmcl-control
python tools/run_job.py --name dsmcl-joint --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- python tools/train_transoss.py data/stage1.zip --split local/splits/identity-v1.json --weights data/weights/transoss-vit-b512-pre.pth --config configs/transoss-dsmcl-joint.toml --output-dir local/transoss/dsmcl-joint
```

不能只看本地微小涨分。当前留出身份反复用于方法/轮次选择，后续先冻结上述参数，再在独立身份划分上从公开初始化分别重训匹配基线和候选，并记录两个跨模态方向、O2O、训练开销和种子。把旧模型直接评估在它可能见过的另一划分上不构成独立确认。相关本地/官方偏差和bootstrap限制见[第16章](16-hard-negatives-local-matching.md)。

## 本轮 CPU 核验

新增11项DSMCL回归检查：单对严格退化、总体标准差项、常量有限梯度、未配对身份处理、标量逐项复算多正例/全负例、观测顺序和身份重命名不变、已知二维频率峰/DC位置、零频谱梯度、掩码可更新、随机流隔离、双层hook清理、配置/调度，以及合成训练→最佳辅助状态恢复→无优化器native推理→独立指标复算。调度测试首次对浮点十进制做精确比较失败，改为数值容差；损失实现未因此改变。

完整CPU套件65项：62通过，3项Linux锁检查按平台跳过。另用现有第53轮完整TransOSS权重和四张**训练**图执行真实前向/反向，没有优化器更新。patch形状为4×256×768，频谱展平4×196608；骨干和掩码梯度有限，掩码梯度非零，同层/不同层hook对CLS均逐值不变，句柄全部移除。本批次SMCL原值348.3447、FDCL0.9680，乘初始权重约0.3483/0.0968；仅用于检查量级，不能当作训练效果。原权重文件未修改。

日志摘要、真实CPU检查和代码/配置摘要见匿名证据。CUDA下FFT、卷积确定性、BF16混合训练的行为及峰值显存尚未测试；正式训练前仍需目标GPU小批次核验。当前结论仅是实现可运行与兼容性检查通过，没有DSMCL比赛提分证据。
