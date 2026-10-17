# TransOSS 实验工具

训练、检查点推理、数据协议和独立评分各有独立入口。当前配置与结构见[项目说明](../docs/14-project-structure.md)，历史工具说明留在[研究归档](../docs/archive/legacy-tools.md)。

## 冻结成本的决策诊断

`study_triangle_matching.py --output-dir local/triangle-matching/新目录`固定K=5、T=.05，在48个验证场景对照三方最大能量/软汇总与固定代理，并核验完整检索；不使用测试图、不训练或生成提交。`study_triangle_capacity.py`固定读取`local/triangle-matching/v1`，用全新`capacity-v1`目录补测已占用副本排除、代理图像容量与null；两块坐标优化不保证全局三方最优。`analyze_triangle_matching.py`独立重建成本、候选、代理和标量指标，发布匿名科学图/证据；所有方法未达门槛，见[第41章](../docs/41-triangle-matching.md)。

`prepare_vlm_blind_pilot.py`固定生成18道验证集匿名原图板，输出`local/vlm-blind-pilot/v1`必须为空；身份答案与模型选择不交给代理。三个独立视觉代理回答锁定后，`score_vlm_blind_pilot.py`核验图板和协议摘要，评分强制选择、多数票及事先固定的弃权保守规则，生成匿名证据/科学图。这是验证能力试验，不调用在线模型预测测试集，也不生成提交，见[第40章](../docs/40-vlm-blind-pilot.md)。

`study_assignment_alternatives.py`在CPU上精确重放V5的40个旧共同身份场景，禁止每条原边后重求最优分配，生成有限候选集；不是完整k-best或后验抽样。固定输出`local/assignment-alternatives/v1`必须为空。`analyze_assignment_alternatives.py`核对候选基线与最好整套方案计数，用真值求候选边并集中的可行重组上界，并生成匿名证据/科学图。真值上界不可部署；均匀投票本轮无收益。两工具不训练、不使用GPU或测试像素、不生成比赛预测，见[第39章](../docs/39-inference-research.md)。

## 新封存与训练覆盖

`prepare_coverage_protocol.py`固定三分区与扩大门槛；`analyze_coverage_research.py --protocol-only`独立检查身份、近重复和像素哈希隔离。`train_coverage_model.py --mode foundation|specialist|full --output-dir local/coverage-research/对应组-v1`通过共享GPU锁运行，分支追加`--foundation local/coverage-research/foundation-v1/model.pt`；不读取封存像素，不加载旧身份微调权重。三个训练报告均完成后，`evaluate_coverage_sealed.py --foundation local/coverage-research/foundation-v1 --specialist local/coverage-research/specialist-v1 --full local/coverage-research/full-v1 --output-dir local/coverage-research/sealed-v1`锁定选择再一次评价封存。最后CPU运行`analyze_coverage_research.py`核验采样、早停、评分与源绑定，绘制科学图。新生产输出目录必须为空，审计只重建证据；参见[第37章](../docs/37-coverage-research.md)。

## V6固定融合与预测

`study_v6_transfer_gap.py`为官方结果后的CPU诊断，固定V5/V6/重叠银行与20场景，对比70%像素接受域、单观测图库、10%身份不重合；不训练、选权重或预测新分。规范输出`local/v6/transfer-gap-v2`须为空；v1排序输入保留，v2独立重排两端查询避免配对顺序泄露给平分规则。`analyze_v6_transfer_gap.py`核对源绑定、均值和两版顺序差并画科学图。复用验证且已看官方反馈，不能称新独立确认。见[第36章](../docs/36-v6-transfer-gap.md)。

`study_v6_fusion.py --output-dir local/v6/fusion-screen-new`只在CPU比较三个固定等权融合，验证其他处理域逐元素回退第五版；`study_pair_verifier.py --run-dir local/v6/fusion-screen-new --observation-groups`独立复算。`analyze_v6_fusion.py`绑定规范`fusion-screen-v3`生成匿名证据与科学图。所有生成目录必须为空，复跑改变目录须相应修改规范路径。

`build_v6_submission.py --output-dir local/submission-v6-native-overlap`通过`run_job.py --gpu-lock local/local-gpu/gpu.lock`运行，仅推理，不训练/上传；先重放两支全部验证银行与第五版公开预测。`audit_v6_submission.py`在CPU绑定规范输出、重建等权成本与像素桥接，使用已安装NetworkX独立匹配算法重建最终预测，发布匿名审计。不安装依赖。口径、模型取舍和本地/官方限制见[第35章](../docs/35-v6-integration.md)。

## 处理域与沿轴研究

第34章固定本机原生缓存和第五版头。prepare_axial_regions.py只读取训练/验证像素，--output-dir指定空目录，先查看区域叠加图；extract_axial_features.py以--region-dir读取该目录，通过GPU锁生成--output-dir缓存。train_domain_axial_head.py的--mode control|domain|axial、--seed和--output-dir控制三组，区域与特征路径可用--region-dir/--axial-cache指定。所有新输出目录必须为空。

study_pair_verifier.py --run-dir保存排名的实验目录独立评分；verify_domain_axial_models.py --root-dir读取其下control|domain|axial-20261009并在GPU新进程重放；analyze_domain_axial.py --root-dir绘制分组、成对区间和训练曲线。study_domain_complement.py只计算两个已选专家的固定平均，--output-dir允许新目录重放，不扫描权重。publish_domain_axial_evidence.py从规范local/axial-domain目录发布匿名统计SVG与证据，不导出原始图像或身份列表。完整限制和操作顺序见[研究记录](../docs/34-domain-axial-matching.md)。

## 学习式配对研究

这些研究入口固定绑定本机已有的原生TransOSS检查点和身份划分，不租GPU、不读取测试图、不自动生成或提交比赛预测。使用已有项目GPU版Python依赖；GPU任务必须通过共享锁运行。

```text
python tools/run_job.py --name pair-local-features --gpu-lock local/local-gpu/gpu.lock -- python tools/extract_pair_local_features.py --output-dir local/pair-verifier/native-local
python tools/run_job.py --name pair-local-train --gpu-lock local/local-gpu/gpu.lock -- python tools/train_local_pair_verifier.py --local-aux-weight 0.5 --output-dir local/pair-verifier/local-aux-v1
python tools/study_pair_verifier.py --run-dir local/pair-verifier/local-aux-v1
```

先前CLS头和训练特征是局部实验的依赖，保存在`local/pair-verifier/cls-v1`和`native-training`。局部特征提取同时验证CLS与已有本机缓存逐元素一致；新输出目录必须为空。复跑不同目录须同步修改研究脚本的固定缓存路径。

按最高development分数固定选择局部头，生成结合第四版桥接和第三版尾序的正式候选：

```bash
python tools/run_job.py --name local-pair-submission --gpu-lock local/local-gpu/gpu.lock -- python tools/build_local_pair_submission.py --output-dir local/submission-v5-local-pair
python tools/validate_submission.py data/stage1.zip local/submission-v5-local-pair/prediction.json
```

入口绑定当前检查点、数据、划分和本机缓存摘要，重放选中头验证矩阵、公开CLS、查询成本及第四版原生对照。只替换跨查询配对成本；O2O、常量SAR及第三版尾序保持，输出目录必须为空。该步骤约23秒、峰值allocated约809MiB，无优化器，不自动上传。

`--local-aux-weight 0`只训练合成分数的残差纠错；`0.5`额外要求局部分支独立区分训练正负对。两者采用相同节点、初始化种子、采样、批次和24轮更新上限，骨干及全局头均冻结。独立标量评分工具支持`--run-dir`，核对完整验证及40个原联合匹配条件模拟；这些都是复用验证数据，不能换算官方成绩。

动态难例对照使用`--hard-refresh-epochs 4 --mining-candidates 128 --mining-score full`或`local`，每4轮由当前整体/局部模型重选训练负池；默认刷新0保持固定余弦前32池。只评分两个训练方向余弦前128异身份候选并集，不读取验证标签挖难例，也不是全图库模型最难例。输出目录必须为空，结果见[第25章](../docs/25-pair-tail-hard-refresh.md)。

`python tools/study_pair_tail.py --output-dir local/pair-verifier/tail-study-new`为CPU研究，固定两既有头，比较前50/全图库尾序与等权融合；核验完整独立检索、原40场景及20%/50%身份不重合压力。固定当前项目缓存路径，不生成或上传预测。

## 数据与初始化

```bash
python tools/inspect_archive.py data/stage1.zip --verify-crc --output local/archive-audit.json
python tools/audit_data.py data/stage1.zip --output-dir local/data-audit --workers 4
python tools/make_split.py data/stage1.zip --audit-dir local/data-audit --output local/splits/identity-v1.json
python tools/fetch_weight.py resources/transoss-paired-vitb16.json data/weights/transoss-vit-b512-pre.pth
```

审查涵盖真实TIFF解码、标签/任务结构、精确与近重复；划分按身份隔离。ZIP工具不解压或执行包内程序。权重下载核对审查清单及精确SHA；固定作者代码为MIT，模型仓库声明CC-BY-4.0。完整来源见[资源登记](../resources/README.md)。

## 训练

```bash
python tools/run_job.py --name transoss-train --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- python tools/train_transoss.py data/stage1.zip --split local/splits/identity-v1.json --weights data/weights/transoss-vit-b512-pre.pth --output-dir local/transoss/run1
```

默认读取 configs/transoss.toml，`--config`选择其他文件，CLI覆盖对应字段；`--no-augment`关闭翻转。配置拼写、数值和轮次边界在加载模型之前检查。默认复用已提交路线：256方形输入、16身份/批、全骨干SGD、CE和无margin欧氏batch-hard triplet、200轮上限、至少30轮、耐心20、最小累计改善0.001。更改轮数上限也改变余弦学习率周期。

训练入口只读取训练身份做梯度更新和验证身份做评分。第0轮和每轮验证均参与选择，任何严格最佳都原子保存，包括BN缓冲区；相同分数保留较早轮次。结束后恢复最佳模型并复算分数。输出 reid.pt、training-history.json、validation-history.json、features.npz、validation-rankings.json 和 report.json，要求新的空目录。

作者位置网格16×8明确插值到本题16×16；全部双模态骨干参数严格加载，禁止缺失SAR入口时静默复制。缺少物理尺寸/GSD时关闭尺度嵌入。本题采样、输入和增强属于比赛适配，不宣称完整论文复现。

## 评估与生成提交文件

```bash
python tools/run_job.py --name transoss-predict --gpu-lock /root/autodl-tmp/cunsl-2026/gpu.lock -- python tools/predict_transoss.py data/stage1.zip --split local/splits/identity-v1.json --checkpoint local/transoss/run1/reid.pt --output-dir local/transoss/predict1 --predict-test
python tools/check_metrics.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/transoss/predict1 --output local/transoss/predict1/verification.json
python tools/validate_submission.py data/stage1.zip local/transoss/predict1/prediction.json
```

推理入口没有优化器，不需要再次读取预训练文件；使用 `weights_only=True`恢复完整模型，检查架构、初始化摘要、数据和身份划分绑定，然后验证选中模型的本地分数。省略 `--predict-test`仅评估验证集；加上该参数才提取正式Query/Gallery并生成 prediction.json。保留 validation_then_test 分批顺序，与已提交v1完整检查点兼容。原检查点按原字节复制到新目录，摘要保持一致。

独立评分以标量逐位累加复算两种AP分母、R@1和0.45/0.45/0.10加权，另核对特征及完整检查点摘要。格式校验不证明隐藏集效果；本地AP口径未获得官方实现确认。工具不上传文件。

## 运行记录与检查

本地Windows GPU依赖使用独立`local/cuda-deps`目录，版本见[依赖清单](../requirements-local-gpu.txt)。在仓库根目录使用当前本机Python入口并持锁检查：

```powershell
$track6Python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = (Resolve-Path 'local/cuda-deps').Path
& $track6Python tools/run_job.py --name local-env-probe --gpu-lock local/local-gpu/gpu.lock -- $track6Python tools/env_probe.py --output local/local-gpu/env-probe.json
```

该入口依赖本机已有的Python运行时；换机需替换`$track6Python`。不要把旧的`local/refactor-deps`放在GPU依赖之前，否则会导入CPU版Torch。上述检查不训练模型，后续GPU实验仍使用同一锁。

```bash
python tools/env_probe.py --output local/env-probe.json
python -m unittest discover -s tests -v
```

run_job.py为每次运行记录命令、代码版本/工作区状态、日志与退出码。共享GPU锁在Linux使用flock，在Windows使用msvcrt字节锁；取得后检查已有GPU进程，占用时拒绝启动，不终止其他工作。`--gpu-wait-seconds`限定等锁时间。日志记录完成不等于评分已验证。CPU小模型回归可用 `--device cpu`，正式GPU成绩必须在目标环境复核。本地CUDA及相关环境由用户安装，助手不自动安装或恢复未完成的依赖下载。

## 错配与表示诊断

```bash
python tools/analyze_retrieval.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/transoss/run1 --output-dir local/transoss/errors
python tools/diagnose_shortlist.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/transoss/run1 --output local/transoss/shortlist.json
python tools/build_pair_atlas.py data/stage1.zip --splits local/splits/identity-v1.json --output-dir local/transoss/atlas --census
```

诊断核对保存产物并只使用有训练标签的验证身份；候选集oracle仅是诊断上界，不能当作模型分数。图像观察册、逐图错误、身份映射和原始像素必须留在 Git 外。

rerank_features.py保留为模型无关的CPU候选偏好实验，默认不用于当前TransOSS提交。历史重排序本地改善但官方下降，任何新策略都需要独立核验。

## 困难负样本与局部匹配

```bash
python tools/train_transoss.py data/stage1.zip --split local/splits/identity-v1.json --weights data/weights/transoss-vit-b512-pre.pth --config configs/transoss-hard-negative.toml --output-dir local/transoss/hard
python tools/study_local_matching.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/transoss/hard --output-dir local/transoss/hard-local
python tools/check_metrics.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/transoss/hard-local --output local/transoss/hard-local/verification.json
```

GPU调用仍须用上文run_job.py和共同锁包裹。实验参数、局部匹配的歧义及结果边界见[研究记录](../docs/16-hard-negatives-local-matching.md)。默认训练的新目标权重均为0；hard-negative配置仅加异模态top4余弦对比，hard-local配置另外加局部对比。局部研究入口不生成测试预测或上传，固定前50候选、局部权重0.2、O2O不改；局部缓存SHA也由独立评分核验。

## DSMCL对照入口

仍用train_transoss.py，选择`--config configs/transoss-dsmcl-{control,spatial,frequency,joint}.toml`之一。四组统一至少80轮，保留200轮学习率周期；额外目标默认关闭，不和上轮局部方案自动叠加。频域分支为训练辅助，最佳轮次参数随reid.pt保存；predict_transoss.py只恢复原native特征，不创建该分支。CPU公式、真实模型CPU梯度及本地GPU四图前后向已核验，完整GPU训练对照尚未执行；原文未给出的实现选择、单对退化及待核验项见[第17章](../docs/17-dsmcl.md)。

## 查询集合联合重排序

`rerank_cohort.py`是研究入口，只读取公开Query/Gallery像素与绑定特征报告；不读取身份标签。默认`conservative`要求桥接目标在两个完整图库中都距最佳余弦不超过0.03，随后允许部分配对；`complete`采用更强的一对一假设。常量查询保留独立排名，光学缺失图库副本可通过同像素O2O查询的首选桥接。共同身份假设未经验证，两种模式都可能损害分数。

通过`--baseline-prediction`和`--baseline-evidence`共同指定已验证的第三版排名，核对预测和模型摘要，再只提升选中的首位、保留后续顺序。省略时以输入特征的独立排名为基线。源ZIP、特征、查询覆盖、目标模态与输出格式均核验；不会上传文件。`input_checkpoint_replay_verified`原样记录输入报告，格式通过不代表严格模型重放或隐藏评分通过。示例、候选及压力结果见[第20章](../docs/20-public-cohort.md)。

固定第四版的研究特征提取源已保存为`tools/extract_cohort_features.py`，在仓库根目录通过`run_job.py`持GPU锁运行，可用`--output-dir`指定新的缓存目录。该入口固定第三版检查点和数据路径、严格加载权重，明确记录跨环境验证分数差异；不修改生产重放保护，不做训练或提交。

## 四次反馈与公开指纹再审查

`study_public_clues.py --output-dir local/dsmcl/public-clues-new`复查9,489张官方图的D4/灰度/黑边摘要、近重复实际像素误差、有限编号生成假设和简单描述的身份隔离验证表现，并比较四份已提交预测。入口固定当前项目内数据、审查、划分与提交路径，先检查原审查的ZIP摘要和覆盖；输出逐图材料仅放local。训练标签只用于训练图质量及留出校准，没有隐藏身份输入或预测生成，见[第21章](../docs/21-public-clues.md)。


## 局部配对与集合级训练

`train_local_pair_verifier.py --output-dir 新目录 --local-aux-weight 0.5`复现固定难例局部头训练；`--hard-refresh-epochs 4 --mining-score full|local`启用训练银行内动态难例。`--set-objective dual|sinkhorn --set-size 32`使用不同训练身份的完整双向矩阵和随机SAR置换；集合目标不与动态刷新同时启用，Sinkhorn归一化次数默认为10。所有GPU入口用run_job.py持共同锁，原模型架构和公开预测入口保持。

`study_pair_verifier.py --run-dir 训练目录`独立复算排名、报告条件bootstrap和40个既有集合场景控制；没有公开测试推理或提交。新训练会保存训练源码和集合目标模块源码快照，SHA与report绑定。协议、本轮对照结果及验证复用限制见[第26章](../docs/26-set-pairing.md)。

## 显著收益筛选与骨干配对微调

`extract_pair_local_features.py --grid 8 --output-dir 新目录`提取64节点；原默认grid4保持。`train_local_pair_verifier.py --pair-model cross --node-cache 缓存目录 --seeds 20261009 --output-dir 新目录 --local-aux-weight 0.5`运行两轮跨图交互头，原local模型不变。两种子结果不稳定，见[第27章](../docs/27-significant-gain-screen.md)。

`extract_pair_local_features.py --suffix-cache --output-dir 新目录`保存前3层float32完整tokens，并逐图核对suffix恢复的CLS和节点。`train_pair_suffix.py --mode joint|frozen --epochs 6 --output-dir 新目录`固定使用已绑定的native-prefix3缓存、原第三版骨干及第五版头，从第4层继续；所有GPU入口仍用run_job.py持共同锁。两组独立验证使用study_pair_verifier.py；仅先导对照，不读取公开测试或生成预测，见[第28章](../docs/28-pair-suffix-finetuning.md)。

`study_relation_pairing.py --output-dir 新目录`用第五版成本和同模态关系图进行40场景固定系数筛选；`--graph-only`移除单对目标，只保留原排列初始化。FAQ式近似算法提高目标不等于提高身份准确率；两种方式均无大幅信号，见[第29章](../docs/29-relation-graph-pairing.md)。

## 观测困难对照与科学分析图

`train_observation_pairs.py --mode augmented|unaugmented --output-dir 新目录`固定完整骨干、两光学/一SAR的匹配4轮对照；`train_real_observation_head.py --mode balanced|natural --output-dir 新目录`固定缓存与第五版局部头，做8轮真实多观测采样对照。GPU入口仍须run_job.py持共同锁。两轮均已结束，没有显著候选，见[第30章](../docs/30-observation-invariance.md)、[第31章](../docs/31-real-observation-exposure.md)。

`study_pair_verifier.py --observation-groups --verification-output 新报告路径`增加困难27/源93分组统计，不覆盖历史verification；`study_observation_proxy.py --output-dir 新目录 --diagnose-selection`加入验证标签诊断，标签选出的第二观测不是可提交模型。

`plot_observation_research.py --output-dir local/research-figures/日期`只用已绑定验证缓存和完成的实验汇总绘制错误结构、条件身份簇区间、损失/验证轨迹与真实错配案例。复现全部40个正确配对控制，PNG/SVG与匿名统计写local，带像素/真实ID的案例不能写tracked目录。图与推断边界见[第32章](../docs/32-visual-observation-analysis.md)。

`study_visual_interventions.py --output-dir local/visual-interventions/新目录 --prepare-only`准备冻结像素干预和私有区域掩码；先检查ROI叠加图，再在同目录去掉--prepare-only、通过run_job.py持GPU锁推理。`plot_visual_interventions.py --run-dir 实验目录 --prepare-only`只画区域QA；完整绘图还需要人口分组诊断与report。区域是处理内容代理，不是船体标注；没有训练/测试/提交。已有v4的完整核验、黑边处理混杂和下一项假设见[第33章](../docs/33-pixel-interventions.md)。

GPU诊断后先运行`study_processing_population.py --run-dir 实验目录`产生训练/验证处理分组和三个既有模型的描述对照，再运行`verify_visual_interventions.py --run-dir 实验目录`独立复算保存矩阵、身份差值、像素变化及源快照。最后去掉plot_visual_interventions.py的--prepare-only绘制完整图。真实像素只输出local目录，所有指标均为复用验证条件诊断。

`study_training_reference.py --output-dir 新目录`用2,819个训练身份拟合固定CLS参照映射/后验和一次等权融合；`study_geometry_likelihood.py --output-dir 新目录`用训练有效区域拟合固定形态似然，再融合V5。两者用`study_pair_verifier.py --run-dir 实验目录`独立核验完整检索和原40条件集合。`study_pixel_profiles.py --output-dir 新目录`只做部分有效区域的细像素回归诊断，不生成整体提交。`plot_reference_geometry_screen.py`绘制已完成固定目录的匿名图，见[第34章](../docs/34-domain-axial-matching.md)。所有结果均为复用验证，不是官方预测或独立确认。

`audit_hoss_training_pixels.py --output-dir 新目录`需要已核对HOSS目录索引/归档，默认64张公开训练光学，只检索本题训练光学候选。`audit_hoss_training_exact.py`校验完整HOSS归档MD5，用固定双线性缩放/黑色居中算子逐像素比对全部574张HOSS训练光学和本题train_images；不读取双方验证/测试像素。私有源图和训练ID对应只保存local，不能把来源审查结果当成未知测试身份预测。

`study_hoss_preprocessing.py --output-dir 新目录`固定训练候选恢复的p0.5/p99.5量化流程，检索全部有效HOSS训练SAR与本题train_images的精确像素，绘制原强度/量化/空间预算科学图；负值源图另记排除。`study_hoss_input.py --output-dir 新目录`通过run_job.py共同GPU锁，在本题复用验证上精确重放V5，然后评估像素门控裁切放大、SAR平方根增强及组合，冻结权重，无训练步。用`study_pair_verifier.py --run-dir 实验目录 --observation-groups`独立评分和40共享集合核验。`analyze_hoss_input.py`分析固定input-probes-v1产物，新增相对V5的错配计数/身份条件区间与匿名科学图；其私有输入QA只使用已确认训练对。没有外部验证/测试图、预测或提交，不能把输入分布偏移的失败当作重新训练也无效。

`train_hoss_expert.py --mode original|packed --output-dir 新目录`通过run_job.py共同GPU锁，固定462个像素门控TRAIN身份、真实2光学/2SAR采样，全骨干/classifier/全局局部头共同续训，默认8轮。每轮保存银行，用开发20个120/26联合场景选轮；报告20个80/18仅最终评价，均属复用验证。两组用study_pair_verifier.py独立评分；analyze_hoss_expert.py分析旧original-joint-v3/packed-joint-v3。追加`--epochs 64 --min-epochs 32 --patience 24 --min-delta 0.001 --save-training-state`按用户授权检查更长训练，实际最高开发分选模型，累计最小提升控制耐心。training-state.pt含当前模型、两个优化器、采样和Torch/CUDA随机状态；本工具尚无自动恢复入口，不能把读取参数称为连续恢复。`analyze_hoss_earlystop.py --run-dir 实验目录`在独立study_pair_verifier核验后检查前8/18轮精确复现、停止逻辑、模型重载和最终收益，绘制各项损失与开发曲线。没有官方新分数或提交。

`study_hoss_batch_competition.py --output-dir 新目录`仅用TRAIN特征与CPU分析同一462身份池，固定100身份/200光学查询、每身份两SAR，计算随机不同身份集合排除所有更高分负例的精确组合概率；同时检查最佳/最弱正例。`analyze_hoss_batch_competition.py --run-dir 实验目录`绘制竞争规模曲线。已见训练身份的固定排名事件，不是实际CE/Hungarian或验证提分；不能据此宣称大批次有效。

上述工具加`--save-score-bank`可保存私有TRAIN分数银行。`study_assignment_training.py --bank-dir 银行目录 --output-dir 新目录`对100已见TRAIN身份各一个固定真实观测，比较随机/困难8和32集合及完整100集合的独立行/Hungarian错误与精确结构化间隔梯度；只有CPU机制诊断，不训练模型。`analyze_assignment_training.py --run-dir 诊断目录`绘制明确区分合成自由分数更新与真实TRAIN排名的四面板图。assignment_margin_loss仅支持唯一排列，不接受每身份两个正观测；不能从该工具获得验证提分。

`train_hoss_expert.py --mode original --training-method pair-control|pair-meta --epochs 24 --min-epochs 12 --patience 8 --save-training-state --output-dir 新目录`运行身份互斥A/B对照；仅配对目标，BN分类颈冻结且不执行，每轮29正式更新。pair-meta是明确的一阶虚拟内步，默认10倍分组学习率，不含内步Hessian/Jacobian。两个实验独立study_pair_verifier核验后，`analyze_identity_meta.py --control 对照目录 --meta 元训练目录 --output-dir 新目录`检查采样/权重还原/最佳银行及TRAIN B变化并画图。已完成固定一阶配置无显著收益，未扩种子。

`train_hoss_expert.py --mode original --training-method native-sam --sam-radius 0.05 --epochs 24 --min-epochs 12 --patience 8 --save-training-state --output-dir 新目录`保留完整原目标，每轮58次正式更新，用标准SAM两次前后向；通过run_job.py持共同GPU锁。Dropout随机状态重放，BN只累计第一遍。首先执行单轮rho=0，再用`verify_sam_zero.py --run-dir 零半径目录`检查历史native首轮损失、采样、指标、完整成本矩阵及BN/AdamW步数精确一致。正式SAM完成独立study_pair_verifier核验后，`analyze_native_sam.py --run-dir SAM目录 --zero-dir 零半径目录`核验最佳模型/停止规则/完整训练状态，按同早停规则重放历史native对照，发布科学图和匿名证据。XSAM尚未实现；训练B改善或参数扰动损失不能当作泛化保证，均无官方新分数。

本轮native-sam-v1已完成16轮、选8，单种子固定rho0.05：报告20/原40联合相对原训练+0.011480/+0.007286，困难27准确率+4.07个百分点，完整检索和开发联合下降。保留局部信号，但没有一致大幅收益；不追加半径搜索/第二种子或提交。执行协议、历史对照限制和科学图见第34章。

按用户要求暂停SAM，筛选更大收益的训练机制。`study_full_gallery_competition.py --output-dir 新目录`通过共同GPU锁，对固定200个TRAIN光学观测比较原462身份和完整2,819身份的SAR竞争，保存私有分数银行及前缀重放差值；`analyze_full_gallery_competition.py --run-dir 诊断目录`独立核对TRAIN归属、身份分组、来源摘要并重算排名事件，发布匿名图与证据。没有参数更新或验证提分，域外无新增首选错误的结论仅限该固定TRAIN观测。

`train_hoss_expert.py --mode original --patch-stride 8 --epochs 24 --min-epochs 12 --patience 8 --save-training-state --output-dir 新目录`保留256原图与16px卷积核，把主干token图从16×16改为31×31；原位置参数沿实际patch中心插值，16节点使用不重叠奇数网格分区。原stride16先精确重放V5，再独立保存stride8未训练银行；新配置仅支持original/native完整目标。不是完整TransReID/JPM复现，增加的是采样位置与计算量。所有GPU训练仍由run_job.py持共同锁；生产推理默认stride16。正式结果经study_pair_verifier独立评分后，`analyze_overlap_training.py --run-dir 正式目录`核验历史对照/实际采样/早停/最佳银行/完整优化器状态，区分未训练采样效应与训练增量，并画科学图。显著收益门槛在正式结果前固定：完整检索或原40联合相对原尺度全模型专家至少+0.03，其他项与报告组退步不超过0.005。

`study_overlap_spatial_budget.py --output-dir 新目录`仅用全部TRAIN光学像素统计步长16/8的含非黑内容窗口，通过积分图计数和逐像素窗口独立核验。它不测注意力、船体分割或身份信息，也不更新模型；匿名图和统计与私有逐图明细分别保存。见第34章。

`finalize_overlap_training.py --producer-manifest 本次GPU任务清单 --run-dir 正式目录 --output-dir 新私有目录`等待指定训练进程正常退出，再顺序执行独立评分与analyze_overlap_training核验，成功后才标记metrics_validated。最多等待3小时，不中断训练、不自动扩GPU种子/生成提交；最终科学图仍需人工查看。`study_hoss_extra_observations.py --output-dir 新目录`只统计既有精确TRAIN来源映射身份的public TRAIN成员名，候选明细私有。`publish_training_method_evidence.py`绘制原论文消融，不能与本地成绩混为一谈。

`study_mcrl_compatibility.py --output-dir 新目录`仅检查全部TRAIN SAR的固定MOS 5/100分位拉伸、独立排序分位复核及真实双模态观测计数，绑定划分/门控/源码。逐图明细私有，匿名汇总写docs/evidence；不导入作者代码、不训练模型或调整分位。它不能用于宣称MCRL无效，只核查原输入和多观测假设是否可直接迁移。

同一工具追加已绑定原生TRAIN CLS缓存的身份均值/方差项分析，使用全部真实观测，不模拟训练Dropout或翻转。`publish_mcrl_compatibility.py --run-dir 检查目录`独立核对TRAIN图像/身份/模态数量与保存项聚合，发布匿名科学图。项值比例不是梯度贡献，不能以小方差项直接证明没有作用。

`train_hoss_expert.py --mode original --cmal-mean-weight 0|2 --epochs 24 --min-epochs 12 --patience 8 --save-training-state --output-dir 新目录`检验归一化类别均值对齐的单独作用：原CLS逐观测归一化，按身份等权模态均值L2距离，权重2固定、不平方，不含方差/去噪/生成。仅支持original/native/stride16，权重0保持原算术；正权重首步独立核对两个stem与第一层辅助梯度。两支都重新从同一初始化训练，共同采样摘要相等，分别早停；控制还需要精确重放历史native银行/轨迹。

`study_pair_verifier.py --run-dir 实验目录 --observation-groups`独立评分后，`analyze_cmal_training.py --audit-dir 单分支目录`核对执行、损失分解、来源、所选银行和最新优化器/RNG；`--control-dir 控制目录 --run-dir 对齐目录`追加两支配对、历史控制和固定+.03扩大门槛核验，发布科学图。`finalize_cmal_training.py --control-manifest 控制任务清单 --output-dir 新私有目录`等待已启动的固定控制，先评分/核验，再顺序执行权重2、独立评分/对照图和TRAIN机制诊断；所有GPU子任务持共同锁，结果不触发第二种子或官方提交，图仍需人工查看。

`study_cmal_geometry.py --control-dir 控制目录 --run-dir 对齐目录 --output-dir 新私有目录`在所选模型固定后，重编码同462 TRAIN身份池的全部真实观测，比较无增强/Dropout的均值距离与余弦正负间隔、首选率；按身份等权，均值项由NumPy和Torch损失分别核验。这里只测TRAIN机制，不是held-out泛化，不参与选轮；原始特征/身份留local，公开匿名CDF图与聚合统计。

`audit_cmal_geometry.py --run-dir TRAIN机制目录`用另一CPU实现重建像素门控的TRAIN身份池和全部观测，核对真实身份划分/显式模态、两个模型元数据、特征摘要；逐身份/逐查询标量重算均值距离、半径/夹角及最好正例/最强异身份负例、间隔和首选率，验证8项汇总和3个逐身份数组。固定权重2对照已完成：完整检索−0.012230、原40联合−0.004216，未通过门槛，停止扩大；TRAIN距离缩小同时正负间隔压缩，不宣称整套MCRL无效。

`study_hoss_scale_fingerprint.py --limit 151 --include-optical --integer-lattice --output-dir 新目录`只在已确认来源TRAIN上检验盲尺寸估计；resampling_lattice.py使用全部像素列的uint8整数链可行性，无原尺寸/身份输入。`study_scale_fusion.py --output-dir 新目录`只以盲TRAIN高度拟合固定Gaussian似然，V5+LLR/20，验证拒绝图保持未知，无系数搜索。`--training-estimates 全训练估计JSON --validation-height-cache 原融合目录`可复用已绑定验证像素估计、换用全TRAIN校准；不接受验证身份拟合。`study_scale_training_census.py --output-dir 新目录`对全部TRAIN做同一门控和盲尺寸检查。`study_lattice_extension.py --output-dir 新目录`单独检查源TRAIN中H225..255扩展的精度，不能把额外接受直接当作正确。analyze_scale_fusion.py发布已完成fusion-v1的匿名机制/覆盖/收益图，公开证据注明旧整数运行报告的继承元数据错误；原图/ID/成本留local，无外部测试图或提交。

`extend_scale_census.py --scope train|validation --input-dir 原224目录 --output-dir 新目录`先绑定原估计/门控/数据摘要，再固定将未知接受图的候选增加225..248；上限由已知TRAIN源的254/255错误边界校准，不能称独立普适置信度。`study_scale_fusion.py --height-cap 248 --training-estimates 全TRAIN248估计JSON --validation-height-cache 验证248目录 --output-dir 新目录`做固定扩展先验；`--expert-score-bank 原尺度专家成本npz`额外生成expert-control，并做一次固定相加，不重新挑轮次。用study_pair_verifier.py分别独立评分；analyze_scale_census.py读取固定四个已完成配置、逐元素重建9份成本并发布匿名校准/覆盖/收益图。该轮均未合格，停止扩大，不生成提交。
