# 研究阶段README历史快照

2026-10-18整理参赛总结前保留的原始README。下文中的“当前”“最新”“尚未提交”均属于各研究阶段的历史表述，不能用来判断最终仓库状态。正文保留，相对链接按归档位置调整。最新总结与正式成绩见[根README](../../README.md)。

---

# CUNSL 2026 第6题 · TransOSS

2026-10-11按用户建议暂停新增模型训练，转向冻结特征的图像关系与联合决策。新CPU诊断精确复现40个V5配对：基线82.52%，候选均匀投票无变化；真值辅助挑整套方案85.21%，在候选边并集中求可行重组90.77%。后两者是标签诊断，不是可部署方案或官方预测；下一重点是第三观测能否提供辨别局部替代的额外证据。见[非训练研究](../../docs/39-inference-research.md)。

2026-10-11完成跨领域文献研究：已确认完整检索改善不能稳定转化为困难联合配对收益，局部不可对应证据和候选竞争信息仍是待检验机制。下一方案分别测试弱监督局部部分匹配与集合条件边判别，包含未见身份式训练错误、参数匹配/上下文打乱对照、外折复核和显著收益停止线；未启动新训练或提交。见[完整研究方案](../../docs/38-literature-research-plan.md)。

2026-10-10继续探索发现原裁剪尺寸痕迹：整数插值约束在396张已确认训练源中接受226张，高度全部精确；过近256会误接受缩小图。全部本题TRAIN校准及固定上限248扩展后，尺寸先验联合收益仍仅+0.002692；与全模型专家组合仅比专家另增+0.000585，报告组仍下降，未跨显著门槛。全模型原尺度专家虽让困难配对47.59%→57.04%，联合仅+0.017803且报告下降，也未合格。停止这些固定配置的扩大，不新增提交，保留官方第五版0.7185908014；科学图、独立评分、覆盖与来源选择偏差见[第34章](../../docs/34-domain-axial-matching.md)。这些是复用本地验证分，源尺寸机制不是隐藏测试标签。

2026-10-10恢复HOSS训练来源的处理规则：245张光学与151张SAR训练图逐字节复现，后者均与光学身份对应一致。SAR为p0.5/p99.5截断、线性8位四舍五入、双线性缩放及黑边。已确认光学宽度中位44像素，约3个ViT patch；但本地GPU固定权重对照中，直接裁切放大、SAR平方根增强与组合均降低完整验证和40组联合配对得分，停止扩大这些推理改法，保留第五版官方0.7185908014。来源审查只读双方训练像素，输入实验使用本题复用验证，不读取外部验证/测试图或生成提交。科学图、独立评分及推断边界见[第34章](../../docs/34-domain-axial-matching.md)。

处理域/沿轴研究已完成三组24轮匹配对照：原4×4专家完整检索仅+0.006869，沿轴8×2仅+0.002015，联合匹配分别+0.001982/+0.003710；固定等权平均仍仅+0.007792条件收益。未达到显著门槛，不扩种子、不提交，保留第五版官方0.7185908014。分组收益方向相反且区间跨零，科学图、区域覆盖、模型重放和完整记录见[处理域与沿轴匹配](../../docs/34-domain-axial-matching.md)。

像素干预新增线索：训练/验证的多光学图全部属于黑边处理域；同域单光学也较难。第五版局部头对其他背景CLS检索50.92%→61.10%，黑边多光学37.88%→33.33%，需要研究处理域适配和船体长轴token聚合。全部765光学/718 SAR的10项冻结干预已核验；直接灰度化、改填充与输入裁切没有显著收益，无新提交。图、区域质量限制及复现见[像素干预研究](../../docs/33-pixel-interventions.md)。

2026-10-09新增科学分析图：身份簇条件区间、正负配对间隔、四组训练/验证曲线及真实验证错配案例。困难组身份选择与缺副本效应混杂；不能继续把47.59%与93.59%的差异仅归因于新视角。完整骨干几何增强和真实多光学平衡采样均选回第0轮，自然采样条件收益仅+0.001691，没有显著候选；保留第五版官方0.7185908014。图、配对决策损益和复现边界见[可视化分析](../../docs/32-visual-observation-analysis.md)。

显著收益筛选已完成：64节点交互头第一种子检索+0.043258，第二种子退步；骨干配对微调完整验证仅+0.000558、条件集合+0.004088，报告组下降；整组关系图固定融合仅+0.002814，纯图目标退步。三条路线均未形成可靠大幅收益，停止扩大，保留官方第五版0.7185908014。见[筛选记录](../../docs/27-significant-gain-screen.md)、[配对微调](../../docs/28-pair-suffix-finetuning.md)和[关系图](../../docs/29-relation-graph-pairing.md)。

最新集合级训练对照：双向CE完整验证0.650276/0.646986，条件集合收益仅+0.002171/+0.005856；Sinkhorn完整验证0.595805/0.604171，两种子集合模拟也退步。保留第五版官方模型；四模型重放、两组逐轮采样一致及13项配对测试通过，零付费GPU/新提交，见[研究记录](../../docs/26-set-pairing.md)。

2026-10-09继续对照：学习式尾序仅约+0.0024条件收益；两个既有头等权融合独立检索0.656129。两种动态难例训练组均没有稳定改善集合配对，独立检索与联合匹配出现相反变化，尚未生成新提交。集合级训练已完成首轮，下一步重点为骨干模态/结构约束，见[新实验](../../docs/25-pair-tail-hard-refresh.md)。

最新正式成绩：第五版局部交互配对结合查询集合联合匹配，Public **0.7185908014**，比第四版0.6699152061提高**0.0486756（约7.27%）**。2026-10-09用户提供已评审截图，见[正式提交证据](../../docs/evidence/2026-10-08-fifth-submission.json)。

本地新增冻结TransOSS的学习式CLS配对头：两个种子完整验证0.558770→0.572452/0.574639，判别头确认组小幅同向、增量区间跨零；联合匹配条件模拟仅+0.004923/+0.011104，未生成提交。见[配对判别器实验](../../docs/23-pair-verifier.md)。

局部交互进一步实测：额外要求局部分支独立区分训练正负对，两个种子完整验证达到**0.643278/0.645383**，相对各自CLS头均约+0.071；联合匹配条件模拟0.791032→约0.835。按development选择的第二种子已与第四版组合，23:36作为第五次正式提交，官方得分0.7185908014；本地数字与官方成绩分别记录。见[局部配对实验](../../docs/24-local-pair-verifier.md)和[提交回执](../../docs/evidence/2026-10-08-fifth-submission.json)。

三路并行调查定位到27张缺光学图库副本查询的配对困难：条件模拟中贡献约56%误配。固定两检查点融合、距离成本和第二观测候选，40次模拟0.791032→0.812437；已生成并核验，尚未提交，这不是预计官方分数。编号/包装及精确局部源图审查未恢复新身份规则，见[并行调查](../../docs/22-parallel-investigation.md)。

多源卫星跨模态舰船重识别。当前研究路线集中于作者 TransOSS：光学/SAR场景配对初始化、全骨干身份微调、逐轮验证和最佳检查点选择。

第三版独立检索基线：第53轮检查点，本地身份隔离验证 **0.56050**，官方 Public **0.5144077657**。官方比首版提高17.1%；验证集参与模型选择，两种分数分别记录。实测见[TransOSS研究](../../docs/13-transoss.md)及[官方提交证据](../../docs/evidence/2026-10-07-third-submission.json)。

最新训练候选：跨模态困难负样本与局部对比联合训练，第49轮最佳加局部匹配，本地 **0.57956**；尚未提交或独立划分确认。本轮费用7.13元、实例已关机，完整对照见[困难负样本与局部匹配](../../docs/16-hard-negatives-local-matching.md)。默认配置继续复现已提交模型。

DSMCL已完成论文公式适配、CPU回归与真实TransOSS梯度检查，四组匹配对照配置已准备。2026-10-08本地4060 Ti的四图GPU前后向也通过，完整训练对照尚未启动，未取得新的官方分数。单对采样下空间方差项退化，细节和边界见[DSMCL实现记录](../../docs/17-dsmcl.md)。

重读数据后发现跨方向Query/Gallery重复结构；在“两个120查询集合共享同一批身份”的**条件模拟**中，第三版独立检索0.533→联合匹配0.800。该身份条件尚未证实，数字不是官方分数；优先核实任务结构，见[任务耦合研究](../../docs/18-task-structure.md)。

官方五页规则已重新取得，未保证共同查询身份。新增部分匹配CPU压力实验：50%身份不重合时，完整分配0.445低于独立0.541，双向间隔控制0.555；完全不重合时拒绝控制仍有损失。当前只作研究，未修改默认推理，见[规则核对与压力实验](../../docs/19-cohort-robustness.md)。

2026-10-08实际公开特征出现查询源副本集合富集：两方向独立首选覆盖74.17%/54.62%，同大小随机图库子集13.16%/10.42%。新图库竞争门控在共享身份条件模拟下+0.064，完全不重合时仍−0.015；不是预计官方提分。以已提交第三版排名为固定基线，保守与完整匹配两份候选已生成、核验；完整版已作为第四版提交，保守版尚未提交。跨环境BF16重放差异及600案例压力实验见[实际查询研究](../../docs/20-public-cohort.md)。

基于四次官方提交再审查公开线索：发现12张Query与训练近重复，但训练47对近重复的身份标签全部不同，不能直接传递身份。旋转/翻转、灰度、黑边与简单编号流程没有恢复新身份规则；全图库硬配对和查询尾部限制对照不足以替代第四版。详见[公开线索再审查](../../docs/21-public-clues.md)。

## 开始使用

Python 3.11以上；训练环境需已有可用PyTorch，CPU可运行小规模回归测试。数据依赖见 requirements-data.txt；作者骨干只依赖PyTorch，不需要模型库或Hub运行时。

```bash
python -m pip install -r requirements-data.txt
python tools/audit_data.py data/stage1.zip --output-dir local/data-audit
python tools/make_split.py data/stage1.zip --audit-dir local/data-audit --output local/splits/identity-v1.json
python tools/fetch_weight.py resources/transoss-paired-vitb16.json data/weights/transoss-vit-b512-pre.pth
python tools/train_transoss.py data/stage1.zip --split local/splits/identity-v1.json --weights data/weights/transoss-vit-b512-pre.pth --output-dir local/transoss/run1
python tools/predict_transoss.py data/stage1.zip --split local/splits/identity-v1.json --checkpoint local/transoss/run1/reid.pt --output-dir local/transoss/predict1 --predict-test
python tools/check_metrics.py data/stage1.zip --split local/splits/identity-v1.json --run-dir local/transoss/predict1 --output local/transoss/predict1/verification.json
python tools/validate_submission.py data/stage1.zip local/transoss/predict1/prediction.json
python -m unittest discover -s tests -v
```

GPU训练和推理均通过 `tools/run_job.py --gpu-lock` 包裹；完整用法见[工具说明](../../tools/README.md)。生成预测与上传分开执行，工具不租机或自动提交。

## 项目结构

| 位置 | 职责 |
|---|---|
| configs/transoss.toml | 实际读取的训练配置；CLI可覆盖 |
| track6/transoss.py、vendor/ | 模型适配、严格加载及固定作者代码/许可 |
| track6/training.py、selection.py | 身份训练、逐轮验证、早停和最佳选择 |
| track6/matching.py、local_study.py | 异模态困难负例、局部损失和固定候选局部匹配验证 |
| track6/dsmcl.py | 训练期空间分布一致性、二维patch频谱和可学习掩码对比 |
| track6/inference.py、checkpoints.py、runtime.py | 完整模型恢复、数据绑定与特征提取 |
| track6/data.py、audit.py、protocol.py | 正式数据审查与身份隔离 |
| track6/metrics.py、experiments.py、submission.py | 检索、报告和提交结构校验 |
| tools/ | 薄命令入口及数据、评分、诊断工具 |
| tests/ | 数据协议、配置、训练/恢复流程及模型回归 |
| resources/ | 当前TransOSS公开资源登记 |
| docs/ | 研究记录与匿名证据，包括历史方法 |

## 研究记录

- [规则与指标](../../docs/01-contest.md)、[数据](../../docs/02-data.md)、[验证协议](../../docs/07-validation.md)
- [TransOSS实测与提分分析](../../docs/13-transoss.md)、[项目重构与后续研究](../../docs/14-project-structure.md)
- [TransOSS优化诊断、新论文与实验顺序](../../docs/15-transoss-optimization.md)
- [困难负样本与局部匹配实验](../../docs/16-hard-negatives-local-matching.md)
- [DSMCL公式适配、四组对照与CPU核验](../../docs/17-dsmcl.md)
- [视觉子代理的18题验证盲测](../../docs/40-vlm-blind-pilot.md)：跨模态直接投票未获收益，同模态纠错与误改抵消。
- [三方联合匹配、代理容量及失败机制](../../docs/41-triangle-matching.md)：冻结缓存的多代理规则与结构对照均未达到扩大门槛。
- [测试公开结构、条件联合匹配与边界](../../docs/18-task-structure.md)
- [研究日志](../../docs/06-status.md)、[验证证据](../../docs/evidence/README.md)
- 历史DINO研究保留于第3、8、9、10、11、12章及[历史配置/资源](../../docs/archive/dino-settings.md)；相关实现、运行入口和专用测试已删除。历史命令仅用于理解实验记录。

默认直接读取ZIP，不执行包内程序。原始数据、身份标签、检查点、特征、预测、登录信息及逐图分析留在 Git 忽略的 data/、local/ 中；文档只保存匿名汇总。历史实验产物保留为研究证据，不是当前模型入口。
