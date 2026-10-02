# 2026 第6题研究

赛题：多源卫星跨模态舰船重识别。仓库为私有研究工作区。

已核验正式训练/初赛包，完成9,489张TIFF解码、标签与任务审查，以及身份隔离协议。已接入来源、日期和许可经过审查的DINOv3通用/卫星权重，完成冻结特征、配对适配与图像级微调对照。进一步阅读直接相关论文及作者实现、审查真实错配，完成分模态入口与结构损失对照：独立光学/SAR入口在两组划分的加权分为0.35548/0.32832，超过原微调0.33659/0.31150；完整三方向指标、耗时和恢复核验见[领域消融实测](docs/11-domain-ablations.md)。预测完整覆盖267个Query，尚未提交排行榜，AP@10正式程序仍待确认。

## 阅读顺序

1. [规则与指标](docs/01-contest.md)
2. [数据与获取状态](docs/02-data.md)
3. [文献与候选资源](docs/03-research.md)
4. [研究执行与验证计划](docs/04-plan.md)
5. [算力环境](docs/05-compute.md)
6. [研究日志](docs/06-status.md)
7. [固定验证协议](docs/07-validation.md)
8. [基准与复现](docs/08-baselines.md)
9. [新模型、识别效果与复现](docs/09-modern-models.md)
10. [光学/SAR领域机制与错误研究](docs/10-domain-research.md)
11. [分模态入口与结构消融实测](docs/11-domain-ablations.md)
12. [表示研究：可见证据、缺失和身份](docs/12-representation-study.md)

原始数据、身份标签、测试预测、权重和登录信息不进入Git。外部数据与预训练模型须分别登记发布时间、许可与用途。

2026-10-02，算力机下线期间完成[表示研究](docs/12-representation-study.md)：核对12组真实训练样例和2,435张光学图的呈现差异，整理信息缺失的数学反例及候选表示，并补齐旧模型候选集诊断。该轮没有新增识别成绩；原图和观察册留在Git外。

## 数据与实验工具

用法见[实验工具说明](tools/README.md)，实测信息见[环境验证证据](docs/evidence/README.md)。

```bash
python tools/inspect_archive.py /path/to/archive.zip --verify-crc --output local/archive-audit.json
python tools/env_probe.py --output local/env-probe.json
python tools/audit_data.py data/stage1.zip --output-dir local/data-audit
python tools/make_split.py data/stage1.zip --audit-dir local/data-audit --output local/splits/identity-v1.json
python tools/validate_submission.py data/stage1.zip local/benchmarks/gray32/prediction.json
python -m unittest discover -s tests -v
```

默认直接读取ZIP，不解压、不执行包内程序。原始ZIP、逐图审查、身份划分、特征、预测和权重只保存在data/、local/及算力机数据盘。仓库证据仅保留不包含身份列表的汇总。
