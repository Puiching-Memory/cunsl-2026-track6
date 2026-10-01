# 2026 第6题研究

赛题：多源卫星跨模态舰船重识别。仓库为私有研究工作区。

已取得并核验正式训练/初赛包，完成全量TIFF解码、标签与任务审查，以及固定身份隔离协议。数据读取、两种显式AP@10口径、严格提交校验和基准实验工具已就绪。模型实验与排行榜成绩分别记录，不能把格式校验当作准确率。

## 阅读顺序

1. [规则与指标](docs/01-contest.md)
2. [数据与获取状态](docs/02-data.md)
3. [文献与候选资源](docs/03-research.md)
4. [研究执行与验证计划](docs/04-plan.md)
5. [算力环境](docs/05-compute.md)
6. [研究日志](docs/06-status.md)
7. [固定验证协议](docs/07-validation.md)
8. [基准与复现](docs/08-baselines.md)

原始数据、身份标签、测试预测、权重和登录信息不进入Git。外部数据与预训练模型须分别登记发布时间、许可与用途。

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
