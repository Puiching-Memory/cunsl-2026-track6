# 2026 第6题研究

赛题：多源卫星跨模态舰船重识别。仓库为私有研究工作区。

已建好基础配置、数据完整性检查与实验记录工具，并在实际5090上完成FP32/BF16反向验证。授权规则、提交指标与初步文献已分章整理。数据接入与真实审查仍待完成；尚未训练模型，没有比赛成绩。

## 阅读顺序

1. [规则与指标](docs/01-contest.md)
2. [数据与获取状态](docs/02-data.md)
3. [文献与候选资源](docs/03-research.md)
4. [研究执行与验证计划](docs/04-plan.md)
5. [算力环境](docs/05-compute.md)
6. [研究日志](docs/06-status.md)

原始数据、身份标签、测试预测、权重和登录信息不进入Git。外部数据与预训练模型须分别登记发布时间、许可与用途。

## 只读检查工具

用法见[实验工具说明](tools/README.md)，实测信息见[环境验证证据](docs/evidence/README.md)。

```bash
python tools/inspect_archive.py /path/to/archive.zip --verify-crc --output local/archive-audit.json
python tools/env_probe.py --output local/env-probe.json
```

压缩包工具不解压、不运行数据内程序。小规模CUDA测试与压缩包完整性检查不能证明赛题准确率。
