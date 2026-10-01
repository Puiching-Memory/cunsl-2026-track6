# 2026 第6题研究

赛题：多源卫星跨模态舰船重识别。仓库为私有研究工作区。

当前完成授权规则读取、提交与指标核对、初步文献和算力环境接入。初赛包下载与真实数据审查仍待完成；尚未训练模型，没有比赛成绩。

## 阅读顺序

1. [规则与指标](docs/01-contest.md)
2. [数据与获取状态](docs/02-data.md)
3. [文献与候选资源](docs/03-research.md)
4. [研究执行与验证计划](docs/04-plan.md)
5. [算力环境](docs/05-compute.md)
6. [研究日志](docs/06-status.md)

原始数据、身份标签、测试预测、权重和登录信息不进入Git。外部数据与预训练模型须分别登记发布时间、许可与用途。

## 只读检查工具

```bash
python tools/inspect_archive.py /path/to/archive.zip --verify-crc --output local/archive-audit.json
python tools/env_probe.py --output local/env-probe.json
```

压缩包工具不解压、不运行数据内程序。小规模CUDA测试与压缩包完整性检查不能证明赛题准确率。
