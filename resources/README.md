# 外部资源登记

实际使用前，为每个代码库、数据集和预训练checkpoint记录名称、固定版本/提交、公开时间、原始URL、许可证、摘要、用途与对应实验。代码与权重分开登记。

第6题外部数据和预训练模型须在2026-09-01之前公开；具体权重许可逐一核对。

[原始DINOv2 ViT-S/14](dinov2-vits14.json)已审查2023年URL证据、具体模型卡Apache-2.0许可与固定代码LICENSE。88,283,115字节权重已下载并核验SHA-256，见[下载来源证据](../docs/evidence/2026-10-01-weight-source.json)。没有具体摘要时模型工具拒绝使用。本清单不授权DINOv2其他变体。权重文件保存在data/weights或算力机数据盘，Git仅保存公开URL与审查记录。

现代对照已使用[维护者转换的DINOv3-B/16通用版](dinov3-vitb16-timm.json)和[DINOv3-L/16卫星版](dinov3-vitl16-sat-timm.json)。两个精确文件于2025-09-17公开，未设置账号访问门槛，已完整下载并核对两端SHA-256。timm 1.0.20代码为Apache-2.0，权重继续受DINOv3 License约束；卫星版为光学RGB预训练。发布时间、初始/当前权重一致性、原始许可摘要与下载记录见[资源证据](../docs/evidence/2026-10-01-dinov3-resources.json)，实际使用及数值差异见[第9章](../docs/09-modern-models.md)。4c2394b实验报告绑定当时清单摘要，后续只增加完成状态说明，精确权重、代码与许可未改变。
