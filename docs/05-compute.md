# 算力环境

2026-10-01已通过用户授权的SSH连接核验AutoDL现有实例。登录地址、密码和连接令牌不写入仓库。

| 项目 | 实测/观察 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，单卡 |
| nvidia-smi显存总量 | 32607 MiB |
| 驱动 | 595.71.05 |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| CUDA运行时 | 13.0 |
| GPU计算能力 | 12.0；PyTorch编译架构包含sm_120 |
| CUDA实际运算 | 0–15平方和得到1240，已通过 |
| 系统盘 | 30G，首次观察可用约30G |
| 数据盘 | /root/autodl-tmp，50G，首次观察几乎为空 |

AutoDL登录横幅显示25核CPU、92GB内存；os.cpu_count()返回208是宿主可见逻辑CPU数，不能据此宣称独占208核。

后续运行tools/env_probe.py记录FP32/BF16卷积反向及环境信息。此类小测试证明CUDA功能可用，不能证明比赛模型速度或训练效果。

建议布局：/root/autodl-tmp/cunsl-2026/track6/{project,data,local}，第5题对应独立track5目录。大文件留在数据盘，不改动现有其他项目。仓库里只保存可复现的非敏感环境记录。
