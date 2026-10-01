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

已在本题独立远程目录运行tools/run_job.py与tools/env_probe.py，FP32与BF16卷积反向梯度均有限，进程退出码为0。原始[环境检查记录](evidence/2026-10-01-env-probe.json)已取回；对应工具版本为ffb651e65ab9e21a3936a94444da9a03067e4875。此类小测试证明CUDA功能可用，不能证明比赛模型速度或训练效果。

实测依赖包括torchvision 0.27.1+cu130、numpy 2.4.6、Pillow 12.2.0。timm与safetensors未安装。本轮只检出审查过的DINOv2源代码并下载原始ViT-S/14权重，不依赖FastReID；实际骨干前向、适配器反向与检查点推理均通过，详见第8章。

建议布局：/root/autodl-tmp/cunsl-2026/track6/{project,data,local}，第5题对应独立track5目录。大文件留在数据盘，不改动现有其他项目。仓库里只保存可复现的非敏感环境记录。

接手时再次核验GPU为32607 MiB、2 MiB占用、利用率0%，无GPU计算进程；数据盘可用约50G。当前实测Python入口为/root/miniconda3/bin/python，非交互SSH的PATH没有python；所有复现命令使用此绝对入口。PyTorch/torchvision/numpy/Pillow版本与初次记录一致，系统环境未覆盖安装。实际模型用时和显存见[基准章](08-baselines.md)，不能使用env_probe结果替代。

GPU共享锁：/root/autodl-tmp/cunsl-2026/gpu.lock。run_job.py在持锁后检查计算进程；拒绝已占用的GPU，也不删除锁文件。代码与数据传输只触及track6目录，另一个赛题独立运行。
