# 算力环境

2026-10-08用户已安装系统CUDA，随后授权重新安装GPU版PyTorch。实测Toolkit为13.4.2、nvcc为13.4.92，RTX 4060 Ti仍为16380 MiB；独立NVRTC编译和CUDA kernel运行通过，0–15平方和为1240。初次NVRTC检查因本进程DLL搜索路径失败，显式预载安装目录内builtins DLL后通过，不需改系统PATH。PyTorch安装使用独立`local/cuda-deps`目录，保留原CPU依赖，不修改用户安装的CUDA或驱动。

已安装PyTorch 2.12.1+cu130，与之前云端模型实验版本一致；使用其自带CUDA 13.0运行时。1,926,431,198字节Windows wheel经官方SHA256核验，使用官方源及南京大学公开镜像下载；[固定依赖清单](../requirements-local-gpu.txt)和[本地持锁入口](../tools/README.md)已保存。首次pip安装遇到沙箱临时目录权限错误，将TMP/TEMP设置到项目目录后成功。

CUDA张量、FP32/BF16反向均通过。第三版完整检查点在四张训练图上严格恢复，TransOSS和DSMCL空间/频域分支前后向通过，骨干与掩码梯度有限，hook不改变全局CLS且已移除；峰值allocated/reserved为1193.46/1246 MiB，脚本计时10.34秒（不含Python导入）。此检查零优化器更新，不构成训练吞吐、留出或官方成绩。新PyTorch环境下完整74项回归71通过、3项Linux专属检查跳过，结束后无GPU计算进程。详见[匿名环境证据](evidence/2026-10-08-local-gpu.json)。

2026-10-07本地实测：NVIDIA GeForce RTX 4060 Ti，显存16380 MiB，驱动616.56；检查时无GPU计算进程。现有项目依赖为PyTorch 2.6.0+cpu，尚未验证本地CUDA模型运行。用户要求自行安装CUDA及相关环境：助手已停止本次GPU版PyTorch下载，未安装该包，也未改动系统CUDA或驱动。残留下载文件不完整，不可作为可用依赖。后续不得自动恢复下载或安装；待用户告知环境就绪后，再核验指定Python环境并运行轻量GPU实验。目前继续使用既有CPU环境。

2026-10-02状态：用户已将算力机下线，本轮未重新连接。以下环境与GPU状态是此前实测记录；再次运行前需确认实例、环境、现有进程及共享锁。本轮[表示研究](12-representation-study.md)全部在本地CPU完成。

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

实测依赖包括torchvision 0.27.1+cu130、numpy 2.4.6、Pillow 12.2.0。首轮系统无timm与safetensors，只检出审查过的DINOv2源代码并下载原始ViT-S/14权重；不依赖FastReID，实际骨干前向、适配器反向与检查点推理均通过，详见第8章。随后将timm 1.0.20、safetensors 0.6.2、huggingface_hub 0.35.1与PyYAML 6.0.2以--no-deps隔离安装到数据盘track6/local/dinov3-deps，运行设置PYTHONPATH，现有PyTorch/CUDA未覆盖。

建议布局：/root/autodl-tmp/cunsl-2026/track6/{project,data,local}，第5题对应独立track5目录。大文件留在数据盘，不改动现有其他项目。仓库里只保存可复现的非敏感环境记录。

接手时再次核验GPU为32607 MiB、2 MiB占用、利用率0%，无GPU计算进程；数据盘可用约50G。当前实测Python入口为/root/miniconda3/bin/python，非交互SSH的PATH没有python；所有复现命令使用此绝对入口。PyTorch/torchvision/numpy/Pillow版本与初次记录一致，系统环境未覆盖安装。实际模型用时和显存见[基准章](08-baselines.md)，不能使用env_probe结果替代。

GPU共享锁：/root/autodl-tmp/cunsl-2026/gpu.lock。run_job.py在持锁后检查计算进程；拒绝已占用的GPU，也不删除锁文件。代码与数据传输只触及track6目录，另一个赛题独立运行。

2026-10-01新模型轮次已完成16次持锁模型运行，B/16与SAT-L/16真实前向、B/16图像监督反向和两组检查点恢复均通过。微调12轮分别109.35/110.43秒，首组PyTorch allocated/reserved峰值约3.30/3.89 GiB，见[新模型章](09-modern-models.md)。结束后无GPU计算进程、flock检查锁可取得，数据盘可用47G；该状态为当时观察，后续启动仍须检查。
