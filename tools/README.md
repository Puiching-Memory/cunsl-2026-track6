# 通用实验工具

这些工具提供数据完整性、环境验证和实验记录，不包含比赛模型，也不自动下载预训练权重。

## 环境

Python 3.11及以上。inspect_archive.py和run_job.py只用标准库；env_probe.py需要已有PyTorch。实际5090环境已经有可用PyTorch，不在系统环境覆盖安装旧版本。

## 命令

```bash
python tools/inspect_archive.py data/source.zip --verify-crc --output local/archive-audit.json
python tools/run_job.py --name env-smoke -- python tools/env_probe.py --output local/env-probe.json
```

run_job.py在local/runs中建立独立运行目录，记录命令、代码提交、开始/结束时间、退出码和日志。它不记录环境变量或登录凭据。命令自身不得包含密码或令牌。

state=completed只表示进程成功退出；metrics_validated始终先为false。赛题成绩需要单独的固定验证证据，不能根据任务进程退出码推断。

## 数据检查边界

ZIP审查报告涵盖摘要、CRC（显式启用）、文件目录、重名、空文件、加密和不安全路径。没有进行TIFF/JPG像素解码或标签质量审核；这些必须在真实数据接入后独立检查。工具永不解压或执行包中脚本。
