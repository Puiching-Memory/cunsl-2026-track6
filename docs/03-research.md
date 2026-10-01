# 初步文献与候选资源

本轮结论：第6题需要身份级验证与跨模态度量核对。通用图像检索、行人跨模态ReID提供研究参考，但尚未证明迁移到本题有效。不能把不同模态简单理解成彩色转灰度。

## 已核对的主要来源

| 来源 | 参考价值 | 未验证或不适用处 |
|---|---|---|
| [FastReID官方仓库](https://github.com/JDAI-CV/fast-reid)与[许可](https://github.com/JDAI-CV/fast-reid/blob/master/LICENSE) | 通用实例ReID平台，Apache-2.0代码；可参考实验记录、身份监督与检索评测组织 | 行人权重、数据集和默认采样设定不自动适用于本题；PyTorch当前版本兼容性未核验 |
| [DINOv2原始S/B/L/g模型卡](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) | 原始通用视觉表示支持最近邻图像检索，原始系列模型卡声明Apache-2.0 | 不是SAR专用表示；具体checkpoint必须核对日期和许可；仓库新增XRay/Cell等模型有不同许可，不能一概而论 |
| [BMDG，WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Alehdaghi_Bidirectional_Multi-Step_Domain_Generalization_for_Visible-Infrared_Person_Re-Identification_WACV_2025_paper.html) | 通过多步桥接域讨论可见光/红外模态差异 | 行人身体部位原型不能直接套用；光学/SAR的成像差异不同；这里只作为已有跨模态研究参考 |

原始DINOv2论文为2023年资源，但仍需选择具体公开权重，记录checkpoint URL、版本、许可证和SHA-256。当前没有下载或使用任何外部模型。

## 待验证的研究问题

1. 官方训练身份是否同时具有两种模态？每身份样本数量会怎样影响稳定的本地验证？
2. 图像按随机样本划分是否因身份重叠高估结果？先固定未见身份验证，且按三个方向独立报告R@1和mAP@10。
3. 通用视觉表示与监督度量学习的差异能否在相同数据/预算/输入尺寸下复现？先建立基准记录，再判断是否需要更复杂方法。
4. 排序后的模态过滤、重复处理和ID映射是否改变有效提交？验证工程正确性与模型准确率应分开。

这些都是研究假设。尚无本题的已验证提升、创新性或最佳框架结论。
