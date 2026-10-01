# 规则、提交与指标

核对日期：2026-10-01。以后续主办方公告与正式评分程序为准。

## 任务与指标

给定Query图像，从Gallery检索相同身份的图像。包括光学→SAR（O2S）、SAR→光学（S2O）、光学→光学（O2O）。训练、初赛和复赛身份互不重叠，目标是未见身份的检索泛化。

每方向得分S=(R@1+mAP@10)/2，总分=0.45×S_O2S+0.45×S_S2O+0.10×S_O2O。mAP@10在相关项超过10时的分母等细节仍需正式评分脚本确认，不能自行假定全排序mAP与该指标等价。

## 提交

上传UTF-8的prediction.json，顶层为对象：键是task.json中的query_id；值为有序数组，包含10个不同、有效、目标模态正确的Gallery image_id。

必须恰好覆盖全部Query。不能写入相似度、路径、模态、注释或额外字段。缺Query、额外Query、数量错误、重复ID、无效ID或模态错误直接无效。

每日最多3次提交，必须保存最高分对应的完整版本。尚未进行任何排行榜提交。

## 资源规则

- 初赛训练与推理使用自有环境。
- 外部数据和预训练模型须在2026-09-01前公开，注明名称、版本、URL、许可和使用方式；不使用付费独占或非公开资源。
- 禁止AIS、航迹、地理位置、成像时间等未公开辅助信息，也不能用在线人工服务或非公开远程模型直接生成测试预测。
- 公开预训练的代码许可与具体权重许可须分别检查；“可下载”不等于符合竞赛条件。
- 当前读取规则没有看到明确参数上限，不能据此保证没有其他资源限制。

复现需源代码、权重、依赖锁定、预处理、训练与推理命令、随机种子、外部资源清单和说明。

## 赛程

10月4日开始提交，10月14日初赛结束选10队；10月15–17日复赛选5队；10月23日深圳决赛；之后复现审核。详细赛程与顶部“结束11.01”的口径需要持续留意。

来源：[赛题](https://www.heywhale.com/org/cunsl/competition/area/6aba33ce6e5beff2fca53add/content/1)、[提交](https://www.heywhale.com/org/cunsl/competition/area/6aba33ce6e5beff2fca53add/content/3)、[评测](https://www.heywhale.com/org/cunsl/competition/area/6aba33ce6e5beff2fca53add/content/4)、[规则](https://www.heywhale.com/org/cunsl/competition/area/6aba33ce6e5beff2fca53add/content/5)。
