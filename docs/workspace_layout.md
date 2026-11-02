# 工作区整理

项目：服务器 `/home/liupengyu/physir-splat`。

## 生产代码位置

- `tools/probe_full_sh_attributes.py`：当前主流程实际调用的属性拟合入口，保留原优化实现。
- `utils/probe_dn_utils.py`：从旧DN实验脚本逐字提取的8个共用定义，只有DN响应、冻结映射、配置读取、采样与文件工具，无旧实验入口。
- `utils/attribute_probe_pipeline.py`：主训练及渲染的属性流程，已更新导入。
- `tools/fit_flir_display_mapping.py`、`tools/segment_sam3_materials.py`、`tools/temporal_residual_corrector.py`：保留当前依赖及材料处理工具。

根目录仅保留 `train.py`、`render.py`、`metrics.py` 三个Python入口。训练和主渲染命令不变。

## 清理范围

删除独立测试/诊断脚本、旧DN实验入口、大气测试入口、旧Probe对照工具，以及40个旧实验输出目录和一份旧日志。保留最新梯度报告与预览，清掉其64份大型中间数组。服务器删除文件逻辑大小约10.58 GB，当前output占用约4.1 GiB。

保留的正式训练结果：当前完整SH版本、旧B、DropGaussian历史基线、SAM3历史基线及原始RoadBlock基线；同时保留现用DN映射、已绑定的属性拟合结果与小型代码备份。数据集、SAM3材料数据和当前物理场缓存均保留。

本地删除临时测试工作副本、匹配器临时权重/源码和旧测试输出，逻辑大小约2.61 GB；保留 `work/Phy-GS` Git仓库副本、当前结果与SAM3说明。

历史报告中的测试命令属于当时记录，相应脚本现已按要求清理，不应再直接执行。

## 验证

- 保留的源码、模型、DN映射和输出共1673个文件的哈希在清理前后一致。
- 8个共用定义的AST与旧代码完全一致。
- 旧根目录Probe文件删除后，train/render入口均可正常加载。
- 新属性拟合入口通过当前30000轮模型的dry-run。
- 主属性流程校验全部原始输入和映射通过；当前属性缓存与点云、IR/RGB几何权重哈希匹配。
- 仅整理位置、导入和文档；没有修改训练损失、学习率、SH配置或Probe数值公式，没有启动训练或重新拟合属性。

完整删除及保留清单见同目录两份JSON；服务器记录位于 `docs/cleanup_20261101.json`。

## 明确保留

用户要求保留 `graduate` 原项目；后续清理不得删除该目录或其内容。本次删除的 `work/graduate_mapping_audit_20261031` 是临时核查副本，不是 `graduate` 原项目。
