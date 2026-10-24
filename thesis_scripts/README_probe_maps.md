# 当前基线的物理量图导出

此工具只读取现有点云、Probe 缓存、相机和几何网络权重，不重新训练或拟合 Probe。
支持 original 和 linear_emissivity 模式；从缓存读取实际模式和参数。线性模型用法见 README_linear_emissivity.md。

## 在服务器复现 RoadBlock 结果

```bash
cd /home/liupengyu/physir-splat
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python \
  thesis_scripts/render_probe_maps.py \
  -m output/RoadBlock_2026-10-20_21-52-29/all \
  --iteration 30000 --split test
```

默认输出到模型目录的 `physical_fields/iteration_30000_test`。若已经存在，工具会停止，防止覆盖；复查时用 `--output-dir` 指定一个新目录。

- `--split test|train|all`：选择视角集合，默认 test。
- `--max-views 3`：在每个集合中均匀抽取最多 3 个视角；默认 0 导出全部。
- `--min-accumulation 0.5`：可见性掩膜阈值，范围 (0,1]。它不是可信度或不确定度。
- 不指定迭代时选择最新的配套点云与 Probe；不会把其他迭代的缓存套用到当前点云。

当前 RoadBlock 的 `iteration_30001` 缺少配套 Probe，因此本次使用配套完整的 `iteration_30000`，并加载同一迭代的 Geometry 权重。

## 输出及定义

| 字段 | 含义 |
| --- | --- |
| temperature_K | 当前 Probe 归一化 Planck LUT 对应温度，非辐射定标真值 |
| emissivity | 材料簇发射率（无量纲） |
| emission_proxy | 黑体发射归一化代理量 u |
| ambient_proxy | 全局环境代理量 a |
| self_emission_proxy | epsilon * u |
| reflected_ambient_proxy | (1-epsilon) * a |
| residual_proxy | 允许正负的模型补偿项 delta |
| q_phys_proxy | epsilon*u + (1-epsilon)*a + delta |
| q_target_proxy | 热 SH DC 转灰度后的拟合目标 |
| fit_abs_error_proxy | 每 Gaussian 的绝对拟合误差经可见权重平均；不是温度误差 |
| accumulation | 不透明度累积和，作为可见性参考 |
| valid_mask | accumulation 达到阈值的像素 |

每视角的 `fields.npz` 包含 float32 数值和掩膜，无效物理量像素为 NaN。PNG 是可视化，所有视角共享 `metadata.json` 中的色标。`overview.png` 含输入、原始基线热图渲染及主要物理量的色条。

```python
import numpy as np
d = np.load('test/000_FLIR5667/fields.npz')
temperature_K = d['temperature_K']
valid = d['valid_mask'].astype(bool)
```

`gaussian_fields.npz` 保留全部 Gaussian 的原始坐标、opacity 和物理属性。三维预览使用 canonical Gaussian 中心，坐标为任意尺度的 COLMAP 单位，并非物体内部体温场。

## 渲染与验证

使用零背景和已有 `override_color` 接口。每个属性先做可逆仿射编码，渲染后除以相同几何下的累积可见权重，再恢复数值；这避免背景偏置并保留负残差。渲染使用当前基线的视角几何变形。此版本对未实现的时间位姿/外观修正显式报错。

工具会检查点数、Probe 迭代、热 SH DC 对应关系、LUT 温度一致性、逐 Gaussian 及逐像素辐射项分解，记录 checkpoint 导出前后的 SHA256，确认源文件未变。源配置和 Probe 的哈希随结果保存在 `metadata.json`。

```bash
/home/liupengyu/miniconda3/envs/3dgs/bin/python -m unittest discover -s tests -p test_probe_maps.py
```

当前输入是处理后的伪彩图，LUT 温区为 260–360 K，波段为 7.5–13.5 um。输出的温度用于基线诊断，不能据此宣称 RoadBlock 的真实物体温度。其他辐射相关字段也是无量纲代理量，不是 W/(m^2 sr)。后续要验证绝对温度，应接入 FLIR 原始热数据及相机标定关系。

新缓存可额外导出 emissivity_ref、emissivity_slope 两种属性图及 material_parameters.npz 材料簇参数。emissivity 始终指拟合温度处的实际发射率。
