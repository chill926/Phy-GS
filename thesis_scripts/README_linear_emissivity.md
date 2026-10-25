# 温度相关发射率 Probe：第一版

直接接入现有训练中的 Probe 调用；没有添加不确定度模块。

## 模型与参数

```text
epsilon_i(T_i) = eps0[material_id_i] + k * (T_i - T_ref)
q_phys_i = epsilon_i * u_i + (1-epsilon_i) * ambient + delta_i
T_i = PlanckLUT_inverse(u_i)
```

每轮优化都先从 u 得到 T，再计算 epsilon(T)，梯度通过 LUT 的分段插值传播。温度不是拟合结束后的发射率后处理。

- `--phys_probe_mode original`：默认值，沿用原材料常数发射率拟合函数。
- `--phys_probe_mode linear_emissivity`：使用线性温度相关模型。
- `--phys_probe_temp_ref 300`：参考温度，单位 K，必须处于拟合温区内。
- `--phys_probe_emissivity_slope 0.0001`：固定斜率 k，单位 K^-1。默认 0，用于零斜率控制。
- `--phys_probe_e_init`：T_ref 处 eps0 的初值和先验中心，原默认 0.95。
- `--phys_decouple_lambda_e_prior`：继续使用现有参考发射率先验权重；在线性模式下必须为正。

**第一版各材料簇分别拟合 eps0，共用一个配置给定的 k；没有学习 k。** 目前聚类标签没有稳定的语义材料身份，不能把不同 Probe 运行中的第 m 簇简单当成同一种实测材料。因此暂不提供按随机簇编号指定不同材料斜率的入口。

eps0 使用可行区间内的 sigmoid 参数化。可行区间由 Tmin/Tmax 两端的发射率约束共同确定，因此整个温区都保证 e_min <= epsilon(T) <= e_max，而非依靠硬 clamp。过大的斜率、不可行的先验中心、越界的 T_ref 或 NaN 参数会明确报错。

零斜率可以退回原模式的数值结果；原模式拟合函数的函数体保持不变。

## 在新训练中启用

沿用原训练命令的数据划分和其他参数，只增加：

```bash
--phys_probe_mode linear_emissivity \
--phys_probe_temp_ref 300 \
--phys_probe_emissivity_slope 0.0001
```

上述非零斜率只是链路验证示例，不是 RoadBlock 的材料测量值。用于正式实验的系数应有依据，或明确作为控制变量扫描。默认 original 模式不会改变现有训练的数值路径。

## 复用已训练点云，只重拟合 Probe

不需要重新训练 3DGS。下面命令会创建一个新的实验目录，保存新 Probe；点云和原几何权重通过符号链接复用，原实验必须保留。它不会覆盖原实验的 Probe、点云或权重。

```bash
cd /home/liupengyu/physir-splat
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python \
  thesis_scripts/refit_probe.py \
  -m output/RoadBlock_2026-10-20_21-52-29/all \
  --iteration 30000 \
  --output-model output/RoadBlock_linear_probe_v1_k1e-4 \
  --phys_probe_mode linear_emissivity \
  --phys_probe_temp_ref 300 \
  --phys_probe_emissivity_slope 0.0001
```

该示例目录已在本次验证中创建；复查时应换一个新的 `--output-model`。

导出新模式的图：

```bash
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python \
  thesis_scripts/render_probe_maps.py \
  -m output/RoadBlock_linear_probe_v1_k1e-4 \
  --iteration 30000 --split test --max-views 3
```

导图工具从缓存读取实际模式、温区、参考温度和系数，不会把线性模型标成常数发射率基线。旧缓存没有这些字段时仍按 original 解释。当前仅支持 normalized_planck_lut 温度模型；以后接入辐射定标，应增加相应的校验器。

## 缓存与输出

保留原有 temperature、emissivity、emission、ambient、delta、q_phys 等字段。新增：

- `probe_metadata`：模式、温区、波段、T_ref、k、参数化方式及未定标标识。
- `material_ids`：本次 Probe 的簇标签。
- `cluster_eps0`、`cluster_k`、`cluster_counts`：材料簇系数与大小。
- `emissivity_ref`、`emissivity_slope`：逐 Gaussian 的参考发射率和固定斜率。

导出图使用的 emissivity 是 epsilon(T_i)，不是 eps0。额外提供参考发射率图、斜率图，以及 `material_parameters.npz`。点数或顺序改变后需要重新拟合 Probe；新字段不会成为旧 checkpoint 的加载必需项。

## 验证


测试覆盖正负斜率的全温区边界、通过 T 的梯度、零斜率退回原模型、非法参数、新旧缓存和清空生命周期。RoadBlock 已完成非零斜率的真实 GPU Probe 拟合和数值关系校验。

目前温度仍是归一化 LUT 定义的估计量。加入线性模型不等于解决单波段温度—发射率歧义，也不构成温度精度提升的证据。
