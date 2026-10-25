# 可学习斜率的温度相关发射率 Probe

在项目根目录中执行。默认模式为 `learnable_linear_emissivity`；`original` 和固定斜率 `linear_emissivity` 模式继续可用。当前材料默认值与 SAM3 掩膜生成方式见 [材料配置说明](../data/materials/README.md)。

## 参数化与优化

每个材料簇拟合两个端点发射率：

```text
e_L = e_min + (e_max-e_min) * sigmoid(c - d/2)
e_H = e_min + (e_max-e_min) * sigmoid(c + d/2)
k = (e_H-e_L)/(T_max-T_min)
eps0 = e_L + k*(T_ref-T_min)
epsilon_i = eps0[material_id_i] + k[material_id_i]*(T_i-T_ref)
T_i = PlanckLUT_inverse(sigmoid(u_raw_i))
q_i = epsilon_i*u_i + (1-epsilon_i)*ambient + delta_i
```

`c`、`d` 均为每簇一个可学习参数。因此 eps0 与 k 都能改变，两者受同一全温区边界约束。T_ref 固定。正、负斜率均可学习。端点约束保证整个 [T_min,T_max] 的 epsilon 都在 [e_min,e_max]，不依赖硬截断。

默认从 k=0 开始；SAM3 材料分支中，金属 eps0 初值为0.30、范围0.20～0.40，非金属及未知类别维持0.95、0.85～0.99。128步拟合在第32步释放参考发射率与残差，第64步释放斜率，阶段设置来自材料JSON。旧聚类分支仍使用 `phys_probe_slope_warmup_fraction` 控制斜率预热。上述阶段都是每次独立 Probe 内部的步数，不是完整3DGS训练的比例；没有跨Probe的参数继承。

新增参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--phys_probe_lambda_slope` | 0.001 | 斜率正则权重；0 可用于消融 |
| `--phys_probe_slope_warmup_fraction` | 0.25 | 每次 Probe 内部冻结斜率的比例，[0,1) |
| `--phys_probe_slope_lr_scale` | 0.25 | d 的学习率相对原 Probe 学习率的倍数 |

斜率正则是 `lambda * mean((k*(T_max-T_min)/(e_max-e_min))**2)`，即归一化端点发射率差的平方。上述默认值是实验起点，不是通过 RoadBlock 真值标定的最优参数。继续使用原有 eps0、环境先验、残差惩罚和温度平滑。

在新模式中 `--phys_probe_emissivity_slope` 必须为 0，仅表示零初始斜率；训练后的 k 从缓存读取。若需非零固定斜率对照，请选择 `linear_emissivity`。

## 完整单卡运行

```bash
cd /home/liupengyu/physir-splat
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  -s data/RGBT-Scenes/RoadBlock \
  -m "output/RoadBlock_sam3_$(date +%Y%m%d_%H%M%S)" \
  --data_branch rgbt --eval --load2gpu_on_the_fly \
  --iterations 30000 --phys_material_mode sam3_depth \
  --phys_material_config data/materials/sam3_materials.json \
  --phys_probe_mode learnable_linear_emissivity \
  --phys_probe_every 1000 --phys_decouple_steps 128
```

该命令直接显示训练日志；完成后可用 `render_probe_maps.py` 显式导出所需检查点。后期一致性及物理辅助损失保留现有默认值。固定检查点重拟合和完整训练的结果应分别解读。

## 固定已训练点云，单独验证 Probe

```bash
cd /home/liupengyu/physir-splat
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python \
  thesis_scripts/refit_probe.py \
  -m output/RoadBlock_linear_20261020_235749/all --iteration 30000 \
  --output-model output/RoadBlock_learnable_refit_my_run \
  --phys_probe_mode learnable_linear_emissivity \
  --phys_probe_emissivity_slope 0 --phys_probe_temp_ref 300 \
  --phys_probe_lambda_slope 0.001 --phys_decouple_steps 128

CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python \
  thesis_scripts/render_probe_maps.py \
  -m output/RoadBlock_learnable_refit_my_run --iteration 30000 --split test
```

输出路径必须不存在；旧实验需要保留，因为新目录复用其点云和几何权重。相同点云、随机种子、Probe 步数下，可将模式改为 `original` 或 `linear_emissivity`（给定固定 k），分别创建新目录进行对照。仅重拟合 Probe 不等于重新训练 RGB/IR 渲染分支，不应将 Probe 拟合误差当作图像 PSNR。

## 缓存、导图与诊断

兼容已有 `phys_probe.pt`，沿用 `cluster_eps0`、`cluster_k`、`emissivity_ref`、`emissivity_slope`。新模式的 k 是实际每簇学习值；`probe_metadata.slope_per_K=null`，不能把它当成固定公共斜率。元数据保存模式、范围、端点参数化、初值、正则和内部预热设置。

原导图命令继续可用，包含 temperature_K、epsilon(T)、参考发射率、斜率及辐射分量。`material_parameters.npz` 保存逐簇 eps0/k/数量。应在逐 Gaussian 的缓存上检查 epsilon(T) 公式：像素上的加权平均系数与平均温度不能简单相乘，因为它们的乘积平均涉及协方差。

日志保存斜率 min/max/mean/std、整个温区的发射率 extrema，并在终端显示斜率范围。发射率实际值接近边界、系数剧烈变化或温度场明显漂移时，应与零斜率对照一起检查。


当前伪彩图经亮度转换，温度仍是归一化 Planck LUT 的估计量。可学习斜率属于先验约束下的有效系数；数值收敛或更低拟合误差不能独立证明真实温度或材料斜率更准确。合成已知温度测试仅验证可辨识子问题中的实现正确性。
