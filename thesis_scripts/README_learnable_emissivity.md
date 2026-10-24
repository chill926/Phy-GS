# 可学习斜率的温度相关发射率 Probe

在 `/home/liupengyu/physir-splat` 中执行。新增模式 `learnable_linear_emissivity`；原来的 `original` 和固定斜率 `linear_emissivity` 模式继续可用，默认仍为 `original`。

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

默认从 k=0、eps0=0.95 开始。每次独立 Probe 拟合的前 25% 内部步数冻结 d，此时 k 严格为 0，共同中心 c、温度、环境和残差仍优化；其后解冻 d。128 步即先 32 步再联合优化 96 步。这是每次 Probe 的内部预热，不是完整 3DGS 训练的前 25%。现有 Probe 每次重新聚类、初始化；本次没有引入跨 Probe 的参数继承。

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
bash thesis_scripts/run_roadblock_learnable.sh 0
```

第一个参数是 GPU 编号；第二个可选参数是新的输出路径：

```bash
bash thesis_scripts/run_roadblock_learnable.sh 0 output/RoadBlock_learnable_my_run/all
```

脚本完成 30000 步训练后，自动导出最终 checkpoint 的所有 test 视角物理量图。保留之前命令的训练参数及已有物理辅助损失默认值；没有加入不确定度模块。该辅助损失在后期可能受 Probe 改变影响，因此固定检查点重拟合和完整训练应分别解读。

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

```bash
/home/liupengyu/miniconda3/envs/3dgs/bin/python -m unittest discover -s tests -p 'test_*emissivity.py' -v
/home/liupengyu/miniconda3/envs/3dgs/bin/python -m unittest discover -s tests -p test_probe_maps.py -v
```

当前伪彩图经亮度转换，温度仍是归一化 Planck LUT 的估计量。可学习斜率属于先验约束下的有效系数；数值收敛或更低拟合误差不能独立证明真实温度或材料斜率更准确。合成已知温度测试仅验证可辨识子问题中的实现正确性。
