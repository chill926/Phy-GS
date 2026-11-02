当前在线属性与GT反馈流程见 [attribute_probe_pipeline.md](attribute_probe_pipeline.md)。下文仅描述不回传梯度的全SH观测缓存；该缓存与在线反馈是分开的。

> 当前默认模式已升级为带独立属性拟合的 full_sh_radiometric_attributes，见 [attribute_probe_pipeline.md](attribute_probe_pipeline.md)。下文保留观测提取模式说明。

# 主训练的完整 SH 观测 Probe

默认 `phys_probe_mode=full_sh_observations`。主重建仍使用 RGB/IR 的 0～3 阶 SH，保留 B 的几何修正与 DropGaussian。此次接入的是已验证的输入提取，没有新增温度/发射率反演，也没有将 SH 渲染替换为物理前向。

每1000轮，新 Probe 对当前训练视角按原渲染器的方向约定计算完整 SH：16×3 系数 → 方向 RGB → `0.299R+0.587G+0.114B` 显示灰度标量。活跃 SH 阶数仍随原训练进度增加。逐视角保留标量、方向和实际 alpha 合成贡献，另存可见性加权均值/标准差/有效视角数；均值不是完整观测的替代品。

IR 观测使用 IR 几何网络，SAM3 分区使用 RGB 几何网络。材料分区及深度聚类每次根据当前高斯重新计算，金属先验仍可记录为0.3、边界0.2～0.4，但这不是已经估计出的发射率。只使用 scene.getTrainCameras，测试视角不进入 Probe。

可见性权重采用完整透明度、原训练图像网格以及 accumulation≥0.98 的像素，经颜色伴随梯度累积；没有 DropGaussian，也不依赖 RoadBlock 的 DN 映射文件。独立测试曾在原始热网格上统计权重，此处使用主训练网格，因此加权统计不应要求逐位相同；相同视角的 SH 标量应完全相同。

按此前确定的独立 Probe 方案，这个模式会关闭旧 Probe 置信加权 MSE、RT guidance 和物理渲染混合；实际值先写回 opt 再保存 cfg_args。其余 B 的重建损失保留。若继续旧检查点，会清除内存中旧 DC 反演的物理缓存；源检查点文件不会被改写。旧模式仍可供历史工具显式使用。

## 从头运行30000轮

```bash
cd /home/liupengyu/physir-splat
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python -u train.py \
  -s data/RGBT-Scenes/RoadBlock \
  -m "output/RoadBlock_B_fullSH_$(date +%Y%m%d_%H%M%S)" \
  --data_branch rgbt --eval --load2gpu_on_the_fly \
  --iterations 30000 --sh_degree 3 \
  --phys_probe_mode full_sh_observations \
  --phys_probe_every 1000 --phys_probe_sh_views 0 \
  --phys_material_mode sam3_depth \
  --phys_material_config data/materials/sam3_materials.json \
  --phys_metal_e_init 0.3 --phys_metal_e_half_range 0.1 \
  --ir_late_l1_weight 0.8 \
  --ir_late_consistency_weight 0.2 \
  --ir_late_ssim_weight 0.2 \
  --late_lr_schedule cosine \
  --late_lr_start_iter 20000 --late_lr_end_iter 30000 \
  --late_lr_final_factor 1.0
```

不需要重新运行 SAM3。此命令保持原 B 的学习率配置，`final_factor=1.0` 表示没有额外后期降幅。日志直接显示在终端。由于 Probe 不反馈，完整 SH 提取本身不应被解释为新的 PSNR 优化机制；与旧 B 的差异还包括按此前选择停用旧 Probe 辅助损失。

## 保存内容

- `phys_probe/iteration_1000/full_sh_observations.pt`，后续每1000轮同样保存。
- `phys_probe/iteration_1000/summary.json`，以及累计 `phys_probe/probe_log.jsonl`。
- `full_sh_observations.pt` 自带源迭代、坐标、完整48维系数、视角名、逐视角标量/方向/贡献权重、支持标记、统计量、SAM3 分区、材料先验。
- 单位明确为 `display_gray_proxy_NOT_DN_NOT_physical_radiance`。不会生成含虚构温度的 `phys_probe.pt`；旧物理场渲染器不能把这个观测包当成温度/发射率缓存。
- 每个视角进行原渲染器一致性和灰度投影检查，超出阈值立即停止。提取不改变参数或已有梯度，并恢复相机设备和 Python/NumPy/Torch RNG。

## 已完成验证

从 B 的30000轮检查点短跑到30002轮，真实执行完整62个训练视角 Probe，再完成后续训练更新、评估和保存。与独立测试共享的49个视角，逐高斯标量最大差异0；主渲染器参考最大误差0；双精度加权均值和标准差与独立计算误差约1e-15。已有参数、梯度、随机数和相机设备均未改变。另从头短跑检查活跃0阶、几何 warmup 和新模型保存路径。短测不是完整30000轮效果评估。
