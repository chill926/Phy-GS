# B_clean_verify基线：可学习发射率斜率与GT反馈

2026-11-02按用户要求恢复 `RoadBlock_B_clean_verify_20261101_171935` 对应源码，再增加可学习epsilon(T)和原Probe形式的加权GT MSE反馈。恢复时46份源码全部匹配历史清理清单中的SHA256；当前仅修改5份原源码，新增 `utils/attribute_feedback.py`，其余41份源码仍逐字匹配B_clean_verify。

回退前全部源码和文档已备份到 `output/code_backups/before_restore_B_clean_feedback_20261102_232152/`。本次未修改graduate、数据集、SAM3掩膜、已完成训练输出或B_clean_verify的原缓存。

## 当前训练行为

- B重建目标保持：IR的0.8 L1 + 0.2 D-SSIM，20000轮后加0.2 RT；RGB、几何、DropGaussian、SH0～3及学习率沿用B。
- 前20000轮不启用新增属性反馈。20000轮保存初始化用的全SH观测、点云及几何。
- 20001轮起，在固定高斯拓扑上维护持续学习的T、区域epsilon0、k、环境及方向参数；实际主相机属于49个fit视角时进行一次属性Adam更新，13个属性validation视角不更新属性。每次属性更新处理一个当前fit视角。
- 属性目标沿用B_clean属性读出的原生DN Huber、金属核心附加项、冻结20000轮全SH显示辅助项和先验。主高斯/几何在属性目标中detach；属性损失不会意外直接修改主SH。
- 独立属性参数生成原Probe的有界启发式可靠性权重，权重detach。额外MSE的目标始终是真实IR GT：`gate * ((1-mix)*MSE + mix*weighted_MSE)`。它通过普通IR图像回传到IR DC、高阶SH及主IR渲染相关几何；不会直接更新RGB SH。
- MSE从20001轮开始，5000轮升至0.05，mix=0.5，归一化权重边界0.75～1.25。B原有损失继续保留。这里的可靠性是继承的启发式，不是经过校准的温度不确定性。
- 最终保存训练过程中学到的属性参数，不再在训练结束后独立重新拟合并覆盖它们。每1000轮的全SH观测导出仍然执行；该观测缓存本身不回传梯度。

## epsilon(T)

`epsilon(T) = epsilon0 + k * (T - 300 K)`，epsilon0和k按材料/深度区域共享。金属参考初值及先验0.3、范围0.2～0.4；非金属沿用0.95及材料配置边界。采用温度区间两端的有界发射率参数化，保证260～360 K整个区间内均满足各自边界。k初始为0，之后可学习。

30000轮完整训练的释放时点：

|参数|起始迭代|
|---|---:|
|T、环境及GT反馈|20001|
|方向项|22501|
|epsilon0|25001|
|k|27501|

后三项分别对应反馈阶段的25%、50%、75%。`phys_attribute_slope_start_fraction`控制k释放时机；`phys_probe_slope_lr_scale=0.25`、`phys_probe_lambda_slope=0.001`控制斜率学习率比例与先验。

`phys_attribute_steps=1024`和`phys_attribute_views_per_step=4`仅用于无反馈的独立属性拟合；当前在线模式依据后10000个主迭代持续更新，并单独记录实际属性更新次数。设置`--phys_aux_mse_max_weight 0`可关闭在线反馈，回到最终独立属性拟合流程。

## 从头训练

```bash
cd /home/liupengyu/physir-splat
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python -u train.py \
  -s data/RGBT-Scenes/RoadBlock \
  -m "output/RoadBlock_Bclean_slope_feedback_$(date +%Y%m%d_%H%M%S)" \
  --data_branch rgbt --eval --load2gpu_on_the_fly \
  --iterations 30000 --sh_degree 3 \
  --phys_probe_mode full_sh_radiometric_attributes \
  --phys_probe_every 1000 \
  --phys_material_mode sam3_depth \
  --phys_material_config data/materials/sam3_materials.json \
  --phys_metal_e_init 0.3 --phys_metal_e_half_range 0.1 \
  --phys_aux_start_iter 20000 --phys_aux_ramp_iters 5000 \
  --phys_aux_mse_max_weight 0.05 --phys_aux_conf_mix 0.5 \
  --phys_attribute_slope_start_fraction 0.75 \
  --phys_probe_slope_lr_scale 0.25 --phys_probe_lambda_slope 0.001 \
  --ir_late_l1_weight 0.8 --ir_late_ssim_weight 0.2 \
  --ir_late_consistency_weight 0.2 \
  --late_lr_schedule cosine --late_lr_start_iter 20000 \
  --late_lr_end_iter 30000 --late_lr_final_factor 1.0
```

实时日志：`ATTRIBUTE DATA GRAD`核查T、epsilon0和k等的非零数据梯度；`ATTRIBUTE DATA PATH`核查属性损失没有泄漏到主参数；`ATTRIBUTE SH FEEDBACK GRAD`核查GT反馈对IR DC和高阶SH的非零梯度。每1000轮写`phys_probe/feedback/optimization.jsonl`，含权重、启用阶段和实际更新次数。

## 缓存和渲染

普通保存迭代的`point_cloud/iteration_*/radiometric_probe.pt`包含可学习斜率、实际epsilon(T)、属性Adam状态、反馈标记与点云/几何哈希。最佳PSNR镜像保持SH-only。旧固定斜率缓存仍可读取。

```bash
CUDA_VISIBLE_DEVICES=0 /home/liupengyu/miniconda3/envs/3dgs/bin/python -u render.py \
  -m output/你的新训练目录 --iteration 30000 --mode physical_fields
```

输出测试集的IR-GT、IR-Render、T、epsilon、RGB-GT、RGB-Render，以及浮点温度、实际epsilon(T)、参考epsilon0和k。温度和发射率仍是先验条件下的估计，不是校准真值。

## 已完成验证

- 从B_clean_verify的20000轮权重真实运行8步，覆盖缩短后的全部释放阶段，6次fit视角属性更新；IR DC和高阶SH均获得反馈梯度，RGB SH对该项梯度为0。
- 斜率具有非零数据梯度，最终非零；缓存通过整个温度区间端点边界及epsilon(T)一致性检查。
- 再做1步真实检查，确认属性数据目标对主Gaussian/SH的梯度为0，GT MSE对IR SH梯度非零。
- 新缓存已通过全部9个test视角物理场渲染；原B_clean_verify固定斜率缓存兼容检查通过。
- 短测位于`/tmp`，项目未新增测试入口。短测缩短了总迭代数和释放阶段，不能用于判断30000轮质量提升。未代替用户运行完整30000轮。
