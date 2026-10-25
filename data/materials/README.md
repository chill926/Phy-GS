# SAM3 材料划分与发射率配置

配置文件：`data/materials/sam3_materials.json`。SAM3 在独立的 `sam3` 环境离线生成掩膜；训练和 Probe 在原来的 `3dgs` 环境读取掩膜，无需在训练环境导入 SAM3。

## 开放词汇如何填写

使用简短、具体的英文名词短语，每个短语单独推理，再在同类内合并。比起只写 `metal` / `nonmetal`，描述画面中实际出现的物体或表面通常更容易检查和修正。当前 v2 按物体分组：通用分组在 `sam3.groups`，场景覆盖分组在 `sam3.scene_groups.<场景目录名>`。每组配置类别、提示词、检测阈值、优先级、面积过滤和检测视图。`sam3.prompts` 保留给 v1 兼容路径；v2 以 groups 为准。

当前 `RoadBlock` 场景使用：

| 类别 | 提示词 |
| --- | --- |
| 金属螺丝/螺栓 | `metal screw`, `metal bolt`, `metal nut` |
| 地面金属块 | `block on the ground`, `metal slab`, `metal block` |
| 非金属路桩 | `bollard`, `red bollard`, `red plastic post` |
| 非金属反光带与地面 | `reflective tape`, `concrete ground` |

其他场景使用通用分组，包括金属表面、螺丝、螺母、金属板，以及塑料、橡胶、木材、混凝土、织物等。新增场景时应按实际物体调整。RoadBlock 的路桩属于非金属，地面小块属于金属，这是场景已知的类别先验（小块由用户确认），并非 SAM3 测得的材质。`block on the ground` 不能直接推广成“任何场景的地面物块都是金属”。这里的 `reflective tape` 假设为聚合物反光膜；如果实物是裸露金属箔，应更改归类。

SAM3 判断的是视觉与文字的匹配，不能测量真实材质或发射率。涂漆金属应按外露涂层处理，细小螺丝也可能漏检。两类都没有可靠响应，或者两类响应接近时，标为未知；不把“没有检测成金属”自动当作非金属。

## 标签与发射率

| 标签 | 类别 | 掩膜预览色 | 发射率初值 | 允许范围 |
| --- | --- | --- | --- | --- |
| 0 | 未知 | 灰 | 继承原 Probe，默认 0.95 | 继承原 Probe，默认 0.85～0.99 |
| 1 | 非金属 | 绿 | 继承原 Probe，默认 0.95 | 继承原 Probe，默认 0.85～0.99 |
| 2 | 金属 | 蓝 | 0.30 | 0.20～0.40 |

`inherit_probe` 保留现有 `--phys_probe_e_init`、`--phys_probe_e_min`、`--phys_probe_e_max` 的设置。金属单独使用 JSON 中的值。原有手工 `material_masks` 不覆盖，v2 数据保存于 `material_masks/sam3_v2/`，原 v1 数据保留在 `material_masks/sam3/`。两种掩膜的标签定义不同，不能混用。

三种 Probe 模式都支持材料约束：`original`、`linear_emissivity`、`learnable_linear_emissivity`。线性模型对整个配置温度区间约束 ε(T)，包括可学习斜率；固定斜率不可行时明确报错。

主训练入口现在默认使用 `learnable_linear_emissivity`，金属默认值来自当前 JSON 的0.30/0.20～0.40；非金属和未知类别维持原设置。`--phys_metal_e_init` 与 `--phys_metal_e_half_range` 仍可显式覆盖各次实验。学习率、调度、损失权重及几何网络训练期限在本次材料默认值更新中没有改动。

`sam3_materials_eps020_reference.json` 保留旧配置原始字节及签名。当前配置的 `mask_source_config` 仅用于兼容已生成的分割缓存：加载器要求旧配置与当前配置的标签及全部 SAM3 设置完全相同，并校验 manifest 的原配置签名。发射率拟合始终读取当前 JSON，不会从参考文件恢复旧发射率值。现有 v2 掩膜和 manifest 无需修改，也无需重新运行 SAM3。

## 生成掩膜

在服务器运行：

```bash
cd /home/liupengyu/physir-splat
conda activate sam3
CUDA_VISIBLE_DEVICES=0 python tools/segment_sam3_materials.py \
  --scene data/RGBT-Scenes/RoadBlock \
  --config data/materials/sam3_materials.json
```

默认仅使用 `rgb/train`，不读取测试图。当前 RoadBlock 的 62 张训练图已经生成，无需再次执行。输出包括标签 PNG、纯标签彩色 PNG、叠加预览 JPG、带图例和金属候选框的 `.qa.jpg`，以及证据分数 NPZ。NPZ 的 metal/nonmetal 是传给三维融合的分数，raw_metal/raw_nonmetal 保留获胜层级内的未按最终类别过滤的分数。manifest 记录配置、权重、生成器/规则代码、图像与缓存 SHA256，以及每个保留实例的提示词、分数、框、区域面积和裁剪窗口。

首次检查其他场景可添加 `--max-images 3 --output output/material_pilot`。该抽样掩膜仅供显式重拟合试验，正式自动训练要求全部训练相机都有掩膜。

中断后，配置和权重未变时使用 `--resume`。修改分割提示词、阈值、裁剪等 SAM3 设置后，应使用新的 `--output` 重新生成掩膜，并在训练时指定对应 `--phys_material_manifest`。仅调整物理拟合设置可通过上述经过校验的参考配置复用掩膜；生成脚本的 `--resume` 仍要求原始生成配置不变。程序拒绝静默混用分割规则不同的缓存。

## v2 对小物体和掩膜覆盖的改进

1. 全图检测路桩、反光带、地面和螺丝。用 `bollard` / `red bollard` 补上 `plastic bollard` 的漏检；移除曾把金属块误认为塑料底座的 `plastic base` 场景提示词。
2. 根据检测到的路桩边框自动裁剪底部，重新编码放大后的区域，检测小块和螺丝；没有写入某一帧的固定位置。小块未检出时最多左右各重试一次；物块被裁剪边界截断或仍未检出时再扩大一次上下文。相关比例在 `anchor_crops` 中配置。
3. 按组过滤检测分数、掩膜面积、占裁剪区域比例，再使用独立的掩膜概率阈值（0.55）。避免简单降低全局阈值后把大块地面判成金属。
4. 掩膜优先级为地面 0、路桩/反光带 10、螺丝/小金属块 30。只有实际检测掩膜覆盖到的像素才参与更高层级覆盖。具体小物体的合格掩膜可覆盖地面或路桩父掩膜；同一层级的相反材料仍按分数差保留未知选项。不同层级的检测分数不直接比较。

v2 的 `pixel_threshold=0.2` 是经过实例过滤之后的最低证据阈值，与实例检测阈值不同；没有检测到实例时不会因此生成金属标签。此版本是基于该场景的改进，仍可能漏检、边缘偏移或误检，需要查看 QA 图，不能用像素数增加宣称分割准确率提高。

## 三维融合与深度聚类

每次 Probe 对当前高斯重新融合，避免增密、删除点后索引失效。默认均匀选择最多 12 个训练视角，利用 RGB 几何及实际光栅化的 alpha/透射贡献累计材料证据，至少两个视角一致才赋予材料类别。

从当前 RGB 高斯渲染归一化深度，相邻像素同类且相对深度跳变不超过 3% 时连接。过滤低覆盖像素和小区域，再结合三维近邻、跨视角区域一致性建立材料区域。未知点继续使用原特征聚类和原发射率先验。不同材料/区域之间不施加温度平滑。

深度连续是区域线索，不能保证区分接触且同深度的两个物体；材料边界也同时参与分区。`fusion` 下的阈值可以按场景调整。`material_confidence` 是融合证据强度，不是经过校准的材质概率。

## 分阶段优化

沿用已有周期 Probe 入口，默认每 1000 次训练迭代执行。材料分支在每次 Probe 内部分阶段释放参数：

1. 前 25% 步：固定发射率和残差，优化温度及环境项。
2. 25%～50%：释放参考发射率和残差；发射率学习率为基础值的 0.25 倍。
3. 后 50%：可学习线性模式再释放温度斜率。

默认 128 步对应第 32、64 步释放。阶段配置位于 `optimization`。这能缓解相互补偿，但不构成物理参数唯一可辨识的证明。非金属的数值先验和范围保持原设置，材料分组和平滑按新分区执行。

## 训练与现有检查点重拟合

`--phys_material_mode auto` 是默认值：配置的 `sam3.mask_directory` 下有完整匹配的 manifest 时启用材料划分；没有时保持旧 Probe。`sam3_depth` 要求掩膜存在，适合明确使用新版材料划分的实验；`legacy` 显式使用原流程。已有完整掩膜时，下列命令可启动一个新实验（选择空闲 GPU 与新的输出目录）：

```bash
conda activate 3dgs
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/RGBT-Scenes/RoadBlock -m output/RoadBlock_sam3_new \
  --data_branch rgbt --eval --load2gpu_on_the_fly \
  --phys_material_mode sam3_depth \
  --phys_probe_mode learnable_linear_emissivity
```

仅重新拟合现有检查点的物理属性：

```bash
CUDA_VISIBLE_DEVICES=0 python thesis_scripts/refit_probe.py \
  -m output/RoadBlock_learnable_20261021_204258/all --iteration 30000 \
  --output-model output/RoadBlock_sam3_refit_new \
  --phys_material_mode sam3_depth \
  --phys_probe_mode learnable_linear_emissivity --phys_decouple_steps 128
```

重拟合结果必须写入新目录；源检查点校验前后哈希一致。导出兼容材料字段：

```bash
CUDA_VISIBLE_DEVICES=0 python thesis_scripts/render_probe_maps.py \
  -m output/RoadBlock_sam3_refit_new --iteration 30000 --split test
```

缓存增加 `material_class`、`material_confidence`、`cluster_class` 和逐点发射率上下界。导出增加金属可见贡献 `metal_fraction`、材料证据图及 NPZ；Probe 温度仍是模型估计，未因此变成实测温度。

## v1 历史验证（2026-10-25，已发现材料漏检）

完整 62 张训练图的掩膜已生成。基于原 iteration 30000、51558 个高斯点进行 128 步重拟合，结果在 `output/RoadBlock_sam3_material_probe_20261025`：

| 类别 | 高斯点数 | 当前温度下拟合的 ε 范围 |
| --- | --- | --- |
| 金属 | 854 | 0.195585～0.262994 |
| 非金属 | 29458 | 0.946211～0.973109 |
| 未知 | 21246 | 0.947758～0.975124 |

该结果验证了流程和约束，不能作为自动分割正确率或材料发射率真值。原检查点保持不变。24 项测试通过，覆盖三种 Probe 模式、全温区边界、多视角融合、深度边界、缓存读写与错误缓存拒绝；另完成一次训练迭代的自动材料入口验证，以及一张测试视角的物理图导出。尚未完成新版材料划分下的全程训练，也未验证重建指标是否改善。

## v2 验证

六张均匀抽样的训练视角均检出了地面小金属块，柱体主体已被识别为非金属。此结论来自实例记录与预览核对，不是带真值标签的准确率统计。全部训练图生成与三维融合验证结果见本次交付说明。旧配置备份在 `output/material_validation/sam3_materials_v1_config.json`；旧检查点、v1 掩膜和物理结果保留。

完整 62 张训练图已生成并逐帧校验图像/掩膜/分数 SHA256，当前默认目录为 `material_masks/sam3_v2/`。62 张均有路桩主体检测，58 张有小块检测；未检出的帧为 `FLIR5669`、`FLIR5702`、`FLIR5703`、`FLIR5704`。这些帧可能存在遮挡或漏检，不能把 58/62 当作准确率。原示例的小块框区域金属标签由 0 个像素变成 520 个像素，预览对比可见修正。

新结果目录：`output/RoadBlock_sam3_v2_material_probe_20261025`，基于原检查点 iteration 30000，独立重拟合 128 步。

| 类别 | 高斯点数 | 当前温度下拟合的 ε 范围 |
| --- | --- | --- |
| 金属 | 1180 | 0.198865～0.262592 |
| 非金属 | 31376 | 0.941567～0.963664 |
| 未知 | 19002 | 0.951723～0.973281 |

金属全温区端点发射率为 0.198096～0.263034，保持在要求的 0.1～0.3 内；非金属仍继承原初值和范围。原检查点哈希校验一致，30 项测试通过。完成单视角物理图导出及自动训练入口的一次迭代短跑；完整重训练和带真值的材质分割评测尚未进行。详细核查日志在 `output/material_validation/quality_v2.json` 和 `verification_v2.json`。
