# PhysIR-Splat：原始 FLIR DN 监督第一版

> **已回退：2026-10-21T20:41:16.462559** 主训练已恢复到 DN 修改前的可学习 ε(T) 版本。DN 专用训练入口、训练模块和测试已移入 `/home/liupengyu/physir-splat/output/code_backups/dn_reverted_20261021_204116`。原始数据、实验结果及独立拟合脚本保留，未启动新训练。以下为已撤下 DN 方案的历史记录，不作为当前训练指引。

> 2026-10-21 更新：本次 DN 实验已停止（约 11010 次）。第 10000 次 test 为 PSNR 19.7929、SSIM 0.835671，MAE_DN 20.3334，较第 3000 次退化。本版不能视为已验证有效的训练方案。代码和检查点保留，尚未回退；不再自动启动后续训练。

已直接修改服务器 `/home/liupengyu/physir-splat`，并启动单卡 RoadBlock 训练。

- GPU：0。
- 正式输出：`/home/liupengyu/physir-splat/output/RoadBlock_DN_20261021_202320/all`。
- 启动记录：父目录 `launch.json`；训练日志：`all/train.log`。
- 修改前备份：`output/code_backups/dn_supervision_20261021_195958/before.tar.gz`。
- 旧实验、原始数据、两个已有 worker 脚本均未改动。

## 独立显示映射

脚本：`tools/fit_flir_display_mapping.py`，包含 `fit`、`evaluate`、`apply` 三个子命令。依赖现有环境的 NumPy、Pillow、PyTorch，不依赖另装 ExifTool，不依赖主训练器。

示例（在服务器项目根目录执行，输出目录必须不存在）：

```bash
/home/liupengyu/miniconda3/envs/3dgs/bin/python tools/fit_flir_display_mapping.py fit \
  --scene data/RGBT-Scenes/RoadBlock \
  --output output/dn_display_mapping/RoadBlock_new_fit \
  --model monotone --order colorize_then_resize \
  --steps 800 --pixels 2048 --device cpu --threads 4
```

仅从 `thermal/train` 读取匹配图像，按文件名排序每五帧留出一帧作验证。本次 49 帧拟合、13 帧验证；test 不参与显示映射的参数拟合和方案选择。验证集选好后，冻结全部参数再评价 9 张 test。

共同拟合参数是六个几何参数、全局增益与偏移，以及可选的 12 节点单调曲线；没有逐图拟合，也没有使用 CNN。保留原始 224 色 Iron 色表，参考有限范围 Y/Cr/Cb 到 RGB 转换；在原始热网格上上色再重采样的方案效果更好。

| 方案 | 验证 PSNR | 验证 SSIM |
|---|---:|---:|
| 线性、先缩放再上色 | 37.6686 | 0.971313 |
| 线性、先上色再缩放 | 39.4284 | 0.973864 |
| 单调、先缩放再上色 | 37.7048 | 0.971595 |
| 单调、先上色再缩放（选用） | 39.4636 | 0.974125 |

选中方案的 test 前向显示复现：PSNR **39.1503 dB**，SSIM **0.976058**，8 位 RGB MAE **1.3267**。这些是**真实原始 DN 的显示转换误差**，不是训练模型的性能，也不是温度恢复精度。不能称为已经证明与原软件完全相同的映射。

拟合输出：`output/dn_display_mapping/RoadBlock_20261021_195046`。冻结参数：`monotone_colorize_then_resize/mapping.json`。每次正式训练都会复制一份到实验目录 `dn_display_mapping.json`，并保存源代码快照和映射哈希。

应用到新原始 FLIR 文件：

```bash
/home/liupengyu/miniconda3/envs/3dgs/bin/python tools/fit_flir_display_mapping.py apply \
  --checkpoint output/dn_display_mapping/RoadBlock_20261021_195046/monotone_colorize_then_resize/mapping.json \
  --raw data/RGBT-Scenes/RoadBlock/raw_images/FLIR5667.jpg \
  --output output/dn_display_mapping/example_FLIR5667.png
```

也支持 `--dn prediction.npy --median 12343 --span 415`，数组须是原始热网格坐标下的实际 DN。主训练器直接调用同一个冻结显示模块，避免另写一套不一致的转换。

## 主模型的监督与评价

1. 从原始 FLIR JPEG 的 APP1/FFF 记录提取 16 位 DN、色表、显示元数据和相机 Planck 定标系数。本数据集的嵌入 PNG 必须交换每个 uint16 的两个字节，脚本显式记录该设置，不靠亮度范围猜测。
2. 热分支渲染标量信号，三个通道仅为兼容现有 CUDA rasterizer；DN 模式下取通道均值，初始热属性一致。RGB 分支仍拟合 RGB。
3. 保留现有 640×480 相机投影，通过冻结几何变换的逆变换，将渲染信号采样到原始 240×180 DN 像素中心。损失直接与原始 DN 样本比较；超出相机图像覆盖的边缘不参与损失。未把伪彩图转灰度作为 DN。
4. 用固定、可逆的归一化训练：`q=(DN-DN_bb(260K))/(DN_bb(360K)-DN_bb(260K))`。`DN_bb(T)=R1/[R2*(exp(B/T)-F)]-O`。本相机两端约为 9631.0305 和 19327.2827 DN。所有视角使用相同的温度边界与相机定标；不做逐图 min/max。
5. 热分支主损失为有效像素上的原始 DN MAE / 415。415 是本次明确保存的固定损失尺度，只影响权重，不裁剪 DN。没有将显示窗口用作物理归一化。RGB 保留原有 L1/SSIM 损失。
6. 停用旧的伪彩图晚期 radiative consistency loss。Probe 保留 `learnable_linear_emissivity` 的可学习截距与斜率，但使用相机 DN 响应替代原先平坦光谱积分 LUT。仿射归一化仍保持发射与反射混合式。Probe 缓存写入信号域和定标参数。
7. 评价时把预测的原始网格 DN 经冻结映射变为三通道伪彩图，与数据集发布的 JPEG 对应图比较。训练日志的 PSNR/SSIM 均在伪彩图上计算；额外记录 MAE_DN、RMSE_DN，不用灰度 PSNR 替代伪彩图指标。
8. 渲染导出同时保存原始单位的 `dn/*.npy`、逐帧元数据 JSON，以及普通 `renders/*.png` 伪彩图和 RGB 图。DN 实验默认不启用原有颜色残差后处理。渲染几何权重明确使用所选迭代，避免混用其他检查点的几何权重。

DN 模式下额外约束几何残差：位移分量不超过场景尺度的 1%，旋转残差分量不超过 0.05，尺度残差不超过原尺度的 ±10%，保证尺度为正；预热后用 1,000 次逐渐启用，并裁剪几何网络梯度为 1；几何网络学习率乘 0.05，减少监督尺度改变导致的剧烈更新。训练与评价使用完全相同的残差变换，参数化版本写入定标记录，避免误加载不兼容检查点。该限制仅作用于 DN 模式。

第一版保留原有 epsilon(T) 优化、发射率边界与先验；没有加入不确定度模块。由于没有可靠的逐像素距离，本版使用大气透过率约为 1 的近场近似，不把文件中的预置 1m 当作真实距离。DN 改善了监督量的物理含义，但 T、发射率与反射项仍有可辨识性限制；输出温度是模型估计，不是独立温度 GT。

## 正式运行

**本次已启动，不必重复启动。** 以后另开实验可用：

```bash
cd /home/liupengyu/physir-splat
bash thesis_scripts/run_roadblock_dn.sh 0
```

可选参数：`bash thesis_scripts/run_roadblock_dn.sh GPU编号 新输出目录 映射JSON路径`。脚本拒绝覆盖已有输出。

默认单卡训练 30,000 次，预热 3,000 次，高斯点上限 150,000，以控制 2080 Ti 的显存占用；Probe 每 1,000 次拟合，内部 128 步。结束后自动导出最终 30,000 次检查点的 DN、伪彩图和物理量图。显示映射只加载冻结参数，不随主模型优化。

查看当前日志：

```bash
tail -f /home/liupengyu/physir-splat/output/RoadBlock_DN_20261021_202320/all/train.log
```

指标文件：`dn_metrics.jsonl`；定标与原始文件来源记录：`dn_radiometry.json`、`dn_sources.json`。原有 best checkpoint 机制仍按 test 伪彩图 PSNR 选择，因此这组 test 实际承担验证作用，不应宣称为从未用于模型选择的独立最终测试集。

## 验证

- 71 张 DN 的数组哈希与此前独立 ExifTool 提取结果全部一致。
- 20 项单元测试通过，包括现有 14 项、几何正尺度和旧模式兼容检查，以及新增的相机响应逆变换、混合式归一化、标量采样梯度、边缘掩码检查。
- 80 次短程训练跑通，提前触发 Probe、辅助损失、评价、保存和重新加载；这是接口检查，刻意缩短预热，不作为模型质量结果。
- 9 张真实 test DN 经过主模型调用的显示模块，得到的 8 位结果与独立拟合脚本逐像素一致。
- 保存 PNG 的 PSNR 与短程训练评价在 1e-4 内一致，SSIM 在 1e-4 内一致。
- DN 模式下的物理量图导出通过了相机响应温度一致性校验和辐射分解校验（最大误差约 1.2e-7），打包标量图时显式保留各个 override_color 通道。
- 最初的正式实验 `RoadBlock_DN_20261021_200813` 在启用无界几何细化后发散，已停止并保留日志。新增上述几何约束后，从头启动当前 `RoadBlock_DN_20261021_202320`，不继承失败实验参数。第二次检查 `RoadBlock_DN_20261021_201632` 验证了几何有界，但更新仍偏快、高斯点增长快，已停止；当前实验另外将几何学习率降低至 0.05 倍并设置 15 万点预算。
- 正式训练使用正常 3,000 次预热；进度和初期性能以单独的训练状态记录为准。

代码备份包用于审阅和复用；服务器已应用相同修改，不需要再次覆盖。

## 正式训练观测快照

截至服务器时间 2026-10-21T20:33:08.045922，当前实验已到 8740/30000 次，仍在运行。第 7000 次 test 的伪彩图 PSNR=20.8465、SSIM=0.846531，MAE_DN=18.3738、RMSE_DN=33.1398；RGB PSNR=27.1935、SSIM=0.879045。这些不是最终成绩。第 3000 次 test 曾达到 22.8076 dB，因此当前尚不能宣称效果优于旧基线，仍需结合完整训练检查几何和先验。已观察到点数约 15 万、显存约 5.6 GiB，未再出现整体崩掉或显存不足。

输出温度仍受发射率、全局反射项和残差先验影响；这些先验尚未用独立温度测量定标。
