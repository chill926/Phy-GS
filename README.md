# Phy-GS

This repository contains research modifications to [PhysIR-Splat](https://github.com/JingyuanGao0919/physir-splat). The upstream Git history, license, and copyright notices are preserved.

The additions include SAM3 open-vocabulary material segmentation, depth-aware material grouping, fixed-slope and learnable linear emissivity probes, probe map export, and FLIR display mapping tools. The default Probe learns the emissivity slope and uses SAM3 depth-based material partitions. Metal starts at 0.30 with bounds 0.20–0.40 (center ±0.10) over the fitted temperature interval; nonmetal and unknown regions retain the original 0.95 initial value and 0.85–0.99 bounds.

The current training defaults retain B loss (IR: 0.8 L1 + 0.2 D-SSIM through 20k; add 0.2 RT after 20k), the metal emissivity settings above, and progressive IR DropGaussian described below. Extra late learning-rate decay is disabled (`late_lr_final_factor=1.0`); the existing XYZ/network schedules and Adam settings are preserved. When reusing a command from a late-LR experiment, remove its `--late_lr_final_factor 0.1` override or change it to `1.0`. New training defaults to `phys_material_mode=sam3_depth` and requires compatible SAM3 train masks. Existing RoadBlock v2 masks can be reused.

Normal training uses the B configuration with the original within-region temperature smoothing (weight 0.05). The experimental graph-based temperature smoothing and cross-region reference-emissivity consistency have been reverted after full-run checks did not show a stable reconstruction benefit. SAM3 partitions, emissivity bounds, learnable slope, B loss, DropGaussian, geometry correction and optimizer schedules are retained. See [material details](data/materials/README.md).

### Progressive IR DropGaussian

The normal `train.py` entry now applies [DropGaussian](https://github.com/DCVL-3D/DropGaussian_release) to the primary IR training render from the first iteration. Each Gaussian gets an independent Bernoulli keep mask shared across its color channels. Only the temporary rasterizer opacity is multiplied by `keep / (1-p)`; stored opacity, Gaussian count and checkpoint layout are unchanged. RGB training, evaluation, Probe targets, material alpha-adjoint/depth fusion and other auxiliary renders always use the complete model. Probe fitting can still change indirectly because the learned IR appearance and geometry change.

The implementation was checked against official commit `c13da4aaddfed432fb556fdcf5d1d028b967ada8` (`gaussian_renderer/__init__.py`, `train.py`, `arguments/__init__.py`). The official schedule is `p=0.2*iteration/10000` for a 10k-step run. Here `p=ir_dropgaussian_max_rate*clip(iteration/iterations,0,1)`, default maximum 0.2: at 10k/20k/30k of a 30k run, p is 6.67%/13.33%/20%. This deliberately avoids reaching 60% at 30k. It is an IR adaptation, not a reproduction of the official sparse-view training recipe. Existing image postprocessing, losses, SH degrees, densification and optimizers are retained; no CUDA extension changes are needed. The compensation preserves expected scalar opacity, not the exact rendered image under nonlinear alpha compositing.

Masks use a separate RNG (`ir_dropgaussian_seed=0`) seeded by global iteration. This leaves the global random streams used by training/Probe untouched and reproduces a step's mask for the same point ordering/count. Weight-only checkpoints still do not restore Adam or camera/random state. Changing the iteration budget changes the probability schedule; choose the intended full budget when resuming. Probability and realized dropped counts are printed every 1000 steps and saved to `ir_dropgaussian.jsonl` and TensorBoard. Densification receives gradients from the dropout IR render, as in the official approach, so training geometry and point counts can differ from older runs.

The pre-DropGaussian code is archived at commit [`75cefeb`](https://github.com/chill926/Phy-GS/commit/75cefeb). Use Git to return to that version if needed; the current training entry directly uses DropGaussian.

SAM3 runs offline in its own environment. Training reads the generated train-view masks without importing SAM3. Scene images, model weights, generated masks, and training outputs are excluded from Git; the material configuration and implementation are included. See the accompanying guides:

- [Linear emissivity probes](thesis_scripts/README_linear_emissivity.md)
- [Learnable emissivity probes](thesis_scripts/README_learnable_emissivity.md)
- [Probe map export](thesis_scripts/README_probe_maps.md)
- [SAM3 material segmentation and emissivity configuration](data/materials/README.md)
- [DN supervision and FLIR display mapping](tools/README_DN_supervision.md)

The original project documentation follows.

---

# PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting

This repository is organized around the CVPR 2026 paper **"PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting"**.

PhysIR-Splat models thermal infrared image formation on Gaussian primitives. Each primitive can carry radiative-transfer attributes such as temperature, emissivity, and environmental irradiance.

## Method Overview

The paper describes two coupled components:

- **PhysIR-Splat**: a thermal Gaussian rendering framework based on passband Planck radiance, environmental reflection, atmospheric transmittance, and monotonic radiometric response.
- **VGGT-IR**: a Transformer-based thermal infrared initializer that directly regresses camera poses, depth, and point maps from multi-view TIR input with optional RGB.

## Dataset

The training examples use TI-NSD scenes organized as:

```shell
data/
  TI-NSD/
    apples/
    basketball_court/
    ...
```

The split follows the paper setup: with `--eval`, every 8th image is used as the test set and the remaining images are used for training.

RGB+thermal scenes are also supported when organized like RGBT-Scenes:

```shell
scene_name/
  sparse/0/                 # or colmap/sparse/0/
  rgb/
    train/
    test/
  thermal/
    train/
    test/
```

## Environment

```shell
conda create -n physir python=3.7
conda activate physir

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

If you already use an existing environment, keep it active and only install missing dependencies.

## Training

```shell
python train.py -s path/to/your/TI-NSD/scene_name --eval
```

RGB+thermal:

```shell
python train.py -s path/to/your/RGBT-Scenes/scene_name --eval
```

For RGBT training, the code keeps a mirrored copy of the best evaluated
thermal checkpoint as `iteration_30001`. This is used only by the RGBT branch;
TI-NSD IR-only training keeps its existing behavior.

During training, validation PSNR/L1 is computed on save-image-compatible
8-bit tensors. This matches `metrics.py` on rendered PNGs and is used only for
logging/checkpoint selection; it does not enter the optimization loss.

Select a data branch explicitly:

```shell
python train.py -s path/to/your/TI-NSD/scene_name --eval --data_branch ir
python train.py -s path/to/your/RGBT-Scenes/scene_name --eval --data_branch rgbt
```

`--data_branch auto` is the default and keeps automatic scene detection.

The default schedule trains for 30k iterations.

## Rendering And Evaluation

```shell
python render.py -m output/exp-name
python metrics.py -m output/exp-name
```

## Acknowledgements

We thank the authors of the following thermal 3D Gaussian Splatting works for their inspiring contributions:

```bibtex
@inproceedings{luthermalgaussian,
  title={ThermalGaussian: Thermal 3D Gaussian Splatting},
  author={Lu, Rongfeng and Chen, Hangyu and Zhu, Zunjie and Qin, Yuhang and Lu, Ming and Yan, Chenggang and others},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

@inproceedings{chen2024thermal3d,
  title={Thermal3d-gs: Physics-induced 3d gaussians for thermal infrared novel-view synthesis},
  author={Chen, Qian and Shu, Shihao and Bai, Xiangzhi},
  booktitle={European Conference on Computer Vision},
  pages={253--269},
  year={2024},
  organization={Springer}
}

@inproceedings{nam2025veta,
  title={Veta-GS: View-dependent deformable 3D Gaussian Splatting for thermal infrared Novel-view Synthesis},
  author={Nam, Myeongseok and Park, Wongi and Kim, Minsol and Hur, Hyejin and Lee, Soomok},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={965--970},
  year={2025},
  organization={IEEE}
}
```

## Paper Reference

```bibtex
@inproceedings{gao2026physir,
  title={PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting},
  author={Gao, Jingyuan and Hu, Yumeng and Gao, Fei and Zhang, Mingjin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11818--11828},
  year={2026}
}
```
