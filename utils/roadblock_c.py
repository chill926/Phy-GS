"""Production RoadBlock C profile and its post-update Probe handoff."""
import json
from pathlib import Path

import torch

from tools.full_training_state import BOUNDARY, WRAPPERS, digest, rng_state, write_json
from utils.full_sh_probe import _preserve_rng
from utils.material_config import load_material_config

ROOT = Path(__file__).resolve().parents[1]
END = 30000
SMOOTHNESS_WEIGHT = 0.3
FIT_VIEWS = 49
PHASE_1_STEPS = 2500
FIELD_LOG_STEPS = (20001, 22500, 22501, 25000, 27500, END)


def validate_profile(dataset, opt):
    if (dataset.data_branch != "rgbt" or not dataset.eval or dataset.sh_degree != 3
            or opt.iterations != END or getattr(dataset, "load_iteration", 0)
            or opt.phys_probe_mode != "full_sh_radiometric_attributes"
            or opt.phys_material_mode != "sam3_depth"
            or (opt.phys_aux_start_iter, opt.phys_aux_ramp_iters,
                opt.phys_aux_mse_max_weight) != (BOUNDARY, 5000, .05)):
        raise ValueError("RoadBlock C requires a fresh 30k RGBT run with SH0–3, SAM3 and the 20k GT feedback schedule")
    mapping = json.loads((ROOT / opt.phys_attribute_mapping).read_text())
    if len(mapping["fit_frames"]) != FIT_VIEWS:
        raise ValueError("RoadBlock C requires the 49 fit views in its frozen DN/display mapping")
    config, _, _ = load_material_config(opt.phys_material_config)
    metal = config["emissivity"]["metal"]
    center = metal["initial"] if opt.phys_metal_e_init == -1 else opt.phys_metal_e_init
    half_range = opt.phys_metal_e_half_range
    bounds = (metal["min"], metal["max"]) if half_range == -1 else (center-half_range, center+half_range)
    if center != .3 or any(abs(a-b) > 1e-6 for a, b in zip(bounds, (.2, .4))):
        raise ValueError("RoadBlock C requires a metal prior center of 0.30 and bounds [0.20, 0.40]")


def clean_ir_smoothness(image):
    neighbors = image.detach()
    return sum((image - torch.roll(neighbors, shift, dims=axis)).abs().mean()
               for axis in (1, 2) for shift in (-1, 1))


def prepare_probe(context, time_interval, *, read_only=False):
    from utils import priornorm_probe as prior

    live = context["live_feedback"]
    output = Path(context["model_path"])
    plan = dict(role="restore_check" if read_only else "full", phase1_steps=PHASE_1_STEPS,
                varpro=dict(every=128, maxiter=20, maxfun=32, gtol=1e-5, ftol=1e-8, margin=1e-6),
                prior_normalization=dict(metal_scale=.1, nonmetal_scale=.1, uniform_mix=.05))
    before_rng = digest(rng_state())
    with torch.enable_grad(), _preserve_rng(context["device"]):
        context["save_training_state"](BOUNDARY)
        context["run_material_probe"](BOUNDARY, time_interval)
        live._initialize(context["gaussians"])
        controls = prior.set_metal_start(live.problem.field, .25)
        write_json(output/"initialization_controls.json", controls)
        runtime = {name: context[name] for name in (*WRAPPERS, "live_feedback", "scene", "pipe", "background")}
        prior.install_varpro(runtime, plan)
    if digest(rng_state()) != before_rng:
        raise RuntimeError("RoadBlock C Probe preparation changed the training RNG")
    write_json(output/"priornorm_integration.json", dict(
        status="passed", profile="roadblock_c", next_iteration=BOUNDARY+1,
        after_update_20k_observations=True, phase1_steps=PHASE_1_STEPS,
        prior_normalization=plan["prior_normalization"], varpro=plan["varpro"],
        main_feedback="detached-confidence true-IR-GT MSE"))
    print("[ROADBLOCK C] PriorNorm Probe ready for step 20001", flush=True)


def log_fields(live, iteration):
    from utils.priornorm_probe import save_attribute_phase, save_varpro_diagnostics

    save_attribute_phase(live, iteration)
    save_varpro_diagnostics(live, iteration)
