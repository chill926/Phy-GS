"""Re-use an existing run's cfg_args for a controlled opacity-offset ablation."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import train


def main():
    parser = ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--offset', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    output = Path(options.output)
    if output.exists():
        raise FileExistsError('Use a new output directory: ' + str(output))
    device = torch.device('cuda:0')
    defaults_parser = ArgumentParser()
    model = train.ModelParams(defaults_parser, device)
    optimization = train.OptimizationParams(defaults_parser)
    pipeline = train.PipelineParams(defaults_parser)
    args = defaults_parser.parse_args([])
    # cfg_args is the repository's trusted argparse.Namespace checkpoint format.
    reference = eval((Path(options.reference) / 'cfg_args').read_text(),
                     {'Namespace': Namespace, 'device': torch.device, '__builtins__': {}})
    vars(args).update(vars(reference))
    args.model_path = str(output)
    args.load_iteration = 0
    args.load_model_path = ''
    args.data_device = device
    args.ir_opacity_offset = options.offset
    args.ir_opacity_offset_lr = .005
    args.ir_opacity_offset_reg_weight = .0001
    tests = [5000, 6000, 7000, 8000, 9000] + list(range(10000, 30001, 1000))
    saves = [7000, 10000, 20000, 30000]
    if options.smoke:
        args.iterations = 600
        args.densify_from_iter = 100
        args.densify_until_iter = 550
        args.opacity_reset_interval = 300
        args.phys_probe_every = 300
        args.phys_probe_steps = 8
        args.phys_decouple_steps = 8
        tests = [300, 600]
        saves = [600]
    elif args.iterations != 30000:
        raise ValueError('The full ablation requires a 30,000-iteration reference')
    print('IR_OPACITY_ABLATION reference={} offset={} smoke={}'.format(options.reference, options.offset, options.smoke))
    train.safe_state(False, device)
    train.training(model.extract(args), optimization.extract(args), pipeline.extract(args), tests, saves, device)
    print('IR_OPACITY_ABLATION_COMPLETE')


if __name__ == '__main__':
    main()
