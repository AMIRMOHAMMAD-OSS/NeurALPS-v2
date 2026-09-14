"""Run one untrained A/C/D forward on artificial frozen-feature inputs."""
import argparse
import importlib.util
import json
from pathlib import Path

from synthetic_manifest import make_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('A', 'C', 'D'), default='A')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/phase2_final_config.json')
    args = parser.parse_args()
    if importlib.util.find_spec('torch') is None:
        parser.error('Install the model extra and a supported PyTorch build first.')
    import torch
    from neuralps_v2.manifest_adapter import collate, prepare_record
    from neuralps_v2.phase2_training import build_bundle, load_config, set_seed

    set_seed(20260914)
    torch.set_num_threads(2)
    record, store = make_example()
    batch = collate([prepare_record(record, store)])['batch']
    model = build_bundle(load_config(args.config), variant=args.variant).eval()
    with torch.inference_mode():
        output = model(batch)
    assert torch.isfinite(output['features']).all()
    assert torch.isfinite(output['activity_logit']).all()
    print(json.dumps(dict(
        status='synthetic software smoke check; untrained model', variant=args.variant,
        features_shape=list(output['features'].shape),
        logit_shape=list(output['activity_logit'].shape),
        passes=len(output['cycle_features']),
        parameters=sum(p.numel() for p in model.parameters()),
    ), indent=2))


if __name__ == '__main__':
    main()
