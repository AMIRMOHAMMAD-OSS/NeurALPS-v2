"""Check configuration/declared paths; never claim production data validation."""
import argparse
import json
from pathlib import Path


def validate_config(c):
    if c.get('design_version') != 'phase2.2':
        raise ValueError('Expected phase2.2 config.')
    if c['inputs']['module_annotations_as_neural_inputs']:
        raise ValueError('Module annotations cannot be neural inputs.')
    if not c['inputs']['plm_frozen'] or c['inputs']['plm_width'] != 1152:
        raise ValueError('Initial reference requires frozen ESM-C 1152 features.')
    local = c['local']
    required = dict(width=256, heads=8, blocks=4, radius_interleaved=6, dropout=0.1)
    if any(local[k] != v for k,v in required.items()):
        raise ValueError('Unsupported reference local configuration.')
    if (c['global']['blocks'], c['global']['slots'], c['recycling']['passes']) != (2,4,2):
        raise ValueError('Unsupported global/recycle reference configuration.')
    if c['activity']['features'] != 770:
        raise ValueError('Common activity head requires 770 input features.')
    if c['inputs']['primary_track'] != 'construct_only':
        raise ValueError('Source-context wrapper is not implemented in this reference.')
    return True


def validate_registry(registry):
    stages = registry['stages']; known = {s['id'] for s in stages}
    if len(known) != len(stages):
        raise ValueError('Duplicate stage IDs.')
    completed = set()
    while len(completed) < len(known):
        eligible = {s['id'] for s in stages if set(s['requires']) <= completed}
        updated = completed | eligible
        if updated == completed:
            raise ValueError('Stage dependencies contain a cycle or missing stage.')
        completed = updated
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--paths')
    args = parser.parse_args()
    config_path = Path(args.config)
    validate_config(json.loads(config_path.read_text()))
    validate_registry(json.loads((config_path.parent/'experiment_registry.json').read_text()))
    missing = []
    if args.paths:
        path_file = Path(args.paths).resolve()
        paths = json.loads(path_file.read_text())
        if paths.get('design_version') != 'phase2.2':
            raise ValueError('Path manifest version mismatch.')
        for key, spec in paths['paths'].items():
            value = spec['path']
            if spec['required'] and (not value or not (path_file.parent/Path(value)).is_file()):
                missing.append(key)
    print(json.dumps(dict(package_config_valid=True, path_check_requested=bool(args.paths),
                         missing_required_paths=missing,
                         production_data_validated=False, training_validated=False), indent=2))
    return 2 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
