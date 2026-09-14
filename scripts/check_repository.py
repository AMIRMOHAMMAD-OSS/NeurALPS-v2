"""Check repository metadata and local documentation links without network access."""
import argparse
import json
from pathlib import Path
import re
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', dest='json_path')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / 'pyproject.toml').read_text())
    import neuralps_v2
    errors = []
    if metadata['project']['version'] != neuralps_v2.__version__:
        errors.append('Package metadata and import version differ.')
    documents = sorted(root.rglob('*.md'))
    links = 0
    for document in documents:
        if any(part in {'build', 'dist', '.venv', '.git'} for part in document.parts):
            continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = target.split('#', 1)[0]
            if not target:
                continue
            links += 1
            if not (document.parent / target).exists():
                errors.append(f'{document.relative_to(root)}: missing target {target}')
    for path in (root / 'configs').glob('*.json'):
        if '.local.' not in path.name:
            json.loads(path.read_text())
    record = dict(local_links_checked=links, errors=errors,
                  package_version=neuralps_v2.__version__, repository_valid=not errors)
    if args.json_path:
        dest = Path(args.json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
