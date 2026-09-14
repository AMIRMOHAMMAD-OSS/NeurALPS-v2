"""Run the repository's numerical and neural contracts against the installed package."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-torch', action='store_true')
    parser.add_argument('--json', dest='json_path')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # This adds the test package, not src: imports must use an installed NeurALPS.
    sys.path.insert(0, str(root))
    try:
        import neuralps_v2
    except ImportError:
        parser.error('Install the project first: python -m pip install -e .')
    files = sorted(p for folder in ('src', 'tests', 'scripts', 'examples')
                   for p in (root / folder).rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(root)))
    suite = unittest.defaultTestLoader.discover(
        str(root / 'tests'), pattern='test_*.py', top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    skipped = len(result.skipped)
    record = dict(
        package_version=neuralps_v2.__version__, design_version=neuralps_v2.DESIGN_VERSION,
        discovered_tests=result.testsRun, executed_tests=result.testsRun-skipped,
        passed_tests=result.testsRun-skipped-len(result.errors)-len(result.failures),
        skipped_tests=skipped, failed_tests=len(result.failures), error_tests=len(result.errors),
        syntax_files=len(files), syntax_passed=True,
        torch_available=importlib.util.find_spec('torch') is not None,
        skipped=[dict(test=str(t), reason=reason) for t, reason in result.skipped],
    )
    if args.json_path:
        dest = Path(args.json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    if args.require_torch and (not record['torch_available'] or skipped):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
