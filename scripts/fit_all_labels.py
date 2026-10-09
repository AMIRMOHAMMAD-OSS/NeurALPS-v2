#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuralps.training import train
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Refit the selected frozen-feature head on all 494 reviewed labels")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = train(ROOT, args.output)
    print("Trained", result["trained_rows"], "rows:", result["by_dataset"])
    print("Saved", args.output.resolve())
