#!/usr/bin/env python3
"""Compare the portable implementation against independently exported originals."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuralps.contracts import require, dump
from neuralps.scoring import Runtime, make_plan

def verify(assets, device="cpu"):
    import torch
    torch.set_num_threads(1)
    runtime = Runtime(assets, device)
    results = []
    for prefix in runtime.manifest["golden_fixtures"]:
        item = json.loads((assets/(prefix+".json")).read_text())
        record, parents = item["record"], item["parents"]
        with np.load(assets/(prefix+".npz"), allow_pickle=False) as z:
            states = z["object_states"]
            vectors = {k: z[k].copy() for k in z.files if k != "object_states"}
        fresh = runtime.object_states(record, vectors)
        require(np.allclose(fresh, states, atol=2e-5, rtol=2e-5), "Frozen states differ: "+prefix)
        plans = {}
        for name, original in item["expected"].items():
            plan = make_plan(record, original["plan"]["targets"], parents)
            require(plan["status"] == original["plan"]["status"], "Mask support status differs")
            require(sorted(map(tuple, plan["hidden"])) == sorted(map(tuple, original["plan"]["hidden"])),
                    "Physical masking closure differs")
            plans[name] = plan
        scored = runtime.score_plans(record, parents, vectors, plans)
        differences = []
        for name, original in item["expected"].items():
            for j, modes in original["objects"].items():
                for mode, value in modes.items():
                    diff = abs(scored[name]["objects"][j][mode]-value)
                    require(diff <= 2e-5, "Reconstruction differs: "+prefix+"/"+mode)
                    differences.append(diff)
        results.append(dict(fixture=prefix, max_state_error=float(np.abs(fresh-states).max()),
                            max_score_error=max(differences, default=0), status="PASS"))
    require(len(results) == 3, "Expected BODE1, BODE2 and T-domain golden fixtures")
    return dict(status="PASS", fixtures=results, device=device,
                scope="Encoder and reconstruction parity on cached inputs; ESMC GPU gate runs separately")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    result = verify(args.assets, args.device)
    if args.report:
        dump(args.report, result)
    print(json.dumps(result, indent=2))
