"""Pinned scientific and numerical contracts for this release."""
import hashlib
import json
from pathlib import Path

SSL_SHA = "9599242bed3ffd7118593874cfe6462b30d1bae97dde1c8b6131d6fa44f70e0f"
ESMC_SHA = "e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8"
TARGETS_SHA = "7d81ce504a6b64e6b1c5cf2e71a639dde1b05ec4b2787e655dd235bad536961f"
FORK_COMMIT = "ef32577f55da19a4989cd7b22e004dc43a4998cb"
ESMC_REPO = "biohub/ESMC-600M"
ESMC_REVISION = "a7e82012c83126b9eedb055fea9fa84b6c02f094"
FEATURE_SCHEMA = "typed_context_2822_v1"
WINDOW_POLICY = "physical_end_extradomain_tail64_v1"
SOLVER = dict(maxiter=1000, retry_maxiter=4000, ftol=1e-12, gtol=1e-7,
              acceptable_gradient_inf=2e-5, l2_grid=[.01, .1, 1., 10., 100.])

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def seqsha(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")

def protocol():
    return {"solver": dict(SOLVER)}
