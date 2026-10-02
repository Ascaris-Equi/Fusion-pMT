"""Record, without changing, the Python runtime used for the AF2 pilot."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from run_af2_pilot import DEFAULT_DATA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    weight_file = args.data / "params/params_model_1_multimer_v3.npz"
    digest = hashlib.sha256()
    with weight_file.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    try:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"
        ], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    packages = sorted(
        ({"name": d.metadata["Name"], "version": d.version}
         for d in importlib.metadata.distributions()),
        key=lambda d: d["name"].lower(),
    )
    payload = {
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "packages": packages,
        "scope": "installed package snapshot, not a standalone conda environment lock",
        "model_weight": "params_model_1_multimer_v3.npz",
        "model_weight_sha256": digest.hexdigest(), "gpu": gpu,
    }
    output = Path(__file__).resolve().parent / "results/af2_runtime.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
