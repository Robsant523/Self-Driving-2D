from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

def config(path=None):
    with open(path or ROOT / "configs/default.yaml") as f:
        return yaml.safe_load(f)

def device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)

def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize(dev)
