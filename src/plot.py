import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot(source, output):
    with Path(source).open() as f: rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    x = [int(r["transitions"]) for r in rows]
    axes[0].plot(x, [float(r["mean_step_reward"]) for r in rows])
    axes[0].set(title="Training reward", xlabel="Transitions", ylabel="Mean reward per step")
    for key in ("collisions", "successes"):
        axes[1].plot(x, [float(r[key])/max(float(r["completed_episodes"]),1) for r in rows], label=key)
    axes[1].set(title="Completed training episodes", xlabel="Transitions", ylabel="Fraction", ylim=(0,1))
    axes[1].legend(); fig.tight_layout(); fig.savefig(output); plt.close(fig)
