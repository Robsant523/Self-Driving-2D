import argparse
import csv
import platform
import statistics
import time
from pathlib import Path
import torch
from .common import config, sync
from .sim.env import BatchedDrivingEnv

def measure(cfg, mode, steps, repeats):
    n = cfg["num_envs"]
    envs = [BatchedDrivingEnv({**cfg, "num_envs": 1, "seed": cfg["seed"]+i}) for i in range(n)] if mode == "python_loop" else [BatchedDrivingEnv(cfg)]
    actions = [torch.zeros((e.n,2),device=e.device) for e in envs]
    for a in actions: a[:,1] = .5
    def step():
        for e,a in zip(envs,actions): e.step(a)
    for _ in range(3): step()
    rates=[]
    for _ in range(repeats):
        sync(envs[0].device); start=time.perf_counter()
        for _ in range(steps): step()
        sync(envs[0].device)
        rates.append(n*steps/(time.perf_counter()-start))
    return {"mode":mode,"device":cfg["device"],"num_envs":n,"median_transitions_per_second":statistics.median(rates),"min_transitions_per_second":min(rates),"max_transitions_per_second":max(rates),"steps":steps,"repeats":repeats,"torch":torch.__version__,"platform":platform.platform(),"threads":torch.get_num_threads()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sizes",nargs="+",type=int,default=[1,64,1024,8192])
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--skip-loop",action="store_true")
    p.add_argument("--output",type=Path,default=Path("results/benchmark.csv"))
    a=p.parse_args()
    if min(a.sizes+[a.steps,a.repeats]) < 1: p.error("Sizes, steps and repeats must be positive")
    torch.set_num_threads(4)
    modes = ([] if a.skip_loop else [("python_loop","cpu")]) + [("vectorized","cpu")]
    if torch.backends.mps.is_available(): modes.append(("vectorized","mps"))
    if torch.cuda.is_available(): modes.append(("vectorized","cuda"))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w") as f:
        writer=None
        for n in a.sizes:
            for mode,dev in modes:
                row=measure({**config(),"num_envs":n,"device":dev},mode,a.steps,a.repeats)
                if writer is None: writer=csv.DictWriter(f,fieldnames=row);writer.writeheader()
                writer.writerow(row);f.flush();print(row,flush=True)

if __name__ == "__main__": main()
