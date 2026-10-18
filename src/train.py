import argparse
import csv
import time
from pathlib import Path
import torch
from .tracks import apply_track
from .common import ROOT, config, sync
from .sim.env import BatchedDrivingEnv
from .rl.networks import ActorCritic
from .rl.buffer import RolloutBuffer
from .rl.ppo import update

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--track", type=Path, help="YAML road points and fixed obstacles")
    p.add_argument("--device", default=None)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--updates", type=int)
    p.add_argument("--obstacles", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/train"))
    args = p.parse_args()
    cfg = config(args.config or (ROOT / "configs/bendy.yaml" if args.track else None))
    for key in ("device", "num_envs", "obstacles"):
        if getattr(args, key) is not None: cfg[key] = getattr(args, key)
    if args.updates is not None: cfg["ppo"]["updates"] = args.updates
    if args.track:
        if args.obstacles is not None: p.error("Edit obstacles in the track file instead of using --obstacles")
        cfg = apply_track(cfg, args.track)
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    env = BatchedDrivingEnv(cfg)
    model = ActorCritic(env.obs_dim, cfg["ppo"]["hidden"]).to(env.device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=env.device, weights_only=True)["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["ppo"]["lr"])
    b = RolloutBuffer(cfg["ppo"]["rollout"], env.n, env.obs_dim, env.device)
    obs = env.observe()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w") as f:
        writer = None
        for iteration in range(cfg["ppo"]["updates"]):
            sync(env.device)
            start = time.perf_counter()
            counts = torch.zeros(3, device=env.device)
            with torch.no_grad():
                for t in range(cfg["ppo"]["rollout"]):
                    b.obs[t] = obs
                    action, b.raw[t], b.log_probs[t], b.values[t] = model.act(obs)
                    obs, b.rewards[t], b.terminated[t], b.truncated[t], info = env.step(action)
                    # Read pre-reset observation for timeout value bootstrapping.
                    b.next_values[t] = model.distribution(info["final_observation"])[1]
                    counts += torch.stack((info["collision"].sum(), info["success"].sum(), (b.terminated[t] | b.truncated[t]).sum()))
            metrics = update(model, optimizer, b, cfg["ppo"])
            sync(env.device)
            c, s, episodes = counts.cpu().tolist()
            row = {"update": iteration + 1, "transitions": (iteration+1)*b.rewards.numel(), "mean_step_reward": b.rewards.mean().item(), "completed_episodes": episodes, "collisions": c, "successes": s, "transitions_per_second": b.rewards.numel()/(time.perf_counter()-start), **metrics}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row)
                writer.writeheader()
            writer.writerow(row); f.flush()
            print(row, flush=True)
            torch.save({"model": model.state_dict(), "config": cfg, "update": iteration+1}, args.output / "checkpoint.pt")
    from .plot import plot
    plot(args.output / "metrics.csv", args.output / "learning.png")

if __name__ == "__main__": main()
