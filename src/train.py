import argparse
import csv
import time
from pathlib import Path
import torch
from .tracks import DEFAULT_TRACK, apply_track
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
    p.add_argument("--live", action="store_true", help="Watch actual training in a local browser")
    p.add_argument("--live-cars", type=int, default=8, help="Number of actual trial cars to display, 1 to 16")
    p.add_argument("--camera", choices=("follow", "overview"), default="overview")
    p.add_argument("--live-every", type=int, default=4, help="Show a frame every N simulator steps")
    p.add_argument("--live-fps", type=int, default=10, help="Maximum live frames/sec; paces watched training")
    p.add_argument("--live-port", type=int, default=0, help="Local port; 0 chooses a free port")
    p.add_argument("--no-browser", action="store_true", help="Print live URL without opening it")
    p.add_argument("--live-exit", action="store_true", help="Exit immediately when training finishes")
    p.add_argument("--output", type=Path, default=Path("results/train"))
    args = p.parse_args()
    if not 1 <= args.live_cars <= 16:
        p.error("--live-cars must be from 1 to 16")
    if args.live_every < 1 or not 1 <= args.live_fps <= 30 or not 0 <= args.live_port <= 65535:
        p.error("Use positive --live-every, --live-fps from 1 to 30, and a valid port")
    if not args.config and not args.track and args.obstacles is None:
        args.track = DEFAULT_TRACK
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
    live = None
    completed_updates = 0
    if args.live:
        from .live import LiveTrainingView
        live = LiveTrainingView(camera=args.camera, fps=args.live_fps, port=args.live_port,
                                open_browser=not args.no_browser, cars=min(args.live_cars, env.n))
    try:
        if live:
            live.frame(env, pace=False, phase="Starting", update=0, completed_updates=0,
                       total_updates=cfg["ppo"]["updates"], transitions=0, num_envs=env.n)
        with (args.output / "metrics.csv").open("w") as f:
            writer = None
            for iteration in range(cfg["ppo"]["updates"]):
                if live:
                    live.status(phase="Collecting experience", update=iteration+1)
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
                        if live:
                            live.capture(info)
                        if live and (t+1) % args.live_every == 0:
                            live.frame(env, transitions=(iteration*cfg["ppo"]["rollout"]+t+1)*env.n)
                if live:
                    live.status(phase="Updating policy")
                metrics = update(model, optimizer, b, cfg["ppo"])
                sync(env.device)
                c, s, episodes = counts.cpu().tolist()
                row = {"update": iteration + 1, "transitions": (iteration+1)*b.rewards.numel(), "mean_step_reward": b.rewards.mean().item(), "completed_episodes": episodes, "collisions": c, "successes": s, "transitions_per_second": b.rewards.numel()/(time.perf_counter()-start), **metrics}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row); f.flush()
                print(row, flush=True)
                completed_updates = iteration+1
                if live:
                    live.status(metrics=row, completed_updates=completed_updates, transitions=row["transitions"])
                torch.save({"model": model.state_dict(), "config": cfg, "update": iteration+1}, args.output / "checkpoint.pt")
        from .plot import plot
        plot(args.output / "metrics.csv", args.output / "learning.png")
        if live:
            live.frame(env, pace=False, phase="Training complete", completed_updates=completed_updates)
            if not args.live_exit:
                live.wait(env)
    except KeyboardInterrupt:
        torch.save({"model": model.state_dict(), "config": cfg,
                    "update": completed_updates}, args.output / "checkpoint.pt")
        print("Stopped. Current policy saved to", args.output / "checkpoint.pt", flush=True)
    finally:
        if live:
            live.close()

if __name__ == "__main__": main()
