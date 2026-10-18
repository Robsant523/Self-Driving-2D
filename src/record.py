import argparse
from pathlib import Path
import imageio.v2 as imageio
import torch
from .common import config
from .evaluate import load_policy
from .sim.env import BatchedDrivingEnv
from .sim.render import Renderer
from .rl.networks import ActorCritic

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cars", type=int, default=4)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--random", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/demo.mp4"))
    args = p.parse_args()
    if not 1 <= args.cars <= 16 or args.frames < 1: p.error("Use 1–16 cars and a positive frame count")
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    cfg = config(args.config); cfg["device"] = args.device
    payload, cfg = load_policy(args.checkpoint, cfg)
    torch.manual_seed(cfg["seed"])
    cfg.update(num_envs=args.cars, seed=args.seed)
    env = BatchedDrivingEnv(cfg)
    model = ActorCritic(env.obs_dim, cfg["ppo"]["hidden"]).to(env.device)
    if payload: model.load_state_dict(payload["model"])
    renderer = Renderer(args.cars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    obs = env.observe()
    label = "random" if args.random else "trained" if payload else "untrained"
    with imageio.get_writer(args.output, fps=round(1/cfg["dt"])) as writer:
        for _ in range(args.frames):
            writer.append_data(renderer.frame(env, label))
            action = torch.rand((env.n,2),device=env.device)*2-1 if args.random else model.act(obs, deterministic=True)[0]
            obs = env.step(action)[0]
    renderer.close()
    print(args.output)

if __name__ == "__main__": main()
