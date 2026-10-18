import argparse
from pathlib import Path
import imageio.v2 as imageio
import torch
from .tracks import DEFAULT_TRACK, apply_track
from .common import ROOT, config
from .evaluate import load_policy
from .sim.env import BatchedDrivingEnv
from .sim.render import Renderer
from .sim.road import follow_road
from .rl.networks import ActorCritic

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config")
    p.add_argument("--track", type=Path, help="YAML road points and fixed obstacles")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cars", type=int, default=4)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seed", type=int, default=10000)
    controls = p.add_mutually_exclusive_group()
    controls.add_argument("--random", action="store_true")
    controls.add_argument("--road-controller", action="store_true", help="Pure-pursuit reference driver, not PPO")
    p.add_argument("--camera", choices=("follow", "overview"), default="follow")
    p.add_argument("--rays", action=argparse.BooleanOptionalAction, default=True, help="Show sensor and road-preview arrows")
    p.add_argument("--output", type=Path, default=Path("results/demo.mp4"))
    args = p.parse_args()
    if not 1 <= args.cars <= 16 or args.frames < 1: p.error("Use 1–16 cars and a positive frame count")
    if not args.config and not args.track and not args.checkpoint:
        args.track = DEFAULT_TRACK
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    cfg = config(args.config or (ROOT / "configs/bendy.yaml" if args.track else None)); cfg["device"] = args.device
    payload, cfg = load_policy(args.checkpoint, cfg)
    if args.track:
        cfg = apply_track(cfg, args.track)
    torch.manual_seed(cfg["seed"])
    cfg.update(num_envs=args.cars, seed=args.seed)
    env = BatchedDrivingEnv(cfg)
    model = ActorCritic(env.obs_dim, cfg["ppo"]["hidden"]).to(env.device)
    if payload: model.load_state_dict(payload["model"])
    if args.road_controller and env.road is None:
        p.error("--road-controller requires --config configs/bendy.yaml or --track")
    if args.checkpoint and (args.random or args.road_controller):
        p.error("Choose a checkpoint OR a reference/random controller")
    renderer = Renderer(args.cars, show_rays=args.rays, camera=args.camera)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    obs = env.observe()
    label = "reference controller (not PPO)" if args.road_controller else "random" if args.random else "trained PPO" if payload else "untrained PPO"
    with imageio.get_writer(args.output, fps=round(1/cfg["dt"])) as writer:
        for _ in range(args.frames):
            writer.append_data(renderer.frame(env, label))
            action = torch.rand((env.n,2),device=env.device)*2-1 if args.random else model.act(obs, deterministic=True)[0]
            if args.road_controller:
                action = follow_road(env)
            obs, _, _, _, info = env.step(action)
            renderer.capture(info)
    renderer.close()
    print(args.output)

if __name__ == "__main__": main()
