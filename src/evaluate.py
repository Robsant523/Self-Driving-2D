import argparse
import json
from pathlib import Path
import torch
from .tracks import DEFAULT_TRACK, apply_track
from .common import ROOT, config
from .sim.env import BatchedDrivingEnv
from .rl.networks import ActorCritic

def load_policy(checkpoint, cfg):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
    if payload:
        cfg = {**payload["config"], "device": cfg["device"]}
    return payload, cfg

@torch.no_grad()
def evaluate(cfg, payload=None, episodes=500, seed=10000):
    if episodes < 1: raise ValueError("episodes must be positive")
    policy_seed = cfg["seed"]
    cfg = {**cfg, "num_envs": episodes, "seed": seed}
    env = BatchedDrivingEnv(cfg)
    torch.manual_seed(policy_seed)
    model = ActorCritic(env.obs_dim, cfg["ppo"]["hidden"]).to(env.device)
    if payload: model.load_state_dict(payload["model"])
    active = torch.ones(episodes, dtype=torch.bool, device=env.device)
    returns = torch.zeros(episodes, device=env.device)
    collisions = torch.zeros_like(active)
    successes = torch.zeros_like(active)
    lengths = torch.zeros(episodes, dtype=torch.int64, device=env.device)
    obs = env.observe()
    for _ in range(cfg["max_steps"]):
        action = model.act(obs, deterministic=True)[0]
        obs, reward, term, trunc, info = env.step(action)
        returns += reward * active
        lengths += active.long()
        collisions |= active & info["collision"]
        successes |= active & info["success"]
        active &= ~(term | trunc)
    # Each lane contributes exactly its first episode, avoiding fast-episode selection bias.
    return {"episodes": episodes, "seed": seed, "policy": "trained" if payload else "untrained", "collision_rate": collisions.float().mean().item(), "success_rate": successes.float().mean().item(), "timeout_rate": (~(collisions | successes)).float().mean().item(), "mean_return": returns.mean().item(), "mean_length": lengths.float().mean().item(), "device": str(env.device), "obstacles": cfg["obstacles"], "world": cfg.get("road", {}).get("kind", "corridor")}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config")
    p.add_argument("--track", type=Path, help="YAML road points and fixed obstacles")
    p.add_argument("--device", default="auto")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--obstacles", type=int)
    p.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = p.parse_args()
    if not args.config and not args.track and not args.checkpoint and args.obstacles is None:
        args.track = DEFAULT_TRACK
    torch.set_num_threads(4)
    cfg = config(args.config or (ROOT / "configs/bendy.yaml" if args.track else None)); cfg["device"] = args.device
    payload, cfg = load_policy(args.checkpoint, cfg)
    if args.track:
        cfg = apply_track(cfg, args.track)
    if args.obstacles is not None:
        if "fixed_obstacles" in cfg: p.error("Edit obstacles in the track file instead of using --obstacles")
        cfg["obstacles"] = args.obstacles
    result = evaluate(cfg, payload, args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
