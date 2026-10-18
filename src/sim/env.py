import torch
from .vehicle import advance
from .world import spawn, collision
from .raycast import cast
from ..common import device

class BatchedDrivingEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = device(cfg["device"])
        self.n = cfg["num_envs"]
        self.obs_dim = cfg["rays"] + 3
        self.offsets = torch.linspace(-cfg["ray_arc"] / 2, cfg["ray_arc"] / 2, cfg["rays"], device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg["seed"])
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.generator.manual_seed(seed)
        self.pos, self.obstacles = spawn(self.n, self.cfg, self.device, self.generator)
        self.heading = torch.zeros(self.n, device=self.device)
        self.speed = torch.zeros_like(self.heading)
        self.steps = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.last_action = torch.zeros((self.n, 2), device=self.device)
        return self.observe()

    def rays(self):
        return cast(self.pos, self.heading, self.obstacles, self.offsets, self.cfg["ray_range"], self.cfg["width"])

    def observe(self):
        return torch.cat((self.rays() / self.cfg["ray_range"], self.speed[:, None] / self.cfg["max_speed"], self.last_action), -1)

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.n, 2):
            raise ValueError(f"Expected {(self.n, 2)}, got {tuple(action.shape)}")
        action = action.to(device=self.device, dtype=torch.float32).clamp(-1, 1)
        old = self.pos
        self.pos, self.heading, self.speed = advance(old, self.heading, self.speed, action, self.cfg)
        self.steps += 1
        hit = collision(old, self.pos, self.obstacles, self.cfg)
        success = (self.pos[:, 0] >= self.cfg["length"]) & ~hit
        terminated = hit | success
        truncated = (self.steps >= self.cfg["max_steps"]) & ~terminated
        done = terminated | truncated
        r = self.cfg["reward"]
        reward = r["progress"] * (self.pos[:, 0] - old[:, 0]) + r["alive"] - r["jerk"] * (action[:, 0] - self.last_action[:, 0]).square() - r["collision"] * hit.float() + r["success"] * success.float()
        self.last_action = action
        final_observation = self.observe()
        info = {"collision": hit, "success": success, "final_observation": final_observation, "episode_steps": self.steps.clone()}
        # Full candidate batches keep the simulator free of .item(), .cpu(), and nonzero().
        pos, obstacles = spawn(self.n, self.cfg, self.device, self.generator)
        self.pos = torch.where(done[:, None], pos, self.pos)
        self.obstacles = torch.where(done[:, None, None], obstacles, self.obstacles)
        self.heading = torch.where(done, 0., self.heading)
        self.speed = torch.where(done, 0., self.speed)
        self.steps = torch.where(done, 0, self.steps)
        self.last_action = torch.where(done[:, None], 0., self.last_action)
        return self.observe(), reward, terminated, truncated, info
