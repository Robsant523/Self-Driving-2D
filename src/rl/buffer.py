import torch

def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]).float() - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])).float() * carry
        advantages[t] = carry
    return advantages, advantages + values

class RolloutBuffer:
    def __init__(self, t, n, obs_dim, device):
        self.obs = torch.empty((t, n, obs_dim), device=device)
        self.raw = torch.empty((t, n, 2), device=device)
        for name in ("log_probs", "values", "next_values", "rewards"):
            setattr(self, name, torch.empty((t, n), device=device))
        for name in ("terminated", "truncated"):
            setattr(self, name, torch.empty((t, n), device=device, dtype=torch.bool))
