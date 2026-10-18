import math
import torch
from torch import nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def distribution(self, obs):
        features = self.body(obs)
        return Normal(self.actor(features), self.log_std.clamp(-5, 2).exp(), validate_args=False), self.critic(features).squeeze(-1)

    def act(self, obs, deterministic=False):
        dist, value = self.distribution(obs)
        raw = dist.mean if deterministic else dist.sample()
        return raw.tanh(), raw, self.log_prob(dist, raw), value

    @staticmethod
    def log_prob(dist, raw):
        # Stable tanh Jacobian; retain latent actions to avoid inverse-tanh rounding.
        correction = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
        return (dist.log_prob(raw) - correction).sum(-1)

    def evaluate(self, obs, raw):
        dist, value = self.distribution(obs)
        # Entropy estimate of the squashed distribution with reparameterized samples.
        sample = dist.rsample()
        return self.log_prob(dist, raw), -self.log_prob(dist, sample), value
