import torch
from .buffer import gae

def update(model, optimizer, buffer, cfg):
    adv, returns = gae(buffer.rewards, buffer.values, buffer.next_values, buffer.terminated, buffer.truncated, cfg["gamma"], cfg["gae_lambda"])
    adv = adv.flatten()
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    obs = buffer.obs.flatten(0, 1)
    raw = buffer.raw.flatten(0, 1)
    old = buffer.log_probs.flatten()
    returns = returns.flatten()
    totals = torch.zeros(4, device=obs.device)
    batches = 0
    for _ in range(cfg["epochs"]):
        order = torch.randperm(len(adv), device=obs.device)
        for ids in order.split(cfg["minibatch"]):
            logp, entropy, value = model.evaluate(obs[ids], raw[ids])
            ratio = (logp - old[ids]).exp()
            policy = -torch.minimum(ratio * adv[ids], ratio.clamp(1-cfg["clip"], 1+cfg["clip"]) * adv[ids]).mean()
            value_loss = (value - returns[ids]).square().mean()
            loss = policy + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            totals += torch.stack((policy.detach(), value_loss.detach(), entropy.mean().detach(), ((ratio - 1) - (logp - old[ids])).mean().detach()))
            batches += 1
    return dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl"), (totals / batches).cpu().tolist()))
