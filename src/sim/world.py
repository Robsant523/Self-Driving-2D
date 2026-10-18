import torch

def spawn(n, cfg, device, generator, road=None):
    """Sample full batches to avoid GPU-to-host synchronization on reset masks."""
    pos = torch.zeros((n, 2), device=device)
    pos[:, 0] = 2.0
    pos[:, 1] = (torch.rand(n, device=device, generator=generator) - 0.5) * 2
    if "fixed_obstacles" in cfg:
        obstacles = road.fixed_obstacles.expand(n, -1, -1).clone()
    else:
        k = cfg["obstacles"]
        u = torch.rand((n, k, 3), device=device, generator=generator)
        obstacles = torch.empty_like(u)
        obstacles[..., 0] = 10 + u[..., 0] * (cfg["length"] - 15)
        obstacles[..., 1] = (u[..., 1] - 0.5) * (cfg["width"] - 3)
        obstacles[..., 2] = 0.4 + u[..., 2] * 0.6
        if road is not None:
            obstacles[..., 1] += road.sample(obstacles[..., 0])[0]
    if road is not None:
        center, tangent, _ = road.sample(pos[:, 0])
        clearance = (cfg["width"]/2 - cfg["car_radius"]/tangent.cos().clamp_min(.1) - .1).clamp(0, 1)
        pos[:, 1] = center + pos[:, 1] * clearance
    return pos, obstacles

def collision(start, end, obstacles, cfg, road=None):
    """Swept circle versus circles avoids tunneling at high speeds."""
    delta = end - start
    rel = obstacles[..., :2] - start[:, None, :]
    t = (rel * delta[:, None, :]).sum(-1) / delta.square().sum(-1)[:, None].clamp_min(1e-8)
    nearest = start[:, None, :] + t.clamp(0, 1)[..., None] * delta[:, None, :]
    hit = ((nearest - obstacles[..., :2]).square().sum(-1) <= (obstacles[..., 2] + cfg["car_radius"]).square()).any(-1)
    if road is not None:
        return hit | road.collision(start, end, cfg["car_radius"])
    wall = (end[:, 1].abs() + cfg["car_radius"] >= cfg["width"] / 2) | (end[:, 0] < cfg["car_radius"])
    return hit | wall
