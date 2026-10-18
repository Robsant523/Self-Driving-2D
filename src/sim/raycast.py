import torch

def cast(pos, heading, obstacles, offsets, max_range, width=None):
    angles = heading[:, None] + offsets[None, :]
    directions = torch.stack((angles.cos(), angles.sin()), -1)
    rel = obstacles[:, None, :, :2] - pos[:, None, None, :]
    projection = (rel * directions[:, :, None, :]).sum(-1)
    distance2 = rel.square().sum(-1)
    radius2 = obstacles[:, None, :, 2].square()
    discriminant = radius2 - (distance2 - projection.square())
    root = discriminant.clamp_min(0).sqrt()
    near = projection - root
    valid = (discriminant >= 0) & (projection + root >= 0)
    hits = torch.where(valid, near.clamp_min(0), max_range)
    distances = hits.amin(-1) if obstacles.shape[1] else torch.full_like(angles, max_range)
    if width is not None:
        dy = directions[..., 1]
        boundary = torch.where(dy >= 0, width / 2, -width / 2)
        walls = (boundary - pos[:, 1, None]) / torch.where(dy.abs() > 1e-7, dy, torch.ones_like(dy))
        walls = torch.where((dy.abs() > 1e-7) & (walls >= 0), walls, max_range)
        distances = torch.minimum(distances, walls)
    return distances.clamp(0, max_range)
