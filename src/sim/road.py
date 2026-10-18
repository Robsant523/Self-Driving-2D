"""Shared piecewise-linear geometry for sensing, collisions, rewards and rendering."""
import math
import torch


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_segment_distance2(p, a, b):
    delta = b - a
    t = ((p - a) * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-8)
    return (p - (a + t.clamp(0, 1)[..., None] * delta)).square().sum(-1)


class Road:
    def __init__(self, cfg, device):
        settings = cfg['road']
        self.fixed_obstacles = torch.tensor(cfg.get('fixed_obstacles', []), dtype=torch.float32, device=device).reshape(1, -1, 3)
        self.width = cfg['width']
        self.length = cfg['length']
        self.custom = settings['kind'] == 'custom'
        margin = cfg['ray_range'] + 5
        if self.custom:
            points = torch.tensor(settings['points'], dtype=torch.float32, device=device)
            # Constant extensions allow sensors to look past the start and finish.
            self.x = torch.cat((points[:1, 0]-margin, points[:, 0], points[-1:, 0]+margin))
            self.y = torch.cat((points[:1, 1], points[:, 1], points[-1:, 1]))
        else:
            spacing = settings.get('spacing', 2.0)
            wavelength = settings.get('wavelength', 70.0)
            amplitude = settings.get('amplitude', 7.0)
            if spacing <= 0 or wavelength <= 0 or self.width <= 2 * cfg['car_radius']:
                raise ValueError('Road requires positive spacing/wavelength and room for the car')
            self.x = torch.linspace(-margin, self.length + margin,
                                    math.ceil((self.length + 2 * margin) / spacing) + 1,
                                    device=device)
            self.y = amplitude * torch.sin(2 * math.pi * self.x / wavelength)
        self.dx = self.x[1] - self.x[0]
        center = torch.stack((self.x, self.y), -1)
        self.lower = center + torch.tensor([0., -self.width / 2], device=device)
        self.upper = center + torch.tensor([0., self.width / 2], device=device)
        self.a = torch.cat((self.lower[:-1], self.upper[:-1]))
        self.b = torch.cat((self.lower[1:], self.upper[1:]))
        lengths = (center[1:] - center[:-1]).norm(dim=-1)
        self.arc = torch.cat((torch.zeros(1, device=device), lengths.cumsum(0)))

    def sample(self, x):
        if self.custom:
            index = (torch.searchsorted(self.x, x.contiguous(), right=True) - 1).clamp(0, len(self.x)-2)
        else:
            index = ((x - self.x[0]) / self.dx).floor().long().clamp(0, len(self.x) - 2)
        dx = self.x[index+1] - self.x[index]
        fraction = ((x - self.x[index]) / dx).clamp(0, 1)
        y = self.y[index] + fraction * (self.y[index + 1] - self.y[index])
        tangent = torch.atan2(self.y[index + 1] - self.y[index], dx)
        progress = self.arc[index] + fraction * (self.arc[index + 1] - self.arc[index])
        return y, tangent, progress

    def features(self, pos, heading):
        center, tangent, _ = self.sample(pos[:, 0])
        error = (pos[:, 1] - center) / (self.width / 2)
        angle = tangent - heading
        lookahead = torch.tensor([5., 12., 20.], device=pos.device)
        target_x = pos[:, :1] + lookahead
        target_y = self.sample(target_x)[0]
        bearing = torch.atan2(target_y - pos[:, 1, None], lookahead) - heading[:, None]
        bearing = torch.atan2(bearing.sin(), bearing.cos()) / math.pi
        return torch.cat((error[:, None], angle.sin()[:, None], angle.cos()[:, None], bearing), -1)

    def rays(self, pos, heading, offsets, max_range):
        angles = heading[:, None] + offsets
        direction = torch.stack((angles.cos(), angles.sin()), -1)[:, :, None, :]
        segment = self.b - self.a
        relative = self.a[None, None] - pos[:, None, None]
        denominator = cross(direction, segment)
        safe = torch.where(denominator.abs() > 1e-7, denominator, 1.)
        distance = cross(relative, segment) / safe
        fraction = cross(relative, direction) / safe
        valid = (denominator.abs() > 1e-7) & (distance >= 0) & (fraction >= 0) & (fraction <= 1)
        return torch.where(valid, distance, max_range).amin(-1).clamp(0, max_range)

    def collision(self, start, end, radius):
        # Minimum distance between a swept center segment and each road-edge segment.
        p, q = start[:, None], end[:, None]
        a, b = self.a[None], self.b[None]
        distance2 = torch.minimum(point_segment_distance2(p, a, b), point_segment_distance2(q, a, b))
        distance2 = torch.minimum(distance2, point_segment_distance2(a, p, q))
        distance2 = torch.minimum(distance2, point_segment_distance2(b, p, q))
        denominator = cross(q - p, b - a)
        safe = torch.where(denominator.abs() > 1e-8, denominator, 1.)
        t = cross(a - p, b - a) / safe
        u = cross(a - p, q - p) / safe
        crossing = (denominator.abs() > 1e-8) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        outside = (end[:, 1] - self.sample(end[:, 0])[0]).abs() >= self.width / 2
        return ((distance2 <= radius ** 2) | crossing).any(-1) | outside | (end[:, 0] < radius)


def follow_road(env):
    """Pure-pursuit reference controller, NOT a learned policy; requires a road map."""
    if env.road is None:
        raise ValueError('Road controller requires a bendy or custom-road configuration')
    lookahead = 5.0 + .3 * env.speed
    target_y = env.road.sample(env.pos[:, 0] + lookahead)[0]
    dy = target_y - env.pos[:, 1]
    alpha = torch.atan2(dy, lookahead) - env.heading
    distance = (lookahead.square() + dy.square()).sqrt()
    steer = torch.atan2(2 * env.cfg['wheelbase'] * alpha.sin(), distance)
    acceleration = (6.0 - env.speed) / env.cfg['max_accel']
    return torch.stack((steer / env.cfg['max_steer'], acceleration), -1).clamp(-1, 1)
