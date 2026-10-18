import torch

def advance(pos, heading, speed, action, cfg):
    """Semi-implicit bicycle integration; all floating tensors are float32."""
    steering = action[:, 0] * cfg["max_steer"]
    speed = (speed + action[:, 1] * cfg["max_accel"] * cfg["dt"]).clamp(0, cfg["max_speed"])
    heading = heading + speed / cfg["wheelbase"] * torch.tan(steering) * cfg["dt"]
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    return pos + speed[:, None] * direction * cfg["dt"], heading, speed
