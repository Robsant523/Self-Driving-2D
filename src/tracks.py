"""Load editable YAML courses and embed their geometry in checkpoint configs."""
from copy import deepcopy
import math
from pathlib import Path
import yaml

DEFAULT_TRACK = Path(__file__).resolve().parents[1] / "tracks/presets/long_winding.yaml"


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def apply_track(cfg, path):
    with Path(path).open() as stream:
        track = yaml.safe_load(stream)
    if not isinstance(track, dict):
        raise ValueError('Track must be a YAML mapping with width, points and obstacles')
    unknown = set(track) - {'name', 'width', 'points', 'obstacles'}
    if unknown:
        raise ValueError(f'Unknown track fields: {sorted(unknown)}')
    width = number(track.get('width'), 'width')
    points = track.get('points')
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError('points must contain at least two [x, y] pairs')
    parsed = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Each road point must be [x, y]')
        parsed.append([number(point[0], 'point x'), number(point[1], 'point y')])
    if parsed[0][0] != 0 or parsed[-1][0] < 20:
        raise ValueError('Road must start at x=0 and finish at x>=20')
    for a, b in zip(parsed, parsed[1:]):
        if b[0] <= a[0]:
            raise ValueError('Road point x values must strictly increase; loops are not supported')
        slope = (b[1]-a[1])/(b[0]-a[0])
        if width / math.sqrt(1+slope*slope) <= 2*cfg['car_radius'] + .2:
            raise ValueError('Road is too narrow or steep for the car footprint')
    obstacles = track.get('obstacles', [])
    if not isinstance(obstacles, list):
        raise ValueError('obstacles must be a list of [x, y, radius] triples')
    fixed = []
    for obstacle in obstacles:
        if not isinstance(obstacle, list) or len(obstacle) != 3:
            raise ValueError('Each obstacle must be [x, y, radius]')
        x,y,r = [number(v, 'obstacle coordinate/radius') for v in obstacle]
        if r <= 0 or x-r < 8 or x+r >= parsed[-1][0]:
            raise ValueError('Obstacle radius must be positive; keep obstacles after x=8 and before the finish')
        for a,b in zip(parsed,parsed[1:]):
            if a[0] <= x <= b[0]:
                cy = a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])
                if abs(y-cy)+r >= width/2:
                    raise ValueError('Obstacle must fit inside the road')
                break
        # A circle must also clear sloping edges and corners.
        for a,b in zip(parsed,parsed[1:]):
            for offset in (-width/2,width/2):
                dx,dy=b[0]-a[0],b[1]-a[1]
                t=max(0.,min(1.,((x-a[0])*dx+(y-a[1]-offset)*dy)/(dx*dx+dy*dy)))
                if math.hypot(x-a[0]-t*dx,y-a[1]-offset-t*dy) <= r:
                    raise ValueError('Obstacle touches a road edge')
        fixed.append([x,y,r])
    result=deepcopy(cfg)
    result.update(width=width,length=parsed[-1][0],obstacles=len(fixed),fixed_obstacles=fixed)
    centerline_length = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(parsed, parsed[1:]))
    # Long courses need enough time to finish at a moderate fraction of max speed.
    result['max_steps'] = max(result['max_steps'], math.ceil(centerline_length / (cfg['max_speed'] * .5 * cfg['dt'])) + 50)
    result['road']={'kind':'custom','points':parsed,'name':str(track.get('name','Custom course'))}
    return result
