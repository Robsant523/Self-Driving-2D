# Five preset roads

Difficulty labels describe the course design, not measured PPO success rates. All courses work with the existing `--track` option.

| File | Course | Length | Width | Obstacles | Intended challenge |
|---|---|---:|---:|---:|---|
| `gentle_s.yaml` | Gentle S | 144 m | 12 m | 0 | Easy; learn wide bends |
| `wide_sweep.yaml` | Wide Sweep | 144 m | 12 m | 3 | Easy–medium; sweeping turn and roadside obstacles |
| `obstacle_slalom.yaml` | Obstacle Slalom | 140 m | 12 m | 6 | Medium; alternate around blockers |
| `narrow_bends.yaml` | Narrow Bends | 144 m | 7 m | 0 | Medium–hard; precise steering through tighter bends |
| `mixed_course.yaml` | Mixed Course | 180 m | 10 m | 6 | Hard; bends and obstacles together |

Lengths are finish x coordinates, not centerline arc lengths; widths are vertical edge separation.

From the project directory, train one preset:

```bash
.venv/bin/python -m src.train --track tracks/presets/gentle_s.yaml --output results/gentle-s
```

Record the resulting policy with the full course in view:

```bash
.venv/bin/python -m src.record --checkpoint results/gentle-s/checkpoint.pt --camera overview --cars 1 --output results/gentle-s.mp4
open results/gentle-s.mp4
```

Replace the YAML filename and output directory to use another preset. To initialize from a bendy-road policy, add `--resume results/bendy/checkpoint.pt` to training. To inspect a course immediately, use `record --track ... --road-controller --camera overview`; the reference driver does not avoid obstacles, so use PPO for learned avoidance.

All five presets were loaded on CPU and checked for valid geometry, collision-free initial placement, finite observations, and persistent obstacle positions after a masked reset. The two obstacle-free courses were also checked with the reference steering controller. These checks do not establish PPO performance on the new courses.
