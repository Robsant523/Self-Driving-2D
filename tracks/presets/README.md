# Preset roads

**Long Winding is the default course for new training runs.** It has ten deep, alternating bends over about 795 m of road, with a straight start and no obstacles. Its 101 sampled points keep the road usable with batched sensors. The road follows increasing x coordinates, so these are open S bends rather than closed or crossing loops.

Difficulty labels describe the course design, not measured PPO success rates. All courses work with the `--track` option, including the original five presets below.

| File | Course | Length | Width | Obstacles | Intended challenge |
|---|---|---:|---:|---:|---|
| `long_winding.yaml` | Long Winding **(default)** | 600 m | 12 m | 0 | Long; ten deep alternating bends |
| `gentle_s.yaml` | Gentle S | 144 m | 12 m | 0 | Easy; learn wide bends |
| `wide_sweep.yaml` | Wide Sweep | 144 m | 12 m | 3 | Easy–medium; sweeping turn and roadside obstacles |
| `obstacle_slalom.yaml` | Obstacle Slalom | 140 m | 12 m | 6 | Medium; alternate around blockers |
| `narrow_bends.yaml` | Narrow Bends | 144 m | 7 m | 0 | Medium–hard; precise steering through tighter bends |
| `mixed_course.yaml` | Mixed Course | 180 m | 10 m | 6 | Hard; bends and obstacles together |

Lengths are finish x coordinates, not centerline arc lengths; widths are vertical edge separation.

From the project directory, train the default course:

```bash
.venv/bin/python -m src.train --output results/long-winding
```

Record the resulting policy with the full course in view:

```bash
.venv/bin/python -m src.record --checkpoint results/long-winding/checkpoint.pt --camera overview --cars 1 --frames 1600 --output results/long-winding.mp4
open results/long-winding.mp4
```

Choose an earlier preset explicitly with `--track tracks/presets/gentle_s.yaml` and a matching output directory. To initialize from a bendy-road policy, add `--resume results/bendy/checkpoint.pt` to training. To inspect a course immediately, use `record --track ... --road-controller --camera overview`; the reference driver does not avoid obstacles, so use PPO for learned avoidance.

The original five presets were loaded on CPU and checked for valid geometry, collision-free initial placement, finite observations, and persistent obstacle positions after a masked reset. Their two obstacle-free courses were also checked with the reference steering controller.

Long Winding was checked with eight CPU environments: all eight reference-controller cars finished in 1,320 steps with no collisions or timeouts, using a 2,400-step episode limit. This verifies geometry and reference-controller feasibility; it does not establish learned PPO performance. Its longer episodes require a larger timeout than the short courses. A 1,600-frame reference recording is long enough to show a complete run at the controller’s target speed.
