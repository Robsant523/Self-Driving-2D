# Batched Driving RL

This is a simplified 2D control experiment, not a photorealistic simulator or an autonomous-driving safety system. Results describe this corridor task only.

## Multiple trial cars

Watch several trials drive simultaneously while every training environment contributes experience to the same PPO model:

```bash
.venv/bin/python -m src.train --num-envs 512 --live --live-cars 8 --output results/multi-trial-live
```

`--num-envs` controls how many trials train in parallel; `--live-cars` controls how many of those real trials you see. They share policy weights and pool their rollouts for each update. This learns steering and acceleration rather than constructing one explicit best-path line. More trials do not guarantee faster wall-clock convergence.

On the default course and custom courses with fixed obstacles, the displayed trials appear together on one road with distinct colors and numbered labels. They are **independent trials**, not interacting traffic: they can overlap and do not collide with each other. At the start they may cluster at the same spawn location, then diverge as they explore. Whole track shows every trial's position; Follow car centers on trial 1, so distant trials may leave that view. Sensor arrows in the shared view belong to trial 1, and the minimap shows all displayed cars.

For random obstacle environments, the dashboard automatically uses separate panels because each trial has different obstacles. Each panel shows its own sensors and obstacles. All displayed trials retain their own crash markers through camera changes. No extra simulation cars are created by the viewer, and training results remain identical in the seeded CPU live/headless check.

## Default course, sensor arrows, and crash history

The default CLI course is now **Long Winding**: ten deep S-bends, a 600 m finish coordinate, approximately 795 m of road, 12 m width, and no obstacles. It is an open winding road, not a closed loop. Run it without selecting a preset:

```bash
.venv/bin/python -m src.train --live --camera follow --output results/long-winding-live
```

Training, recording, and baseline evaluation select this course when no track, configuration, or checkpoint is supplied. Explicit `--config` / `--track` choices and checkpoint course snapshots take precedence. The legacy `--obstacles` workflow still uses the corridor configuration. Custom course timeouts now scale with centerline length; Long Winding receives a 1,816-step horizon with the default vehicle settings.

The dashboard keeps its neutral interface while the road, grass, car, and overlays display in color. Cyan arrows show the actual ray-distance sensors. Three dashed amber arrows point to the 5 m, 12 m, and 20 m road-preview inputs supplied to road policies. These are observations, not steering commands or a planned trajectory. Use **Follow car** to see them clearly on this long course.

Collisions leave faded red car markers at their pre-reset positions and headings. They survive camera switches and episode resets. Markers are visual only and do not become physical obstacles. The live view retains a share of up to 200 recent crash markers across its displayed trial cars; camera changes preserve all of them. Recorded grids retain up to 200 markers per displayed lane. Successes and timeouts do not leave markers. Recording also retains crash history and shows arrows by default; use `record --no-rays` to hide the arrows.

Crash poses are captured on-device before automatic reset and transferred with the displayed frames. Rendering does not change training inputs or actions. Restart the live process after updating the code to load these features. Existing checkpoints keep their original track and timeout unless you explicitly override the track.

## Watch it train live

Add `--live` to training to open a local browser dashboard:

```bash
.venv/bin/python -m src.train --track tracks/presets/gentle_s.yaml --output results/gentle-s-live --live
```

The dashboard shows **eight actual trial cars by default**, including independent exploratory actions and automatic episode resets. Set `--live-cars` from 1 to 16 to choose how many to display; it is capped by `--num-envs`. Buttons switch between **Whole track** and **Follow car**, which follows trial 1. The colored number labels and dashboard legend identify each trial. It displays the update number, collected transitions, and the most recent rollout's reward, successes, and collisions across all cars. The image pauses during the PPO optimization phase and the dashboard labels that phase explicitly.

Watched training is intentionally paced: defaults display every four simulation steps at up to ten frames per second. This makes movement visible and slows training compared with headless mode. Camera rendering transfers only the displayed lane to CPU. Training rates reported in this mode include visualization and waiting, so do not use them as simulator benchmarks. Rendering does not change actions, rewards, or the policy update; a CPU integration test verifies identical model weights with and without live viewing for a seeded short run.

Useful options:

```bash
# Slower, roughly real-time car motion at the default 0.1-second timestep:
.venv/bin/python -m src.train --track tracks/presets/gentle_s.yaml --output results/gentle-s-slow --live --live-every 1 --live-fps 10

# Start with the close-up camera:
.venv/bin/python -m src.train --track tracks/presets/gentle_s.yaml --output results/gentle-s-follow --live --camera follow
```

`--no-browser` prints the URL without opening a tab. `--live-port 8765` selects a port; otherwise a free port is chosen. The server listens only on `127.0.0.1` and serves the dashboard and training snapshots, not project files. No extra package installation is needed beyond the project requirements.

When training finishes, the final frame and statistics stay available until you press **Ctrl+C in Terminal**. Ctrl+C during training also saves the current policy and closes the viewer. As with ordinary training, saved checkpoints contain model weights but not optimizer/rollout state. `--live-exit` exits immediately after training instead of keeping the dashboard available. Closing the browser tab alone does not stop training. Remove `--live` to train at normal speed.

Use a new output directory for a fresh run to preserve earlier results. To watch continued learning from a trained policy, add `--resume results/gentle-s/checkpoint.pt`; this loads the weights with a fresh optimizer. Viewing an already-running headless training process is not supported; `--live` must be enabled when starting it.

## Five preset roads

Ready-made courses are in [tracks/presets](tracks/presets/README.md): **Gentle S**, **Wide Sweep**, **Obstacle Slalom**, **Narrow Bends**, and **Mixed Course**. They vary road width, turns, and fixed obstacles. Start with:

```bash
.venv/bin/python -m src.train --track tracks/presets/gentle_s.yaml --output results/gentle-s
```

See the preset guide for course details and recording commands.

## Full-track camera and custom courses

Use `--camera overview` for a fixed view of the entire playable track. `--camera follow` is the default close-up view. Camera selection does not change training or vehicle physics. These commands use the project's virtual environment:

```bash
.venv/bin/python -m src.record --checkpoint results/bendy/checkpoint.pt --cars 1 --camera overview --frames 300 --output results/bendy-overview.mp4
open results/bendy-overview.mp4
```

### Build your own road

Edit [tracks/custom.yaml](tracks/custom.yaml). `points` describe the road centerline in meters, and `obstacles` contain fixed circles as `[x, y, radius]` in world coordinates. For example:

```yaml
name: My course
width: 10
points:
  - [0, 0]
  - [20, 0]
  - [40, 6]
  - [70, 0]
obstacles:
  - [15, 2, 0.8]
  - [45, 3, 0.7]
```

On macOS, `open -e tracks/custom.yaml` opens the editable file. Use `obstacles: []` for no obstacles. Road points are joined with straight segments; add intermediate points for smoother bends. Points must start at x=0, end at x>=20, and have strictly increasing x coordinates. Loops, overpasses, and roads that double back are not supported yet. Width is vertical edge separation, as in the original bendy-road environment. Validation rejects nonfinite coordinates, roads too narrow/steep for the car, obstacles outside the road, and obstacles in the first eight meters. It does not guarantee every custom course is solvable; leave enough clearance and avoid abrupt turns.

Preview the geometry, then train:

```bash
.venv/bin/python -m src.record --track tracks/custom.yaml --road-controller --camera overview --cars 1 --frames 1 --output results/custom-preview.mp4
.venv/bin/python -m src.train --track tracks/custom.yaml --output results/custom
.venv/bin/python -m src.evaluate --checkpoint results/custom/checkpoint.pt --output results/custom-evaluation.json
.venv/bin/python -m src.record --checkpoint results/custom/checkpoint.pt --camera overview --cars 1 --output results/custom-trained.mp4
open results/custom-trained.mp4
```

The reference controller follows the centerline and **does not plan around obstacles**. Use it to inspect the road, not as an obstacle-avoidance demonstration. PPO receives obstacle distances and collision penalties, but needs training on the custom course before expecting successful avoidance.

`--track` works with training, evaluation, and recording. Without `--config`, it starts from bendy-road settings, retaining the same 25 inputs. You can initialize from your bendy policy with `--resume results/bendy/checkpoint.pt` during training. You can also test a road policy on another course with `--checkpoint ... --track tracks/another.yaml` during recording/evaluation. Corridor policies with 19 inputs are incompatible. Custom track obstacles replace random spawning; edit the YAML instead of passing `--obstacles`.

Checkpoint configuration embeds the course and obstacle coordinates, so playback does not require the original YAML. Editing YAML does not change an existing checkpoint unless you explicitly pass `--track` to override its course. The camera follows the same geometry used by sensing and collision checks. Custom obstacles remain fixed across resets; evaluation on one course varies spawn offsets, not obstacle layouts.

## New: bendy road and top-down car

`configs/bendy.yaml` adds an obstacle-free S-shaped asphalt road with painted edges, dashed center markings, and a finish line. The car has wheels, steered front tires, windows, and lights. A following camera, speed display, progress indicator, and minimap show its movement through the turns.

Create and open the demo on macOS:

```bash
python -m src.record --config configs/bendy.yaml --road-controller --cars 1 --frames 300 --output results/bendy-road.mp4
open results/bendy-road.mp4
```

The delivered demo uses a **pure-pursuit reference controller with access to the road map**, not a learned policy. Add `--rays` for sensor beams or `--cars 4` for parallel cars.

Train a fresh PPO policy on the bends:

```bash
python -m src.train --config configs/bendy.yaml --output results/bendy
python -m src.evaluate --checkpoint results/bendy/checkpoint.pt --output results/bendy-evaluation.json
python -m src.record --checkpoint results/bendy/checkpoint.pt --cars 1 --output results/bendy-trained.mp4
```

Start fresh: corridor checkpoints have 19 inputs, while bendy-road policies have **25**. The six extra inputs are normalized lateral offset, sine/cosine of road-relative heading, and three upcoming road bearings. These assume access to a road map. The original corridor configuration and checkpoints keep their existing interface.

Road edges drive ray sensing and swept collision checks as well as rendering. Going off-road ends an episode; reaching the finish succeeds. Reward uses centerline arc progress at the car's x-coordinate, plus small offset and heading penalties. Cars reset facing the road tangent. A conservative circular footprint encloses the car drawing; oriented rectangular body collision is not modeled.

Tune `road.amplitude`, `road.wavelength`, and `road.spacing`, along with world `length` and `width`. Width is vertical separation between edges, so perpendicular width narrows slightly on slopes. This first version has one fixed course: evaluation varies spawn offsets, not course shape. Batched edge sensing uses `(N,R,S)` tensors and is more expensive than corridor sensing.

Validation: **23 passed, 1 skipped**. All eight tested reference-driver episodes finished without collision. A five-update PPO smoke run completed, but its policy has not learned the course. See [feature validation](results/BENDY-VALIDATION.md) for exact results.

## Quick start

Run from the `driving-rl` directory with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.record --random --cars 4 --frames 200 --output results/random.mp4
```

On macOS, the package sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing PyTorch. `--device auto` chooses CUDA, then MPS, then CPU. Floating tensors use float32. Optional fallback may affect measured performance. Tests skip MPS when unavailable.

## Train, evaluate, record

Start with no obstacles, then initialize obstacle training from that policy:

```bash
python -m src.train --obstacles 0 --output results/empty
python -m src.train --resume results/empty/checkpoint.pt --obstacles 12 --output results/obstacles
python -m src.evaluate --checkpoint results/obstacles/checkpoint.pt --episodes 500 --seed 10000 --output results/trained.json
python -m src.evaluate --episodes 500 --seed 10000 --output results/untrained.json
python -m src.record --checkpoint results/obstacles/checkpoint.pt --output results/trained.mp4
python -m src.record --output results/untrained.mp4
```

`--resume` initializes model weights; optimizer, environment state, and update count restart. Use a matching network configuration. Evaluation and recording load the checkpoint's environment configuration; evaluation permits an explicit `--obstacles` override. Compare policies with identical obstacle counts, episode counts, and evaluation seeds. The untrained baseline uses a seeded randomly initialized network with deterministic mean actions; `record --random` is a separate random-action demonstration.

For a short CPU smoke test:

```bash
python -m src.train --device cpu --num-envs 64 --updates 10 --obstacles 0 --output results/smoke
```

Training saves `checkpoint.pt`, `metrics.csv`, and `learning.png`. Rates logged during training include policy inference and optimization, unlike simulator-only benchmarks. Train several seeds before making robust performance claims. A single learning curve is not evidence of generalization.

## Simulator contract

`BatchedDrivingEnv.reset(seed=None)` returns a float32 observation tensor `(N, R + 3)`. `step(actions)` accepts `(N, 2)` normalized steering and acceleration and returns `(obs, reward, terminated, truncated, info)`. With 16 rays, observations have **19** components: normalized ray lengths, speed, and the previous two actions.

State contains `pos (N,2)`, `heading (N,)`, `speed (N,)`, `obstacles (N,K,3)`, and integer `steps (N,)`. The bicycle model uses semi-implicit integration with a fixed timestep. Speed is restricted to forward motion. Collision checks sweep a circular car along the discrete movement segment and test corridor boundaries. The circular footprint and straight movement segment are approximations to a real vehicle.

Rays use a batched `(N,R,K)` ray-circle calculation, plus intersections with the two side walls. Cars see raw obstacle boundaries; collision uses the additional car radius. Rays return zero from inside a circle, detect tangencies, and return maximum range for misses. No car or ray loops occur in the simulator; rendering deliberately uses loops over at most 16 cars.

Reward is forward displacement plus an alive bonus, minus squared steering-command change and collision cost, plus a finish bonus. Collision or crossing the finish terminates an episode. The time limit truncates it. A stalled car can earn the small alive bonus; success, timeout, return, and collision metrics must be read together.

Finished lanes reset automatically. `obs` describes the state after resets; `info['final_observation']` describes the state before resets. PPO bootstraps timeouts from that final observation, zeroes bootstrap at true termination, and stops GAE traces across either boundary. Resets sample a full candidate batch every step and apply masks: this avoids synchronization for dynamic mask sizes but adds RNG and memory work even when no car resets. Obstacles may overlap; the generator does not guarantee every scene is traversable.

Each environment owns a seeded device generator. Reproducibility is tested for the same device, configuration, batch size, and software stack. Identical trajectories across CPU, MPS, and CUDA are not promised.

## PPO

A two-layer tanh MLP shares features between actor and critic. The Gaussian actor uses a tanh transform for bounded actions and includes its log-Jacobian in likelihoods. Rollouts retain latent actions to avoid numerical inverse-tanh errors. Storage uses `(T,N)` with trailing observation/action axes. Updates use GAE, advantage normalization, the clipped surrogate objective, value regression, squashed-policy entropy estimates, and gradient clipping.

## Evaluation protocol

The default evaluation creates 500 lanes and counts **only the first episode of each lane**, using seed 10000. It runs for the configured time limit, ignoring later auto-reset episodes. This prevents faster-finishing episodes from dominating the sample. JSON reports collision, success, timeout, return, and length. Evaluation uses deterministic mean actions. Do not tune repeatedly on the reporting seed; use separate development and final evaluation seeds.

## Benchmarks

```bash
python -m src.benchmark --sizes 1 64 1024 8192 --steps 20 --repeats 3
```

The Python-loop baseline calls the same PyTorch environment once per car on CPU. Vectorized modes call it once per batch on CPU and available accelerators. Each uses the same configuration and action pattern, including sensors, rewards, and reset candidate generation; individual scene samples differ. Initialization and rendering are excluded. Three warm-up steps precede measurements. GPU synchronization occurs at timing boundaries, using [PyTorch's MPS synchronization API](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html). Reported throughput counts car transitions (`N × steps / elapsed`), not batched calls. CSV records median/min/max, hardware platform, thread count, and PyTorch version.

The large Python-loop case is intentionally slow. Use `--skip-loop` for device-only exploration. Run benchmarks without concurrent training, in short bursts with cooling breaks, and repeat under similar power conditions. MPS may lose at small batches. The `(N,R,K)` allocation grows with all three dimensions; reduce batch size or obstacle count if memory is constrained. A tiny smoke benchmark is not a stable performance result.

## Layout

```text
configs/default.yaml     simulator, rewards, PPO parameters
src/sim/                 bicycle model, world, rays, environment, renderer
src/rl/                  actor-critic, rollout buffer, GAE, PPO update
src/train.py             training CLI and checkpoints
src/evaluate.py          fixed-seed first-episode evaluation
src/record.py            MP4 videos with rays
src/benchmark.py         loop / vectorized CPU / available GPU timing
src/plot.py              learning curves
tests/                   geometry, dynamics, resets, determinism, PPO checks
results/                 generated runs, metrics, figures, and videos
```


