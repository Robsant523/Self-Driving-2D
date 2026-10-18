# Batched Driving RL

This is a simplified 2D control experiment, not a photorealistic simulator or an autonomous-driving safety system. Results describe this corridor task only.

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


