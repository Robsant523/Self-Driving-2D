# Delivered validation

These measurements were produced by the delivered code on CPU. They are a single training seed and single evaluation seed, not a multi-seed study. MPS was unavailable to PyTorch in this execution environment; CUDA was unavailable. Accelerator throughput and learning remain unverified.

## Correctness

`python -m pytest -q`: **16 passed, 1 skipped** (MPS unavailable). Checks cover ray hits/misses/tangencies/inside-circle origins/nearest hit/rotation/walls, bicycle motion, swept collision, state shapes and dtypes, seeded replay, selective resets, final observations, success/collision flags, timeout bootstrapping and GAE boundaries, policy log-probability consistency, finite PPO updates, and exact first-episode evaluation.

The random-action and trained MP4s were decoded and visually inspected. The learning curve was rendered and inspected. Trained and untrained demonstration files use four cars at 10 FPS for 12 seconds. Rendering intentionally transfers data to CPU.

## Learning and evaluation

Training seed: **42**. Empty corridor: 60 updates × 128 steps × 256 cars = **1,966,080 transitions**. Obstacle stage: 100 updates with the same batch dimensions = **3,276,800 additional transitions**, initialized from the empty-corridor weights with a fresh optimizer. Default PPO settings otherwise apply. Total training: **5,242,880 transitions**.

Each row below evaluates **500 first episodes**, seed **10000**, using deterministic mean actions and a 400-step time limit. Empty means zero obstacles; obstacle runs use 12. The randomly initialized baseline uses training seed 42 and separate evaluation scenes. Baseline and trained scene generation use the same configuration and seed.

| Policy / world | Success | Collision | Timeout | Mean return |
|---|---:|---:|---:|---:|
| untrained-empty | 0.0% | 0.0% | 100.0% | 4.99 |
| trained-empty | 100.0% | 0.0% | 0.0% | 119.38 |
| untrained-obstacles | 0.0% | 0.0% | 100.0% | 5.40 |
| trained-obstacles | 44.2% | 55.8% | 0.0% | 65.21 |

The untrained policy mostly stays near its initial position; its zero collision rate does not represent competent driving. The trained obstacle policy moves and finishes some courses, but collision performance needs improvement. Do not claim a collision-rate reduction against this stationary baseline. The 100% empty-corridor success result is specific to this evaluation set. Overlapping obstacles and limited observations can make some scenarios difficult or impossible.

Reproduction (from the project directory):

```bash
python -m src.train --device cpu --num-envs 256 --updates 60 --obstacles 0 --output results/empty
python -m src.train --device cpu --num-envs 256 --updates 100 --obstacles 12 --resume results/empty/checkpoint.pt --output results/obstacles
python -m src.evaluate --device cpu --obstacles 0 --output results/untrained-empty.json
python -m src.evaluate --device cpu --checkpoint results/empty/checkpoint.pt --output results/trained-empty.json
python -m src.evaluate --device cpu --output results/untrained-obstacles.json
python -m src.evaluate --device cpu --checkpoint results/obstacles/checkpoint.pt --output results/trained-obstacles.json
```

Use `requirements-tested.txt` to reproduce the installed dependency versions. The main `requirements.txt` provides lower bounds. Byte-for-byte reproducibility across hardware or library versions is not assumed.

## Throughput

Short CPU benchmark, three warm-up calls and three repeats of three timed calls per case. No training or rendering ran concurrently. These short timings establish functionality and batch scaling; collect longer repeated measurements under controlled thermal conditions before adding a speedup claim to a resume. The Python-loop baseline uses one instance of the same tensor environment per car.

| Cars | Mode | Median car transitions/s |
|---:|---|---:|
| 1 | python_loop | 5,216 |
| 1 | vectorized | 5,303 |
| 64 | python_loop | 5,332 |
| 64 | vectorized | 127,645 |
| 1024 | python_loop | 5,253 |
| 1024 | vectorized | 324,421 |
| 8192 | python_loop | 3,168 |
| 8192 | vectorized | 305,856 |

Command: `python -m src.benchmark --sizes 1 64 1024 8192 --steps 3 --repeats 3`. Full metadata and min/max rates are in `benchmark.csv`. GPU cases were skipped automatically.
