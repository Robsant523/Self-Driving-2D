# Live training validation

- CPU suite: **40 passed, 1 skipped** (MPS unavailable).
- Live HTTP tests verify PNG snapshots, camera changes, unchanged environment state/RNG, invalid-request handling, and server cleanup.
- A seeded training integration test produced exactly matching model weights with and without live rendering and pacing. The live process also exited successfully with `--live-exit` and wrote its checkpoint and chart.
- A two-update watched run completed and served final metrics, transitions, and a switch from overview to follow camera after training finished.
- An additional subprocess check sent Ctrl+C during rollout collection: it saved a loadable checkpoint and exited cleanly. The completed test viewer was also stopped successfully with Ctrl+C.
- The dashboard JavaScript passed Node's syntax check. Browser automation was unavailable in this execution environment, so full browser interaction and screenshot verification remain untested; HTTP endpoints and rendered frame data were verified directly.

The viewer shows the first actual training lane. Outcomes/reward metrics describe all cars in the last completed rollout. The frame freezes during policy optimization. Live rendering intentionally paces rollouts, so its throughput is not a simulator benchmark. Checkpoints save current policy weights, not full optimizer or rollout state.
