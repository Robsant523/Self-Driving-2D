# Long Winding and visualization validation

- CPU suite: **44 passed, 1 skipped** (MPS unavailable).
- New tests verify pre-reset crash positions/headings, ignored timeout markers, capture between rendered frames, history retention across cameras, bounded marker storage, road-preview arrow endpoints, and default CLI course selection with sufficient timeout.
- The seeded live/headless integration test still produces identical policy weights.
- Long Winding: 101 centerline points, x finish 600 m, approximately 794.63 m arc length, ten deep alternating bends, 12 m vertical width, and zero obstacles.
- An eight-car CPU reference-controller run finished every first episode by step 1320 with no collisions or timeouts. The default course loader gives 1816 steps. This is a reference-controller geometry check, not a PPO result.
- `long-winding-sensors-preview.png` was generated with two deliberately constructed collision transitions to inspect marker location, opacity, colors, and arrows. It is a visualization fixture, not a training outcome.
- `long-winding-overview.png` shows the complete new course. Both figures were visually inspected.

Cyan arrows visualize sensor rays. Amber arrows visualize the same three lookahead centerline targets used in road observations; they do not represent steering commands. Crash cars are visual markers only. The live view keeps lane 1's most recent 200 crashes across camera changes. No crash marker is recorded for timeout/success, and no rendered marker changes collision geometry.

The preset has S bends along increasing x coordinates, not closed or crossing loops. Existing checkpoint snapshots preserve their original course unless explicitly overridden.
