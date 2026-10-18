# Multiple simultaneous trial cars

- CPU suite: **49 passed, 1 skipped** (MPS unavailable).
- Renderer tests verify distinct colors and exact actual-lane poses, multiple-lane crash markers, focused sensor arrows, shared fixed obstacles drawn once, and preserved environment state/RNG. Random obstacle worlds reject overlays and retain separate panels.
- Live HTTP tests verify multi-car frame delivery, displayed count, and preservation of count/history during camera changes.
- Seeded training with four displayed trials produces exactly the same policy weights as headless training with the same four environments. The shared PPO training algorithm and rewards were not changed.
- Dashboard JavaScript passed syntax validation.
- `multi-car-preview.png` is a visual fixture with eight deliberately positioned simulator states, inspected to verify colors, labels, and sensor visibility. It is not a learned-policy result.

Live viewing now defaults to eight trials. `--live-cars` accepts 1–16, capped by the actual training batch size. Common fixed/no-obstacle courses use one shared road view; randomized obstacle worlds use a grid of independent views. Trials do not collide with each other. Their experiences are pooled for the existing shared-policy update, not combined into an explicit path.

Whole track displays all selected trials; follow mode centers on trial 1. Shared-view arrows belong to trial 1. Trial IDs/colors remain stable through independent resets. Live crash history is bounded to about 200 markers across the selected trials and survives camera switches.
