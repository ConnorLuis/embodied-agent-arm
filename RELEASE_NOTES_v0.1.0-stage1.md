# v0.1.0-stage1 — Stage 1 closed/frozen

## Summary

Stage 1 is closed as a documented engineering success and task-level negative result.

- Completed the SO-ARM101 data, ACT training, Front/Wrist camera, observation, safety-guard and controlled-deployment pipeline.
- Completed one controlled policy episode with 180/180 replans and 900/900 acknowledged commands.
- Recorded 0 rate clips, 0 new boundary crossings, 0 tracking trips and guard invariant error 0.
- Returned to Home and closed torque, serial and camera transports.
- The cube was not grasped. Gripper commands collapsed to a narrow 27.648575–27.837341 range.
- No V3/V4 checkpoint passed the frozen bidirectional gripper-transition gate, so further training and hardware deployment were stopped.

## Included in this release

- Runtime safety contract and staged deployment scripts.
- Dataset, training and offline audit code.
- Rewritten public README and corrected closeout terminology.
- Curated public evidence JSON/CSV and Front/Wrist outcome frames.
- CPU-only static CI and public-closeout verifier.

## Deliberately excluded

- Raw episodes and validation media.
- Model checkpoints and full `outputs/` artifacts.
- Calibration files and serial/device identifiers.
- Vendor source archives and local patches.

## Claims boundary

This release does not claim successful autonomous pick-and-place, a deployable V4 policy, or a live-validated Home-to-Park fold.
