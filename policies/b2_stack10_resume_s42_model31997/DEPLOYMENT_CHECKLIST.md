# Deployment checklist

No model-file-only swap is permitted: the 423-input actor and Stack10 adapter
must be integrated and validated together.

## 1. Artifact and interface gate

- [ ] Keep the currently deployed baseline and its configuration as the
  rollback image; do not overwrite it in place.
- [ ] Run `sha256sum -c SHA256SUMS` with every line passing.
- [ ] Run `python verify_bundle.py` with all four `PASS` lines.
- [ ] Confirm the runtime sees exactly one float32 input named `obs` with
  shape `[1, 423]` and one float32 output named `actions` with shape `[1, 12]`.
- [ ] Record the deployed ONNX SHA-256 in the robot's run manifest.

## 2. Controller integration gate

- [ ] Preserve the joint order exactly: FR, FL, RR, RL; hip, thigh, calf
  within each leg.
- [ ] Preserve the current-frame sensor conventions, units, scaling, command
  order, default joint offsets, and raw previous-action values in
  `policy_contract.json`.
- [ ] Use five independent 10-sample histories, oldest to newest, and insert
  only the current command once at offsets 60:63.
- [ ] Append exactly once per 20 ms policy tick, not once per faster motor or
  DDS loop tick.
- [ ] Commit the new raw action only after successful inference; never put
  joint targets, clipped torques, or measured joint positions in the
  `previous_action` history.
- [ ] Reset both history and previous action on startup, mode entry, policy
  switch, operator reset, E-stop recovery, invalid/stale sensors, timing
  overrun recovery, and localization/IMU reinitialization.
- [ ] Repeat the first valid post-reset frame ten times. Do not initialize
  history with zeros unless that complete first frame is actually zero.
- [ ] Preserve the 50 Hz policy period and hold each target between inferences.
- [ ] Preserve all existing real-robot watchdogs, joint limits, velocity
  limits, torque/current limits, fall detection, command limits, and E-stop
  behavior. The policy output is a position offset, not a torque command.
- [ ] Treat any non-finite input/output, stale sensor sample, wrong tensor
  shape, inference exception, or missed policy deadline as a transition to
  the existing safe state; reset history before re-entry.

## 3. Offline and simulation gate

- [ ] Add unit tests that compare the production adapter against
  `stack10_adapter.py` for reset and at least 20 sequential frames.
- [ ] Reproduce both golden reset actions in `policy_contract.json` using the
  production runtime and its target hardware architecture.
- [ ] Replay representative recorded robot telemetry through the new adapter;
  inspect action and target discontinuities, extrema, NaNs, inference latency,
  and deadline misses without enabling motors.
- [ ] Run the same MuJoCo full and oblique suites with the production model and
  adapter. Compare against `evidence/full_summary.csv` and
  `evidence/oblique_summary.csv`.
- [ ] Exercise startup, reset, command changes, policy reload, sensor dropout,
  inference error, deadline overrun, and E-stop recovery in simulation.
- [ ] Confirm the first target after every reset is bounded and does not depend
  on history from the previous episode.
- [ ] Confirm rollback selects the original model **and** original 45-input
  adapter atomically.

## 4. Controlled hardware gate

- [ ] Obtain the normal internal review/authorization from the robot owner and
  safety operator.
- [ ] Use a controlled test area with the team's normal physical safety
  support, tested E-stop, exclusion zone, and at least two operators.
- [ ] First run the complete control loop in shadow mode: read sensors, build
  history, infer, and log, but do not send policy-derived motor targets.
- [ ] Compare shadow targets with the active baseline for finiteness, magnitude,
  rate of change, joint-order correctness, and timing.
- [ ] Enter the policy only from the established stable stand state. Reset the
  adapter immediately before the first active policy tick.
- [ ] Begin with zero command, then use the team's smallest approved forward
  and yaw commands on flat ground. Do not begin on stairs or rough terrain.
- [ ] Monitor body attitude, joint tracking, motor current/torque, action and
  target extrema, policy latency, history-reset count, stale-sensor count, and
  watchdog state live.
- [ ] Define abort thresholds before motion and stop at the first unexpected
  oscillation, foot placement, target jump, timing overrun, sensor fault, or
  safety-limit activation.
- [ ] Expand speed, yaw, duration, terrain, and payload one variable at a time,
  only after reviewing the preceding logs.

## 5. Required run logging

For every trial, retain at minimum:

- model hash, adapter/firmware revision, robot ID, operator, and timestamp;
- commanded `[vx, vy, wz]` and policy/reset state;
- all 423 actor inputs or the six source terms needed to reconstruct them;
- raw 12-action output, 12 joint targets, measured positions/velocities, and
  applied torque/current;
- inference duration, policy-tick jitter, missed deadlines, stale-sensor
  events, history resets, safety clamps, E-stop, and termination reason.

