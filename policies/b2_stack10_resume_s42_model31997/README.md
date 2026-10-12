# B2 Stack10 `model_31997` deployment handoff

This folder is a **separate handoff bundle** for the Seed-42 Stack10
continuation policy. It does not modify `B2_sim2real` or the policy currently
used on the robot.

## Critical integration difference

The current baseline actor accepts one 45-value frame. This actor accepts a
423-value, term-major history:

```text
ang_vel[10] | gravity[10] | current_command |
joint_pos[10] | joint_vel[10] | previous_action[10]
```

Replacing only `policy.onnx` will not work. The deployment owner must also add
the history adapter and its reset/update semantics. Do not reshape, zero-pad,
or concatenate ten complete 45-value frames: all three produce the wrong
layout while still looking numerically plausible.

## Contents

- `checkpoints/model_31997.pt`: original RSL-RL training checkpoint.
- `models/policy.onnx`: deployable actor reconstructed from the checkpoint's
  `actor.*` tensors.
- `stack10_adapter.py`: standalone NumPy reference for the exact observation,
  history, reset, and action-target transforms.
- `policy_contract.json`: machine-readable dimensions, offsets, constants,
  timing, joint order, and golden reset actions.
- `verify_bundle.py`: hashes, ONNX interface, adapter, and two-tick golden
  inference verification.
- `DEPLOYMENT_CHECKLIST.md`: integration, simulation, hardware, logging, and
  rollback gates.
- `evidence/`: the preserved full and oblique MuJoCo summary tables.
- `PROVENANCE.md` and `SHA256SUMS`: source and artifact identity.

## Verify before integration

From this folder, using the existing `b2_sim2sim` environment:

```bash
sha256sum -c SHA256SUMS
conda run --no-capture-output -n b2_sim2sim python verify_bundle.py
```

The verifier must report:

```text
PASS: artifact hashes
PASS: Stack10 adapter self-test
PASS: ONNX float32 [1, 423] -> [1, 12]
PASS: two-tick golden inference
```

## Required policy-cycle order

At each **50 Hz policy tick**:

1. Read one coherent sensor sample.
2. Build the current 45-value frame, using the raw actor output from the
   previous policy tick as `previous_action`.
3. Shift/append that frame once and assemble the term-major 423-vector.
4. Run the ONNX actor.
5. Validate that all 12 outputs are finite.
6. Save the raw output as `previous_action` for the next policy tick.
7. Convert it to joint targets with
   `q_target = q_default + action_scale * raw_action`.
8. Hold those targets until the next policy tick while retaining the existing
   robot-side joint, velocity, torque, timeout, fall, and E-stop protections.

If the low-level controller runs at 200 Hz, append history only on every
fourth tick. Appending duplicate samples at 200 Hz changes the learned history
window from 0.18 seconds to 0.045 seconds and violates the policy contract.

On controller start, policy activation, reset, E-stop recovery, sensor
recovery, or policy reload, clear `previous_action` to zero. On the first
valid post-reset policy tick, repeat that first frame across all ten history
positions. Never carry history across a reset or policy switch.

See `stack10_adapter.py` for executable reference code. A typical integration
sequence is:

```python
adapter.reset()

actor_input = adapter.build_input(
    body_angular_velocity=omega_body,
    projected_gravity=gravity_body,
    command=(vx, vy, wz),
    joint_position=q,
    joint_velocity=dq,
)
raw_action = session.run(["actions"], {"obs": actor_input})[0].reshape(12)
adapter.commit_action(raw_action)
q_target = action_to_joint_targets(raw_action)
```

## Safety boundary

The included results are MuJoCo diagnostics, not a physical-robot safety
guarantee. The deployment owner should not enable motor output until every
offline and simulation gate in `DEPLOYMENT_CHECKLIST.md` passes. Initial
hardware trials require the team's normal E-stop, controlled test area,
safety support, conservative commands, live telemetry, and a tested rollback
to the existing baseline.

