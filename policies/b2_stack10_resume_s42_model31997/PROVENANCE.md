# Artifact provenance

## Training checkpoint

- policy: Seed-42 Stack10 continuation, `model_31997.pt`
- source checkpoint:
  `history/logs/rsl_rl/unitree_b2_piper_robust_heading_rough_history_stack10/2026-08-15_20-12-26_compare_s42_stack10_resume_gpu2_20260815_201220_e4816697_heading_rough/model_31997.pt`
- checkpoint iteration: 31997
- checkpoint SHA-256:
  `9c5c66662a90b399bf282d2106fa9a07f9ee0a8352f2c6a8933454279f55ba1d`
- actor: feed-forward MLP, `423 -> 512 -> 256 -> 128 -> 12`, ELU
- initialization: function-preserving expansion of the canonical no-history
  `model_19998.pt`, followed by 12,000 robust heading-rough PPO updates

The copied checkpoint is byte-identical to the source.

## Deployment actor

- ONNX SHA-256:
  `bbb474e2fddc800f609c2360ee083e1cda250e8076c2805dde0c2e1203c70265`
- ONNX interface: float32 `[1, 423] -> [1, 12]`
- input name: `obs`
- output name: `actions`
- export: reconstructed from the checkpoint's `actor.*` tensors; critic,
  optimizer, exploration state, and other training-only state are excluded
- weight parity: the existing import/export verification reconstructed the
  inference actor without changing its weights
- adapter parity: the included reference adapter bitwise matched the MuJoCo
  Stack10 implementation across 20 sequential randomized frames

## Preserved MuJoCo evidence

The copied August 16 diagnostic summaries report:

- full suite: 45/48 provisional passes, 0 falls, 0 execution errors;
- oblique suite: 48/56 provisional passes, 0 falls, 0 execution errors.

These are simulator diagnostics, not evidence of physical-robot safety.

## Baseline rollback reference

At handoff preparation time, the untouched baseline at
`B2_sim2real/models/policy.onnx` had interface float32 `[1, 45] -> [1, 12]`
and SHA-256
`688bdbc4897c6dd9cc1cce54249131a35e9206614cfad2f383d2e907978b67eb`.
It is not duplicated in this folder. The robot owner should preserve their
actual deployed image and configuration as the authoritative rollback target.
