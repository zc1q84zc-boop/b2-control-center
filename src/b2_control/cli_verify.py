from __future__ import annotations

import argparse
from collections.abc import Sequence
import json

import numpy as np

from .bundle import open_verified_bundle
from .onnx_policy import OnnxPolicy
from .policy_math import Q_DEFAULT, build_observation
from .stack10_adapter import Stack10ObservationAdapter


EXPECTED_HOLD_ACTION = np.array(
    (
        0.3856276273727417,
        -0.4630189836025238,
        1.292534351348877,
        -0.4785652160644531,
        -0.5396023988723755,
        1.1809988021850586,
        0.3231983184814453,
        -0.3763878345489502,
        0.8902627229690552,
        -0.055662669241428375,
        -0.16711482405662537,
        0.8141254186630249,
    ),
    dtype=np.float32,
)
EXPECTED_HOLD_ACTION.setflags(write=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a B2 policy handoff bundle"
    )
    parser.add_argument("bundle", help="Path to the extracted deployment bundle")
    args = parser.parse_args(argv)

    bundle = open_verified_bundle(args.bundle)
    if getattr(bundle, "adapter", "b2_45x12_v1") == "b2_stack10_v1":
        policy = OnnxPolicy(
            bundle.policy_path,
            input_dim=getattr(bundle, "input_dim", 423),
            output_dim=getattr(bundle, "output_dim", 12),
        )
    else:
        policy = OnnxPolicy(bundle.policy_path)
    if getattr(bundle, "adapter", "b2_45x12_v1") == "b2_stack10_v1":
        adapter = Stack10ObservationAdapter()
        observation = adapter.build_input(
            np.zeros(3), np.array((0.0, 0.0, -1.0)), np.zeros(3), Q_DEFAULT, np.zeros(12)
        )
        action = policy.infer(observation)
        contract_path = bundle.root / "policy_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected = contract.get("golden_reset_inference", {}).get("tick_1_action")
        if expected is not None:
            np.testing.assert_allclose(action, np.asarray(expected, dtype=np.float32), rtol=1e-5, atol=1e-6)
    else:
        observation = build_observation(
            omega_body=np.zeros(3),
            rotation_body_to_world=np.eye(3),
            command=np.zeros(3),
            q=Q_DEFAULT,
            dq=np.zeros(12),
            previous_action=np.zeros(12),
        )
        action = policy.infer(observation)
        np.testing.assert_allclose(action, EXPECTED_HOLD_ACTION, rtol=1e-5, atol=1e-6)
    print(
        f"Bundle verified: hashes, float32 {getattr(bundle, 'input_dim', 45)} -> "
        f"{getattr(bundle, 'output_dim', 12)} contract, "
        "and golden inference match."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
