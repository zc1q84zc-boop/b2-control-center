#!/usr/bin/env python3
"""Verify artifact identity, Stack10 layout, ONNX interface, and golden outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from stack10_adapter import (
    ACTION_DIM,
    Q_DEFAULT,
    Stack10ObservationAdapter,
    self_test,
)


ROOT = Path(__file__).resolve().parent
EXPECTED_HASHES = {
    "checkpoints/model_31997.pt": (
        "9c5c66662a90b399bf282d2106fa9a07f9ee0a8352f2c6a8933454279f55ba1d"
    ),
    "models/policy.onnx": (
        "bbb474e2fddc800f609c2360ee083e1cda250e8076c2805dde0c2e1203c70265"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts() -> None:
    for relative_path, expected in EXPECTED_HASHES.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_path}: expected {expected}, got {actual}"
            )


def verify_onnx() -> None:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError(
            "ONNX verification requires onnxruntime; use the b2_sim2sim environment."
        ) from error

    session = ort.InferenceSession(
        str(ROOT / "models" / "policy.onnx"),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Expected exactly one ONNX input and output.")
    actor_input, actor_output = inputs[0], outputs[0]
    if (
        actor_input.name != "obs"
        or actor_input.type != "tensor(float)"
        or actor_input.shape != [1, 423]
    ):
        raise RuntimeError(
            f"Wrong ONNX input contract: {actor_input.name} "
            f"{actor_input.type} {actor_input.shape}"
        )
    if (
        actor_output.name != "actions"
        or actor_output.type != "tensor(float)"
        or actor_output.shape != [1, 12]
    ):
        raise RuntimeError(
            f"Wrong ONNX output contract: {actor_output.name} "
            f"{actor_output.type} {actor_output.shape}"
        )

    contract = json.loads((ROOT / "policy_contract.json").read_text())
    golden = contract["golden_reset_inference"]
    adapter = Stack10ObservationAdapter()
    actual_actions = []
    for _ in range(2):
        observation = adapter.build_input(
            np.zeros(3, dtype=np.float32),
            np.array((0.0, 0.0, -1.0), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            Q_DEFAULT,
            np.zeros(ACTION_DIM, dtype=np.float32),
        )
        action = np.asarray(
            session.run(["actions"], {"obs": observation})[0],
            dtype=np.float32,
        ).reshape(ACTION_DIM)
        if not np.all(np.isfinite(action)):
            raise RuntimeError("ONNX actor returned non-finite actions.")
        actual_actions.append(action)
        adapter.commit_action(action)

    tolerance = golden["tolerance"]
    for tick, (actual, expected) in enumerate(
        zip(
            actual_actions,
            (golden["tick_1_action"], golden["tick_2_action"]),
            strict=True,
        ),
        start=1,
    ):
        np.testing.assert_allclose(
            actual,
            np.asarray(expected, dtype=np.float32),
            rtol=float(tolerance["rtol"]),
            atol=float(tolerance["atol"]),
            err_msg=f"golden reset inference tick {tick}",
        )


def main() -> None:
    verify_artifacts()
    print("PASS: artifact hashes")
    self_test()
    print("PASS: Stack10 adapter self-test")
    verify_onnx()
    print("PASS: ONNX float32 [1, 423] -> [1, 12]")
    print("PASS: two-tick golden inference")


if __name__ == "__main__":
    main()

