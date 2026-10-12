from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .policy_math import ACTION_DIM, OBSERVATION_DIM


def validate_tensor_contract(
    label: str,
    shape: Any,
    tensor_type: str,
    feature_dim: int,
) -> None:
    if tensor_type != "tensor(float)":
        raise ValueError(f"Policy {label} must use float32, got {tensor_type!r}")
    try:
        rank = len(shape)
    except TypeError as exc:
        raise ValueError(f"Policy {label} must be rank 2, got {shape!r}") from exc
    if rank != 2:
        raise ValueError(f"Policy {label} must be rank 2, got {shape}")
    batch_dim, actual_feature_dim = shape
    if batch_dim != 1 and batch_dim is not None and not isinstance(batch_dim, str):
        raise ValueError(
            f"Policy {label} batch dimension must accept one sample, got {batch_dim!r}"
        )
    if actual_feature_dim != feature_dim:
        raise ValueError(
            f"Policy {label} feature dimension must be {feature_dim}, got {shape}"
        )


class OnnxPolicy:
    def __init__(self, path: str | Path, *, input_dim: int | None = None, output_dim: int = ACTION_DIM):
        candidate = Path(path)
        try:
            self.path = candidate.expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Cannot resolve ONNX policy path {candidate!s}") from exc
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Actor must have exactly one input and one output")
        self._input = inputs[0]
        self._output = outputs[0]
        actual_input_dim = input_dim if input_dim is not None else OBSERVATION_DIM
        validate_tensor_contract("input", self._input.shape, self._input.type, actual_input_dim)
        validate_tensor_contract(
            "output", self._output.shape, self._output.type, output_dim
        )
        self.input_dim = actual_input_dim
        self.output_dim = output_dim
        self.input_name = self._input.name
        self.output_name = self._output.name

    def warmup(self) -> None:
        self.infer(np.zeros((1, self.input_dim), dtype=np.float32))

    def infer(self, observation: Any) -> np.ndarray:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                obs = np.asarray(observation, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("observation must be convertible to float32") from exc
        if obs.shape == (self.input_dim,):
            obs = obs.reshape(1, self.input_dim)
        if obs.shape != (1, self.input_dim):
            raise ValueError(f"observation must have shape (1, {self.input_dim})")
        if not np.all(np.isfinite(obs)):
            raise ValueError(
                "observation must be representable as finite float32"
            )
        feed = np.ascontiguousarray(obs)

        result = self._session.run(
            [self.output_name], {self.input_name: feed}
        )
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError("Policy output must contain exactly one tensor")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                action = np.asarray(result[0], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Policy output is not convertible to float32") from exc
        if action.shape != (1, self.output_dim):
            raise RuntimeError(f"Policy output must have shape (1, {self.output_dim})")
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                "Policy output must be representable as finite float32"
            )
        return np.ascontiguousarray(action.reshape(self.output_dim))
