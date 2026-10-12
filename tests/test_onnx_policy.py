import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from b2_control.onnx_policy import OnnxPolicy, validate_tensor_contract


class FakeSession:
    def __init__(self, *, inputs=None, outputs=None, result=None):
        self._inputs = inputs or [_meta("observation", [1, 45])]
        self._outputs = outputs or [_meta("action", [1, 12])]
        self.result = (
            [np.zeros((1, 12), dtype=np.float32)] if result is None else result
        )
        self.run_calls = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        self.run_calls.append((output_names, feeds))
        return self.result


def _meta(name, shape, tensor_type="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=tensor_type)


def _make_policy(tmp_path, monkeypatch, session=None):
    model = tmp_path / "actor.onnx"
    model.write_bytes(b"fake-onnx")
    fake_session = session or FakeSession()
    calls = {}

    runtime = ModuleType("onnxruntime")

    def inference_session(path, *, providers):
        calls["path"] = path
        calls["providers"] = providers
        return fake_session

    runtime.InferenceSession = inference_session
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return OnnxPolicy(model), fake_session, calls


def test_accepts_static_actor_contract():
    validate_tensor_contract("input", [1, 45], "tensor(float)", 45)
    validate_tensor_contract("output", [1, 12], "tensor(float)", 12)


@pytest.mark.parametrize(
    ("shape", "tensor_type", "message"),
    [
        ([1, 44], "tensor(float)", "feature dimension"),
        ([1, 45], "tensor(double)", "float32"),
        ([45], "tensor(float)", "rank 2"),
    ],
)
def test_rejects_invalid_contract(shape, tensor_type, message):
    with pytest.raises(ValueError, match=message):
        validate_tensor_contract("input", shape, tensor_type, 45)


@pytest.mark.parametrize("batch_dimension", (None, "batch", "batch_size", "N"))
def test_accepts_dynamic_batch_dimension(batch_dimension):
    validate_tensor_contract(
        "input", [batch_dimension, 45], "tensor(float)", 45
    )


def test_rejects_static_batch_larger_than_one():
    with pytest.raises(ValueError, match="batch dimension"):
        validate_tensor_contract("input", [2, 45], "tensor(float)", 45)


@pytest.mark.parametrize("shape", (None, 45))
def test_rejects_malformed_shape_with_context(shape):
    with pytest.raises(ValueError, match="rank 2"):
        validate_tensor_contract("input", shape, "tensor(float)", 45)


def test_missing_model_path_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnnxPolicy(tmp_path / "missing.onnx")


def test_constructor_uses_cpu_provider_and_exact_metadata_names(
    tmp_path, monkeypatch
):
    policy, _, calls = _make_policy(tmp_path, monkeypatch)
    assert policy.path == (tmp_path / "actor.onnx").resolve()
    assert policy.input_name == "observation"
    assert policy.output_name == "action"
    assert calls == {
        "path": str((tmp_path / "actor.onnx").resolve()),
        "providers": ["CPUExecutionProvider"],
    }


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        ([], [_meta("action", [1, 12])]),
        (
            [_meta("one", [1, 45]), _meta("two", [1, 45])],
            [_meta("action", [1, 12])],
        ),
        ([_meta("observation", [1, 45])], []),
        (
            [_meta("observation", [1, 45])],
            [_meta("one", [1, 12]), _meta("two", [1, 12])],
        ),
    ],
)
def test_constructor_requires_exactly_one_input_and_output(
    tmp_path, monkeypatch, inputs, outputs
):
    session = FakeSession()
    session._inputs = inputs
    session._outputs = outputs
    with pytest.raises(ValueError, match="exactly one input and one output"):
        _make_policy(tmp_path, monkeypatch, session)


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        (
            [_meta("observation", [1, 44])],
            [_meta("action", [1, 12])],
            "input.*feature dimension",
        ),
        (
            [_meta("observation", [1, 45])],
            [_meta("action", [1, 12], "tensor(double)")],
            "output.*float32",
        ),
    ],
)
def test_constructor_validates_input_and_output_contracts(
    tmp_path, monkeypatch, inputs, outputs, message
):
    session = FakeSession(inputs=inputs, outputs=outputs)
    with pytest.raises(ValueError, match=message):
        _make_policy(tmp_path, monkeypatch, session)


@pytest.mark.parametrize(
    "observation",
    (
        np.arange(45, dtype=np.float64),
        np.arange(90, dtype=np.float64).reshape(1, 90)[:, ::2],
    ),
)
def test_infer_feeds_exact_contiguous_float32_tensor_and_returns_vector(
    tmp_path, monkeypatch, observation
):
    expected_action = np.arange(12, dtype=np.float64).reshape(1, 12)
    session = FakeSession(result=[expected_action])
    policy, _, _ = _make_policy(tmp_path, monkeypatch, session)

    action = policy.infer(observation)

    assert action.shape == (12,)
    assert action.dtype == np.float32
    assert action.flags.c_contiguous
    np.testing.assert_array_equal(action, np.arange(12, dtype=np.float32))
    output_names, feeds = session.run_calls[-1]
    assert output_names == ["action"]
    assert set(feeds) == {"observation"}
    feed = feeds["observation"]
    assert feed.shape == (1, 45)
    assert feed.dtype == np.float32
    assert feed.flags.c_contiguous
    np.testing.assert_array_equal(
        feed, np.asarray(observation, dtype=np.float32).reshape(1, 45)
    )


@pytest.mark.parametrize(
    "observation",
    (
        np.zeros(44),
        np.zeros((45, 1)),
        np.zeros((2, 45)),
    ),
)
def test_infer_rejects_wrong_input_shape(tmp_path, monkeypatch, observation):
    policy, _, _ = _make_policy(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"observation.*\(1, 45\)"):
        policy.infer(observation)


def test_infer_rejects_nonfinite_input(tmp_path, monkeypatch):
    policy, _, _ = _make_policy(tmp_path, monkeypatch)
    observation = np.zeros(45)
    observation[0] = np.nan
    with pytest.raises(ValueError, match="observation.*finite"):
        policy.infer(observation)


def test_infer_normalizes_unconvertible_input(tmp_path, monkeypatch):
    policy, _, _ = _make_policy(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="observation") as exc_info:
        policy.infer(["invalid"] * 45)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_infer_rejects_input_float32_overflow_without_warning(
    tmp_path, monkeypatch
):
    policy, _, _ = _make_policy(tmp_path, monkeypatch)
    observation = np.zeros(45)
    observation[0] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="observation.*float32"):
        policy.infer(observation)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([np.zeros((12,), dtype=np.float32)], r"\(1, 12\)"),
        ([np.full((1, 12), np.nan, dtype=np.float32)], "finite"),
        ([["invalid"] * 12], "output"),
        ([np.array([[10**1000] + [0] * 11], dtype=object)], "output"),
        ([np.array([[np.finfo(np.float64).max] + [0] * 11])], "float32"),
    ],
)
def test_infer_rejects_malformed_policy_output(
    tmp_path, monkeypatch, result, message
):
    session = FakeSession(result=result)
    policy, _, _ = _make_policy(tmp_path, monkeypatch, session)
    with pytest.raises(RuntimeError, match=message):
        policy.infer(np.zeros(45))


def test_warmup_runs_one_valid_zero_observation(tmp_path, monkeypatch):
    policy, session, _ = _make_policy(tmp_path, monkeypatch)

    policy.warmup()

    assert len(session.run_calls) == 1
    output_names, feeds = session.run_calls[0]
    assert output_names == ["action"]
    np.testing.assert_array_equal(
        feeds["observation"], np.zeros((1, 45), dtype=np.float32)
    )
