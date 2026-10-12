from pathlib import Path

import numpy as np

from b2_control.bundle import open_verified_bundle
from b2_control.core import B2PolicyCore
from b2_control.onnx_policy import OnnxPolicy
from b2_control.stack10_adapter import Q_DEFAULT, Stack10ObservationAdapter


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "policies" / "b2_stack10_resume_s42_model31997"


def test_stack10_bundle_contract_and_golden_reset() -> None:
    bundle = open_verified_bundle(BUNDLE)
    assert bundle.adapter == "b2_stack10_v1"
    assert bundle.input_dim == 423
    policy = OnnxPolicy(bundle.policy_path, input_dim=bundle.input_dim)
    adapter = Stack10ObservationAdapter()
    core = B2PolicyCore(policy, observation_adapter=adapter, target_slew_rad_s=None)
    core.reset()
    first = core.step(
        omega_body_rad_s=np.zeros(3),
        rotation_body_to_world=np.eye(3),
        joint_position_rad=Q_DEFAULT,
        joint_velocity_rad_s=np.zeros(12),
        target_yaw_rad=0.0,
        current_yaw_rad=0.0,
        nominal_speed_m_s=0.0,
        hold=True,
    )
    contract = __import__("json").loads(
        (BUNDLE / "policy_contract.json").read_text(encoding="utf-8")
    )
    np.testing.assert_allclose(
        first.raw_action,
        np.asarray(contract["golden_reset_inference"]["tick_1_action"], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    assert first.observation.shape == (1, 423)
