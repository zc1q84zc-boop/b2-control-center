import time

from b2_control.runtime_metrics import RuntimeMetricsSampler


def test_runtime_metrics_are_finite_without_network_interface() -> None:
    sampler = RuntimeMetricsSampler(None)
    time.sleep(0.001)
    sample = sampler.sample()
    assert sample["policy_cpu_percent"] >= 0.0
    assert sample["network_packet_loss_percent"] == 0.0
    assert sample["network_rx_dropped"] == 0
