"""Low-overhead runtime and physical-network counters for the operator UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time


def _read_counter(interface: str, name: str) -> int:
    try:
        return int(
            Path(f"/sys/class/net/{interface}/statistics/{name}")
            .read_text(encoding="ascii")
            .strip()
        )
    except (OSError, ValueError):
        return 0


@dataclass
class RuntimeMetricsSampler:
    """Sample this policy process and one Linux network interface."""

    interface: str | None = None

    def __post_init__(self) -> None:
        self._wall = time.monotonic()
        self._cpu = time.process_time()
        self._rx_packets = self._network("rx_packets")
        self._tx_packets = self._network("tx_packets")
        self._rx_dropped = self._network("rx_dropped")
        self._tx_dropped = self._network("tx_dropped")

    def _network(self, name: str) -> int:
        return 0 if not self.interface else _read_counter(self.interface, name)

    def sample(self) -> dict[str, float | int]:
        now = time.monotonic()
        cpu_now = time.process_time()
        wall_delta = max(now - self._wall, 1e-9)
        cpu_percent = max(0.0, 100.0 * (cpu_now - self._cpu) / wall_delta)

        rx_packets = self._network("rx_packets")
        tx_packets = self._network("tx_packets")
        rx_dropped = self._network("rx_dropped")
        tx_dropped = self._network("tx_dropped")
        packet_delta = max(
            0, rx_packets - self._rx_packets
        ) + max(0, tx_packets - self._tx_packets)
        drop_delta = max(
            0, rx_dropped - self._rx_dropped
        ) + max(0, tx_dropped - self._tx_dropped)
        loss_percent = 100.0 * drop_delta / max(packet_delta + drop_delta, 1)

        self._wall = now
        self._cpu = cpu_now
        self._rx_packets = rx_packets
        self._tx_packets = tx_packets
        self._rx_dropped = rx_dropped
        self._tx_dropped = tx_dropped
        return {
            "policy_cpu_percent": cpu_percent,
            "network_packet_loss_percent": loss_percent,
            "network_rx_dropped": rx_dropped,
            "network_tx_dropped": tx_dropped,
        }


__all__ = ["RuntimeMetricsSampler"]
