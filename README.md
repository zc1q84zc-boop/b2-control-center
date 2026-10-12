# B2 Control Center

Standalone B2 simulation and hardware-diagnostics control console.

## 中文快速入口

- [电脑连接与控制 Unitree B2](docs/PC_CONNECT_B2_ZH.md)
- 控制台源码现已直接展开在 `src/`、`gateway/`、`scripts/` 和 `tests/`；历史压缩包仅
  作为快照保留。
- 首次连接请从只读状态检查开始，不要直接运行低层 `LowCmd`。

## Scope

The console provides simulation, policy loading, keyboard/remote command
mapping, LowState diagnostics, logging, video capture, and a gated hardware
integration path. It starts passively and never connects to or commands a
robot automatically.

The real-robot LowCmd safety checks, emergency-stop handling, watchdogs, and
policy approval gates are part of the design and must not be removed for a
deployment. Use simulation or read-only shadow mode when evaluating a new
policy.

## Requirements

- Linux with Python 3.10+;
- Python dependencies from `pyproject.toml`;
- Unitree SDK2 and official B2 MuJoCo assets for the corresponding local
  simulation/hardware workflow;
- a working OpenGL/X11 display for recorded interactive simulation.

## Start the console

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
scripts/start_b2_control_center.sh --no-open-browser
```

The command prints a loopback URL containing a random access token. Open that
complete URL in a browser. The console is simulation-first; hardware actions
remain disabled until the normal read-only and approval checks pass.

## Policy bundles

Policy bundles are verified by SHA-256 and must declare their input contract.
The included Stack10 bundle uses a 423-value input, 50 Hz inference, and a
ten-frame selective history adapter. Do not replace its ONNX file without its
matching adapter and contract.

## Tests

```bash
PYTHONPATH=src pytest -q
```

If ONNX Runtime is not installed, the policy golden-inference test is expected
to be skipped or reported as an environment failure; install the documented
runtime before accepting a policy artifact.

## Distribution

This repository intentionally excludes captured hardware logs, generated
videos, build directories, local worktrees, and machine-specific runtime
state. Add site-specific hardware profiles outside the public source package.

The checked-in hardware profiles are reference-only and do not authorize real
low-level activation. The software stop path is not a replacement for a
physical emergency stop or mechanical fall protection.
