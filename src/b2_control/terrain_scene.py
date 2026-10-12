"""Deterministic MuJoCo terrain scenes for the B2 control center.

The generated files are written beside Unitree's B2 model at runtime so the
official ``b2.xml`` include and its relative mesh paths remain unchanged.
These scenes model useful contact conditions, not calibrated replicas of a
particular outdoor site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import re


@dataclass(frozen=True)
class TerrainSpec:
    id: str
    name: str
    description: str
    difficulty: str
    category: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


TERRAINS: tuple[TerrainSpec, ...] = (
    TerrainSpec("flat", "标准平地", "标准摩擦系数平面，基础回归测试。", "基础", "平地"),
    TerrainSpec("grass", "草地近似", "高摩擦、轻微软接触和毫米级起伏。", "中等", "软地面"),
    TerrainSpec("mud", "泥路近似", "低摩擦、软接触地面，用于打滑敏感性测试。", "困难", "软地面"),
    TerrainSpec("gravel", "石子路", "确定性散布的低矮石块和不平接触。", "困难", "非结构地形"),
    TerrainSpec("stairs_10cm", "10 cm 楼梯", "六级、踏面 38 cm、级高 10 cm。", "中等", "楼梯"),
    TerrainSpec("stairs_15cm", "15 cm 楼梯", "六级、踏面 38 cm、级高 15 cm。", "困难", "楼梯"),
    TerrainSpec("stairs_20cm", "20 cm 楼梯", "五级、踏面 42 cm、级高 20 cm。", "极限", "楼梯"),
    TerrainSpec(
        "stairs_reference_12cm",
        "参考 12 cm 楼梯",
        "Stack10 README 参考场景：五级、踏面 35 cm、级高 12 cm。",
        "参考",
        "楼梯",
    ),
    TerrainSpec("ramp_12deg", "12° 斜坡", "长 3 m 的 12° 上坡和顶部平台。", "中等", "坡道"),
    TerrainSpec("mixed", "综合地形", "草地、泥路、石块和 10 cm 台阶连续组合。", "综合", "测试路线"),
)

_BY_ID = {terrain.id: terrain for terrain in TERRAINS}
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def terrain_catalog() -> list[dict[str, str]]:
    return [terrain.to_dict() for terrain in TERRAINS]


def get_terrain(scene_id: str) -> TerrainSpec:
    try:
        return _BY_ID[scene_id]
    except KeyError as exc:
        raise ValueError(f"unknown terrain scene: {scene_id}") from exc


def _geom(**attributes: object) -> str:
    encoded = " ".join(
        f'{name.rstrip("_").replace("__", ":")}="{value}"'
        for name, value in attributes.items()
    )
    return f"    <geom {encoded}/>"


def _stairs(step_height: float, *, steps: int, tread: float, start: float) -> list[str]:
    geoms: list[str] = []
    for index in range(steps):
        height = step_height * (index + 1)
        center_x = start + tread * (index + 0.5)
        geoms.append(
            _geom(
                name=f"stair_{index + 1:02d}",
                type="box",
                pos=f"{center_x:.4f} 0 {height / 2:.4f}",
                size=f"{tread / 2 + 0.003:.4f} 1.60 {height / 2:.4f}",
                material="stair_alt" if index % 2 else "stair",
                friction="0.85 0.005 0.0001",
            )
        )
    top_height = step_height * steps
    top_center = start + tread * steps + 0.75
    geoms.append(
        _geom(
            name="stairs_top",
            type="box",
            pos=f"{top_center:.4f} 0 {top_height / 2:.4f}",
            size=f"0.75 1.60 {top_height / 2:.4f}",
            material="top",
            friction="0.85 0.005 0.0001",
        )
    )
    return geoms


def _gravel(*, start: float = 0.9, length: float = 5.3) -> list[str]:
    rng = random.Random(20260813)
    geoms: list[str] = []
    for index in range(78):
        x = start + rng.random() * length
        y = -1.05 + rng.random() * 2.10
        sx = rng.uniform(0.035, 0.085)
        sy = rng.uniform(0.025, 0.070)
        sz = rng.uniform(0.012, 0.032)
        yaw = rng.uniform(-math.pi, math.pi)
        geoms.append(
            _geom(
                name=f"stone_{index:03d}",
                type="ellipsoid",
                pos=f"{x:.4f} {y:.4f} {sz * 0.72:.4f}",
                size=f"{sx:.4f} {sy:.4f} {sz:.4f}",
                euler=f"0 0 {yaw:.4f}",
                material="stone_a" if index % 2 else "stone_b",
                friction="0.75 0.008 0.0002",
            )
        )
    return geoms


def _grass_bumps() -> list[str]:
    geoms: list[str] = []
    for index in range(18):
        x = 0.9 + index * 0.30
        y = (-0.65, 0.0, 0.65)[index % 3]
        height = (0.004, 0.007, 0.010)[index % 3]
        geoms.append(
            _geom(
                name=f"grass_bump_{index:02d}",
                type="box",
                pos=f"{x:.3f} {y:.3f} {height / 2:.4f}",
                size=f"0.18 0.34 {height / 2:.4f}",
                material="grass_dark" if index % 2 else "grass",
                friction="0.95 0.006 0.0001",
                solref="0.025 1",
                solimp="0.85 0.95 0.002",
            )
        )
    return geoms


def _ramp() -> list[str]:
    angle = math.radians(-12.0)
    half_length = 1.5
    half_height = 0.045
    center_x = 2.5
    center_z = half_length * math.sin(-angle) + half_height
    quat = f"{math.cos(angle / 2):.7f} 0 {math.sin(angle / 2):.7f} 0"
    top_z = 2 * half_length * math.sin(-angle)
    return [
        _geom(
            name="ramp",
            type="box",
            pos=f"{center_x:.4f} 0 {center_z:.4f}",
            size=f"{half_length:.4f} 1.40 {half_height:.4f}",
            quat=quat,
            material="ramp",
            friction="0.85 0.005 0.0001",
        ),
        _geom(
            name="ramp_top",
            type="box",
            pos=f"4.65 0 {top_z / 2:.4f}",
            size=f"0.65 1.40 {top_z / 2:.4f}",
            material="top",
            friction="0.85 0.005 0.0001",
        ),
    ]


def _mixed() -> list[str]:
    geoms = [
        _geom(
            name="grass_patch",
            type="box",
            pos="1.35 0 0.004",
            size="0.55 1.20 0.004",
            material="grass",
            friction="0.95 0.006 0.0001",
            solref="0.025 1",
        ),
        _geom(
            name="mud_patch",
            type="box",
            pos="2.45 0 0.003",
            size="0.55 1.20 0.003",
            material="mud",
            friction="0.28 0.003 0.0001",
            solref="0.05 1",
            solimp="0.75 0.93 0.006",
        ),
    ]
    for index in range(16):
        x = 3.10 + (index % 4) * 0.22
        y = -0.75 + (index // 4) * 0.50
        size = 0.018 + (index % 3) * 0.006
        geoms.append(
            _geom(
                name=f"mixed_stone_{index:02d}",
                type="ellipsoid",
                pos=f"{x:.3f} {y:.3f} {size * .75:.4f}",
                size=f"{size * 1.7:.4f} {size * 1.25:.4f} {size:.4f}",
                material="stone_a",
                friction="0.75 0.008 0.0002",
            )
        )
    geoms.extend(_stairs(0.10, steps=4, tread=0.40, start=4.20))
    return geoms


def render_scene(scene_id: str) -> str:
    """Render one official-B2-compatible MJCF scene."""
    return render_scene_with_timestep(scene_id, 0.002)


def render_scene_with_timestep(scene_id: str, physics_timestep: float) -> str:
    """Render a scene with an explicit MuJoCo physics timestep."""

    terrain = get_terrain(scene_id)
    if not math.isfinite(physics_timestep) or physics_timestep <= 0.0:
        raise ValueError("physics_timestep must be finite and positive")
    floor = _geom(
        name="floor",
        type="plane",
        size="0 0 0.05",
        material=(
            "grass" if scene_id == "grass" else "mud" if scene_id == "mud" else "ground"
        ),
        friction=(
            "0.95 0.006 0.0001"
            if scene_id == "grass"
            else "0.28 0.003 0.0001"
            if scene_id == "mud"
            else "0.80 0.005 0.0001"
        ),
        solref="0.05 1" if scene_id == "mud" else "0.02 1",
        solimp="0.75 0.93 0.006" if scene_id == "mud" else "0.9 0.95 0.001",
    )
    extras: list[str] = []
    if scene_id == "grass":
        extras = _grass_bumps()
    elif scene_id == "gravel":
        extras = _gravel()
    elif scene_id == "stairs_reference_12cm":
        # Match the Stack10 reference scenario: 5 risers, 0.35 m tread.
        extras = _stairs(0.12, steps=5, tread=0.35, start=-0.875)
    elif scene_id.startswith("stairs_"):
        height = int(scene_id.removeprefix("stairs_").removesuffix("cm")) / 100.0
        extras = _stairs(
            height,
            steps=5 if height >= 0.20 else 6,
            tread=0.42 if height >= 0.20 else 0.38,
            start=1.20,
        )
    elif scene_id == "ramp_12deg":
        extras = _ramp()
    elif scene_id == "mixed":
        extras = _mixed()

    return f'''<mujoco model="B2 Control Center - {terrain.name}">
  <include file="b2.xml"/>
  <option timestep="{physics_timestep:.9g}"/>
  <statistic center="2.8 0 0.45" extent="4.8"/>
  <visual>
    <headlight diffuse="0.72 0.72 0.72" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.22 0.28 1"/>
    <global azimuth="-125" elevation="-23"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.52 0.72" rgb2="0.02 0.03 0.05" width="512" height="3072"/>
    <texture type="2d" name="ground_tex" builtin="checker" mark="edge" rgb1="0.20 0.29 0.34" rgb2="0.10 0.16 0.20" markrgb="0.65 0.72 0.76" width="300" height="300"/>
    <material name="ground" texture="ground_tex" texuniform="true" texrepeat="12 12" reflectance="0.10"/>
    <material name="grass" rgba="0.20 0.43 0.18 1"/>
    <material name="grass_dark" rgba="0.13 0.31 0.12 1"/>
    <material name="mud" rgba="0.31 0.19 0.10 1"/>
    <material name="stone_a" rgba="0.39 0.41 0.42 1"/>
    <material name="stone_b" rgba="0.23 0.26 0.28 1"/>
    <material name="stair" rgba="0.69 0.50 0.23 1"/>
    <material name="stair_alt" rgba="0.50 0.34 0.15 1"/>
    <material name="ramp" rgba="0.48 0.51 0.55 1"/>
    <material name="top" rgba="0.27 0.47 0.28 1"/>
  </asset>
  <worldbody>
    <light pos="2 -2 5" dir="0.1 0.15 -1" directional="true"/>
{floor}
{chr(10).join(extras)}
  </worldbody>
</mujoco>
'''


def stage_scene(
    scene_id: str,
    robot_model_dir: Path,
    output_dir: Path | None = None,
    physics_timestep: float = 0.002,
) -> Path:
    """Stage an MJCF scene with links to the official B2 model assets."""

    get_terrain(scene_id)
    model_dir = robot_model_dir.expanduser().resolve()
    if not (model_dir / "b2.xml").is_file():
        raise ValueError(f"Unitree B2 model directory is invalid: {model_dir}")
    if not _SAFE_ID.fullmatch(scene_id):
        raise ValueError("terrain id contains unsafe characters")
    staging = (
        Path.home() / ".cache" / "b2-control-center" / "scenes"
        if output_dir is None
        else output_dir.expanduser().resolve()
    )
    staging.mkdir(parents=True, exist_ok=True)
    for name in ("b2.xml", "assets"):
        link = staging / name
        source = model_dir / name
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"scene staging path is occupied: {link}")
        link.symlink_to(source)
    target = staging / f"scene_control_center_{scene_id}.xml"
    target.write_text(
        render_scene_with_timestep(scene_id, physics_timestep), encoding="utf-8"
    )
    return target


__all__ = [
    "TERRAINS",
    "TerrainSpec",
    "get_terrain",
    "render_scene",
    "render_scene_with_timestep",
    "stage_scene",
    "terrain_catalog",
]
