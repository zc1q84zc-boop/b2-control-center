from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from b2_control.terrain_scene import (
    TERRAINS,
    render_scene,
    render_scene_with_timestep,
    stage_scene,
)


def test_catalog_contains_requested_surface_and_stair_scenes() -> None:
    ids = {terrain.id for terrain in TERRAINS}
    assert {"flat", "grass", "mud", "gravel", "mixed"} <= ids
    assert {"stairs_10cm", "stairs_15cm", "stairs_20cm", "stairs_reference_12cm"} <= ids


@pytest.mark.parametrize("scene", TERRAINS, ids=lambda scene: scene.id)
def test_every_generated_scene_is_well_formed_mjcf(scene) -> None:
    root = ET.fromstring(render_scene(scene.id))
    assert root.tag == "mujoco"
    assert root.find("include").attrib["file"] == "b2.xml"
    assert root.find("option").attrib["timestep"] == "0.002"
    assert root.find("worldbody/geom[@name='floor']") is not None


def test_surface_contact_models_have_distinct_friction() -> None:
    grass = ET.fromstring(render_scene("grass"))
    mud = ET.fromstring(render_scene("mud"))
    assert grass.find("worldbody/geom[@name='floor']").attrib["friction"] == (
        "0.95 0.006 0.0001"
    )
    assert mud.find("worldbody/geom[@name='floor']").attrib["friction"] == (
        "0.28 0.003 0.0001"
    )


@pytest.mark.parametrize(
    ("scene_id", "expected_height", "expected_steps"),
    [("stairs_10cm", 0.10, 6), ("stairs_15cm", 0.15, 6), ("stairs_20cm", 0.20, 5)],
)
def test_stair_geometries_match_named_riser_height(
    scene_id: str, expected_height: float, expected_steps: int
) -> None:
    root = ET.fromstring(render_scene(scene_id))
    stairs = root.findall("worldbody/geom")
    stairs = [geom for geom in stairs if geom.attrib.get("name", "").startswith("stair_")]
    assert len(stairs) == expected_steps
    first_half_height = float(stairs[0].attrib["size"].split()[2])
    assert first_half_height * 2 == pytest.approx(expected_height)


def test_reference_stair_geometry_matches_stack10_scenario() -> None:
    root = ET.fromstring(render_scene("stairs_reference_12cm"))
    stairs = [
        geom
        for geom in root.findall("worldbody/geom")
        if geom.attrib.get("name", "").startswith("stair_")
    ]
    assert len(stairs) == 5
    assert float(stairs[0].attrib["size"].split()[2]) * 2 == pytest.approx(0.12)
    assert float(stairs[0].attrib["size"].split()[0]) * 2 == pytest.approx(0.356)


def test_stage_scene_uses_links_to_official_assets(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "b2.xml").write_text("<mujoco/>", encoding="utf-8")
    (model / "assets").mkdir()
    staging = tmp_path / "staging"
    scene = stage_scene("flat", model, staging)
    assert scene.is_file()
    assert (staging / "b2.xml").resolve() == (model / "b2.xml").resolve()
    assert (staging / "assets").resolve() == (model / "assets").resolve()


def test_reference_timestep_can_match_stack10_runtime() -> None:
    root = ET.fromstring(render_scene_with_timestep("flat", 0.005))
    assert root.find("option").attrib["timestep"] == "0.005"
