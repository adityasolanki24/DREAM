import json
import pathlib

from experiments import isaac_kitchen_repair as k

BASE = "../../SimReady/Residential/Kitchen/Appliances/"


def test_refrigerator_reference_is_remapped_to_the_existing_file():
    old = BASE + "Refrigerator_A01/sm_rc_appliance_refrigerator_a01_01.usd"
    assert k.repair_reference(old) == BASE + "Refrigerator_A01/sm_refrigerator_a01_01.usd"
    old2 = BASE + "Refrigerator_A01_2/sm_rc_appliance_refrigerator_a01_01.usd"
    assert k.repair_reference(old2) == BASE + "Refrigerator_A01/sm_refrigerator_a01_01.usd"


def test_the_repair_target_is_a_validated_asset_path():
    manifest = json.loads(
        pathlib.Path(k.__file__)
        .parents[1]
        .joinpath("experiments/configs/isaac_asset_candidates.json")
        .read_text()
    )
    fixed = k.repair_reference(BASE + "Refrigerator_A01/sm_rc_appliance_refrigerator_a01_01.usd")
    assert any(fixed.endswith(c.split("SimReady/")[1]) for c in manifest["classes"]["refrigerator"])


def test_oven_is_reported_unrepairable_and_other_references_untouched():
    oven = BASE + "Oven_A01/sm_rc_appliance_oven_a01_01.usd"
    assert k.repair_reference(oven) is None and k.unrepairable_reason(oven)
    assert k.repair_reference("../Props/anything.usd") is None
