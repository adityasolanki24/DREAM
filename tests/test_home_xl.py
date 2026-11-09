"""The ``xl`` generator (12-room homes) and the guarantee that adding it left the existing
``nominal`` and ``shifted`` generators byte-for-byte unchanged."""

import dataclasses
import hashlib

import numpy as np
import pytest

import dream.sim.home as H

# sha256 of repr(HomeSpec)[:16], taken on the commit before the xl generator existed:
# sample_home(default_rng([seed, 0]), scenario, "stove", generator).
GOLDEN = {
    ("nominal", "A", 3): "9810c00034146418", ("nominal", "A", 17): "653041f33b7c0f5b",
    ("nominal", "B", 3): "439010651a019b01", ("nominal", "B", 17): "bb1d058ab88b5e81",
    ("nominal", "D", 3): "83a42f58203ab812", ("nominal", "D", 17): "c97c023369bdc669",
    ("nominal", "E", 3): "401dafdd319fa25b", ("nominal", "E", 17): "db5a4fc6326cf928",
    ("shifted", "A", 3): "7e99db8d7ff7e443", ("shifted", "A", 17): "045d9b576f50a462",
    ("shifted", "B", 3): "2c19ac431408cb94", ("shifted", "B", 17): "d4841b0d8b89cad4",
    ("shifted", "D", 3): "d5c040b72f0cdd09", ("shifted", "D", 17): "22f43d27c9f34523",
    ("shifted", "E", 3): "b98fddd70ffaba1f", ("shifted", "E", 17): "288deef7ef8e9a0f",
}  # fmt: skip
TARGET_NAMES = tuple(H.TARGETS)  # the eight search targets


def _home(scenario, seed, target="stove", generator="xl"):
    return H.sample_home(np.random.default_rng([seed, 0]), scenario, target, generator)


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_existing_generators_are_byte_identical(key):
    generator, scenario, seed = key
    digest = hashlib.sha256(repr(_home(scenario, seed, "stove", generator)).encode())
    assert digest.hexdigest()[:16] == GOLDEN[key]


def test_public_output_format_is_unchanged():
    assert [f.name for f in dataclasses.fields(H.HomeSpec)] == [
        "width", "depth", "rooms", "doorways", "objects", "start", "scenario", "target_class",
        "request", "memory_prior", "generator", "seed", "walls",
    ]  # fmt: skip
    assert _home("B", 100).generator == "xl"


def _check_xl(scenario, seed, target):
    home = _home(scenario, seed, target)
    types = [r.type for r in home.rooms]
    assert len(home.rooms) == 12
    assert set(H.ROOM_TYPES) <= set(types)  # every room type the generator knows
    assert types.count("bedroom") >= 3 and types.count("bathroom") >= 2
    assert H.problems(home) == []  # every room reachable, every object has a standing cell
    assert H.verify_scenario(home) == []
    info = H.TARGETS[target]
    present = H.instances_of(home, target)
    if scenario == "E":
        assert not present
    else:
        assert len(present) == 1
        room_type = types[present[0].room_id]
        assert (room_type == info.typical) if scenario == "B" else (room_type in info.atypical)
    assert H.EXTENT_LIMIT >= home.width / 2 + abs(home.start[0])
    return home


@pytest.mark.parametrize("scenario", ["B", "D", "E"])
def test_xl_homes_are_valid(scenario):
    _check_xl(scenario, 100, "washing_machine")
    _check_xl(scenario, 101, "refrigerator")


def test_xl_is_larger_than_the_nominal_homes():
    xl = _home("B", 100)
    nominal = _home("B", 100, generator="nominal")
    assert xl.width * xl.depth > 1.8 * nominal.width * nominal.depth


@pytest.mark.slow
@pytest.mark.parametrize("target", TARGET_NAMES)
def test_xl_dev_seeds_all_targets(target):
    for seed in (100, 101, 102, 103):
        for scenario in "BDE":
            _check_xl(scenario, seed, target)
