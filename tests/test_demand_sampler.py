# tests/test_demand_sampler.py

from typing import List

import pytest

from ..src.minitransit_simulation.demand.sampler import DemandSampler
from ..src.minitransit_simulation.models import DemandInput
from ..src.minitransit_simulation.primitives.hex import Hex

# ---------- Fixtures ----------


@pytest.fixture
def hexes() -> List[Hex]:
    """Provide a small reusable set of Hex objects."""
    return [Hex(hex_id=i) for i in range(1, 5)]
    # hexes[0] -> Hex(hex_id=1)
    # hexes[1] -> Hex(hex_id=2)
    # hexes[2] -> Hex(hex_id=3)
    # hexes[3] -> Hex(hex_id=4)


# ---------- Tests ----------


def test_zero_lambda_produces_no_demands(hexes):
    hex1, hex2, hex3, hex4 = hexes

    sampler = DemandSampler(unit_sizes=[1, 2, 3], seed=42)

    inputs: List[DemandInput] = [
        DemandInput(hour=0, start_hex=hex1, end_hex=hex2, unit=0),
        DemandInput(hour=1, start_hex=hex3, end_hex=hex4, unit=0),
    ]

    demands = sampler.sample_hourly_demand(inputs)

    assert demands == [] or len(demands) == 0


def test_units_are_integers_and_from_unit_sizes(hexes):
    unit_sizes = [1, 2, 5]
    sampler = DemandSampler(unit_sizes=unit_sizes, seed=123)

    hex1, hex2, hex3, _ = hexes

    inputs = [
        # λ = 5 for OD (hex1 -> hex2)
        DemandInput(hour=0, start_hex=hex1, end_hex=hex2, unit=5),
        # λ = 10 for OD (hex1 -> hex3)
        DemandInput(hour=0, start_hex=hex1, end_hex=hex3, unit=10),
    ]

    demands = sampler.sample_hourly_demand(inputs)

    # We don't know exactly how many demands (Poisson),
    # but for λ>0 we expect possibly >0; the main thing
    # is that they have the correct OD and unit types.
    for d in demands:
        # All demands originate from hex1
        assert d.start_hex is hex1
        # Destinations are either hex2 or hex3
        assert d.end_hex in (hex2, hex3)

        # unit should be an int and one of the configured sizes
        assert isinstance(d.unit, int)
        assert d.unit in unit_sizes


def test_times_are_within_the_correct_hour_for_numeric_hour(hexes):
    """
    When hour is numeric (e.g. 0, 1, 2, ...), DemandSampler should
    generate times hour + U(0,1).
    """
    sampler = DemandSampler(unit_sizes=[1], seed=999)

    hour = 5
    hex1, hex2, *_ = hexes

    inputs = [DemandInput(hour=hour, start_hex=hex1, end_hex=hex2, unit=20)]

    demands = sampler.sample_hourly_demand(inputs)

    for d in demands:
        # time must be within [hour, hour+1)
        assert d.time.hour >= hour
        assert d.time.hour < hour + 1


def test_multiple_inputs_keep_correct_ods(hexes):
    """
    If we have multiple OD pairs, each demand should keep the OD from its input row.
    """
    sampler = DemandSampler(unit_sizes=[1], seed=777)

    hex1, hex2, hex3, hex4 = hexes

    inputs = [
        DemandInput(hour=0, start_hex=hex1, end_hex=hex2, unit=15),
        DemandInput(hour=0, start_hex=hex3, end_hex=hex4, unit=25),
    ]

    demands = sampler.sample_hourly_demand(inputs)

    valid_pairs = {(hex1, hex2), (hex3, hex4)}

    for d in demands:
        assert (d.start_hex, d.end_hex) in valid_pairs
