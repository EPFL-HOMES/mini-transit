import pytest
from datetime import datetime, timedelta

from src.network import Network
# 🔁 CHANGE THIS to the actual module where FixedRouteService lives
from src.services.fixedroute import FixedRouteService
from src.hex import Hex


class DummyNetwork(Network):
    """
    Simple network where every move between two different stops has distance 1,
    and staying at the same stop has distance 0.
    """
    def get_distance(self, a, b):
        return 0 if a == b else 1


def create_basic_service(bidirectional=True) -> FixedRouteService:
    """
    Helper to build a simple FixedRouteService with 3 stops and one departure hour.
    """
    network = Network(geojson_file_path="data/Lausanne/Lausanne.geojson")
    stops = [Hex(0), Hex(1), Hex(2)]
    period_start = datetime(2025, 1, 1, 8, 0)
    period_end = datetime(2025, 1, 1, 8, 0)  # single departure
    frequency = timedelta(hours=1)

    service = FixedRouteService(
        name="test_service",
        stops=stops,
        capacity=10,
        stopping_time=timedelta(minutes=5),
        travel_time=timedelta(minutes=10),  # per unit distance
        network=network,
        freq_period=[(period_start, period_end, frequency)],
        bidirectional=bidirectional,
    )
    return service


def test_timetables_forward_and_backward():
    """
    For a bidirectional line with one departure in the period, we expect
    2 vehicles: one forward A->B->C and one backward C->B->A.
    """
    service = create_basic_service(bidirectional=True)

    # One period, one departure time, bidirectional => 2 vehicles
    assert len(service.vehicles) == 2

    v_forward, v_backward = service.vehicles

    # Check stop order via timetable keys (indices)
    assert list(v_forward.timetable.keys()) == [0, 1, 2]
    assert list(v_backward.timetable.keys()) == [2, 1, 0]

    # Check that arrival times are strictly non-decreasing within each vehicle
    for vehicle in service.vehicles:
        arrivals = [arr for (arr, _) in vehicle.timetable.values()]
        assert arrivals == sorted(arrivals)


def test_forward_timetable_times_are_correct():
    """
    With travel_time=10min and stopping_time=5min and distance=1 between stops:

      departure at stop 0 at 08:00
      08:00-08:05 dwell at stop 0
      08:05-08:15 travel to stop 1
      08:15-08:20 dwell at stop 1
      08:20-08:30 travel to stop 2
      08:30-08:35 dwell at stop 2 (departure)
    """
    service = create_basic_service(bidirectional=False)
    assert len(service.vehicles) == 1
    vehicle = service.vehicles[0]

    t0_arr, t0_dep = vehicle.timetable[0]
    t1_arr, t1_dep = vehicle.timetable[1]
    t2_arr, t2_dep = vehicle.timetable[2]

    base = datetime(2025, 1, 1, 8, 0)

    assert t0_arr == base
    assert t0_dep == base + timedelta(minutes=5)

    assert t1_arr == t0_dep + timedelta(minutes=10)
    assert t1_dep == t1_arr + timedelta(minutes=5)

    assert t2_arr == t1_dep + timedelta(minutes=10)
    assert t2_dep == t2_arr + timedelta(minutes=5)


def test_get_next_departure_forward_direction():
    """
    Using the private __get_next_departure to ensure direction and timing are correct.
    """
    service = create_basic_service(bidirectional=True)
    start_index = 0  # "A"
    end_index = 2    # "C"

    # Request after 08:02; first departure at stop 0 is at 08:05
    current_time = datetime(2025, 1, 1, 8, 2)

    # Access the mangled private method
    vehicle = service._FixedRouteService__get_next_departure(
        current_time=current_time,
        start_index=start_index,
        end_index=end_index,
    )

    arr_start, dep_start = vehicle.timetable[start_index]
    arr_end, _ = vehicle.timetable[end_index]

    # It should depart after current_time
    assert dep_start >= current_time
    # And reach the destination after departure from the origin
    assert arr_end > dep_start


def test_get_route_returns_wait_and_ride_in_order():
    """
    get_route should give a Wait and Ride where:
      - ride starts when waiting ends
      - arrival is after departure
    """
    service = create_basic_service(bidirectional=True)
    start_time = datetime(2025, 1, 1, 8, 2)
    start_hex = service.stops[0]  # "A"
    end_hex = service.stops[2]    # "C"

    wait, ride = service.get_route(
        unit=3,
        start_time=start_time,
        start_hex=start_hex,
        end_hex=end_hex,
    )

    # Basic sanity checks – these rely on typical Wait/Ride attributes.
    # Adapt attribute names if your implementations differ.
    assert ride.service is service
    assert ride.unit == 3

    # ride must start after we start waiting, and after waiting ends
    assert wait.start_time == start_time
    assert wait.end_time == ride.start_time
    assert ride.end_time > ride.start_time