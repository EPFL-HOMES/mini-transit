import json
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List
from typing import OrderedDict as TypingOrderedDict
from typing import Tuple

from src.actions import Ride, Wait
from src.hex import Hex
from src.models import NetworkModel
from src.services import Service


class FixedRouteService(Service):
    """
    Represents a fixed-route transportation service.

    Attributes:
        name (str): Name of the service.
        stops (List[Hex]): List of Hex objects representing the stops.
        stop_hex_lookup (Dict[Hex, int]): Mapping from Hex to its index in stops.
        vehicles (List[FixedRouteVehicle]): List of vehicles operating on this route.
        capacity (float): Maximum capacity of each vehicle.
        stopping_time (timedelta): Time spent at each stop.
        travel_time (timedelta): Time taken to travel between hexes (per unit distance).
        freq_period (List[Tuple[datetime, datetime, timedelta]]): List of frequency periods where each tuple contains
            (start_time, end_time, frequency).
        bidirectional (bool): Whether to create reverse-direction services as well.
    """

    def __init__(
        self,
        name: str,
        stops: List[Hex],
        capacity: float,
        stopping_time: timedelta,
        travel_time: timedelta,
        network: NetworkModel,
        freq_period: List[Tuple[datetime, datetime, timedelta]],
        bidirectional: bool = True,
    ):
        super().__init__(name)
        self.stops = stops
        self.stop_hex_lookup: Dict[Hex, int] = {stop: index for index, stop in enumerate(stops)}
        self.capacity = capacity
        self.network = network
        self.stopping_time = stopping_time
        self.travel_time = travel_time
        self.bidirectional = bidirectional

        timetables = self.__freq_to_timetables(freq_period)
        self.vehicles = [
            FixedRouteVehicle(self, timetable=t) for t in timetables
        ]  # List of OrderedDicts mapping stop index to (arrival_time, departure_time)

    # -------------------------------------------------------------------------
    # Timetable construction
    # -------------------------------------------------------------------------
    def __build_timetable_for_direction(
        self,
        first_departure: datetime,
        stop_indices: List[int],
    ) -> TypingOrderedDict[int, Tuple[datetime, datetime]]:
        """
        Build the timetable for a single vehicle given the order of stops.

        Args:
            first_departure: Departure time from the first stop in `stop_indices`.
            stop_indices: Indices of stops in the order they are visited.

        Returns:
            OrderedDict[int, Tuple[datetime, datetime]]: timetable mapping stop index
            to (arrival_time, departure_time).
        """
        timetable: TypingOrderedDict[int, Tuple[datetime, datetime]] = OrderedDict()
        departure_time = first_departure

        for pos, idx in enumerate(stop_indices):
            if pos == 0:
                distance = 0
            else:
                prev_idx = stop_indices[pos - 1]
                distance = self.network.get_distance(self.stops[prev_idx], self.stops[idx])

            # Travel from previous stop, then stop dwell time
            arrival_time = departure_time + distance * self.travel_time
            departure_time = arrival_time + self.stopping_time
            timetable[idx] = (arrival_time, departure_time)

        return timetable

    def __freq_to_timetables(
        self,
        freq_period: List[Tuple[datetime, datetime, timedelta]],
    ) -> List[TypingOrderedDict[int, Tuple[datetime, datetime]]]:
        """
        Convert frequency periods to timetables for vehicles.

        Args:
            freq_period: List of tuples (start_time, end_time, frequency),
                         where frequency is a timedelta giving headway.

        Returns:
            List[OrderedDict[int, Tuple[datetime, datetime]]]: List of timetables
            for each vehicle (each dict is one vehicle).
        """
        timetables: List[TypingOrderedDict[int, Tuple[datetime, datetime]]] = []
        n = len(self.stops)

        for period_start, period_end, frequency in freq_period:
            current_departure = period_start

            while current_departure <= period_end:
                # Forward direction: 0 -> 1 -> ... -> n-1
                forward_indices = list(range(n))
                timetables.append(
                    self.__build_timetable_for_direction(current_departure, forward_indices)
                )

                # Reverse direction: n-1 -> ... -> 0 (if enabled)
                if self.bidirectional and n > 1:
                    backward_indices = list(reversed(range(n)))
                    timetables.append(
                        self.__build_timetable_for_direction(current_departure, backward_indices)
                    )

                current_departure += frequency

        return timetables

    # -------------------------------------------------------------------------
    # Routing and fares
    # -------------------------------------------------------------------------
    def get_next_departure(
        self,
        current_time: datetime,
        start_index: int,
        end_index: int,
    ) -> "FixedRouteVehicle":
        """
        Get the next available vehicle that departs from start_index after current_time
        and reaches end_index after that departure.

        Args:
            current_time: Time after which we search for a departure.
            start_index: Index of the starting stop.
            end_index: Index of the destination stop.

        Returns:
            FixedRouteVehicle: The selected vehicle.

        Raises:
            ValueError: If no suitable vehicle is found.
        """
        best_vehicle = None
        best_departure = None

        for vehicle in self.vehicles:
            timetable = vehicle.timetable

            if start_index not in timetable or end_index not in timetable:
                continue

            _, depart_start = timetable[start_index]
            arrival_end, _ = timetable[end_index]

            # Vehicle must depart after current_time
            if depart_start < current_time:
                continue

            # Vehicle must arrive at end after it departs from start
            if arrival_end <= depart_start:
                continue

            if best_departure is None or depart_start < best_departure:
                best_vehicle = vehicle
                best_departure = depart_start

        if best_vehicle is None:
            raise ValueError(
                f"No available departures from stop {start_index} to {end_index} "
                f"after {current_time}"
            )

        return best_vehicle

    def get_fare(self, start_hex, end_hex, time=None) -> float:
        # Simple flat fare for now
        def _read_fixedroute_base_fare_from_config():
            # Placeholder for reading from config
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../data/config.json"
            )
            with open(config_path, "r") as f:
                config = json.load(f)
            base = config.get("fixedroute_base_fare", 2.4)  # Default to 2.4 if not specified
            return base

        return _read_fixedroute_base_fare_from_config()

    def get_route(
        self,
        unit: float,
        start_time: datetime,
        start_hex: Hex,
        end_hex: Hex,
    ) -> Tuple[Wait, Ride]:
        """
        Get a (Wait, Ride) tuple representing the trip from start_hex to end_hex.

        Args:
            unit: Number of units (e.g., passengers) to be transported.
            start_time: When the trip planning starts.
            start_hex: Starting hexagon.
            end_hex: Destination hexagon.

        Returns:
            Tuple[Wait, Ride]: A tuple containing the Wait action and Ride action.

        Raises:
            ValueError: If start_hex or end_hex are not part of the service stops.
        """
        if start_hex not in self.stop_hex_lookup or end_hex not in self.stop_hex_lookup:
            raise ValueError("Start or end hex not in stops")

        start_index = self.stop_hex_lookup[start_hex]
        end_index = self.stop_hex_lookup[end_hex]

        # Pick vehicle that actually travels from start to end in the right time order
        vehicle = self.get_next_departure(start_time, start_index, end_index)

        _, next_departure = vehicle.timetable[start_index]
        arrival_time, _ = vehicle.timetable[end_index]

        if arrival_time <= next_departure:
            raise RuntimeError("Selected vehicle does not travel from start to end in time order")

        wait_action = Wait(start_time, next_departure, start_hex, unit)
        ride_action = Ride(
            next_departure,
            arrival_time,
            start_hex,
            end_hex,
            unit,
            service=self,
            vehicle=vehicle,  # Store vehicle reference
        )

        return wait_action, ride_action


class FixedRouteVehicle:
    """
    Represents a vehicle operating on a fixed-route service.

    Attributes:
        service (FixedRouteService): The service this vehicle operates on.
        stopping_time (timedelta): Time spent at each stop.
        current_load (float): Current number of units on board.
        capacity (float): Maximum capacity of the vehicle.
        timetable (OrderedDict[int, Tuple[datetime, datetime]]): Mapping from stop
            index to (arrival_time, departure_time).
    """

    def __init__(
        self,
        service: FixedRouteService,
        timetable: TypingOrderedDict[int, Tuple[datetime, datetime]],
    ):
        self.service = service
        self.stopping_time = service.stopping_time
        self.current_load: float = 0.0
        self.capacity: float = service.capacity
        # OrderedDict mapping stop index to (arrival_time, departure_time)
        self.timetable: TypingOrderedDict[int, Tuple[datetime, datetime]] = timetable

        # Optional: sanity-check the timetable
        self.__verify_timetable()

    def __verify_timetable(self):
        """
        Verify that the timetable is consistent with the service's travel times.

        Raises:
            ValueError: If the timetable is inconsistent.
        """
        items = list(self.timetable.items())
        for (idx_curr, (_, depart_curr)), (idx_next, (arr_next, _)) in zip(items, items[1:]):
            start_stop = self.service.stops[idx_curr]
            end_stop = self.service.stops[idx_next]

            distance = self.service.network.get_distance(start_stop, end_stop)
            expected_travel = self.service.travel_time * distance

            # Time between departing current stop and arriving at next stop
            actual_travel = arr_next - depart_curr

            if actual_travel < expected_travel:
                raise ValueError(
                    f"Inconsistent timetable between stops {start_stop} and {end_stop}: "
                    f"actual travel {actual_travel}, expected at least {expected_travel}"
                )

    def load_passengers(self, unit: float):
        if self.current_load + unit > self.capacity:
            raise ValueError("Exceeding vehicle capacity")
        self.current_load += unit

    def unload_passengers(self, unit: float):
        if self.current_load - unit < 0:
            raise ValueError("Cannot unload more passengers than currently loaded")
        self.current_load -= unit
