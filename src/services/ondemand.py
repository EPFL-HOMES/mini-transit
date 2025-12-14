import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from typing import OrderedDict as TypingOrderedDict
from typing import Tuple

from src.actions import Ride, Wait
from src.hex import Hex
from src.route import Route
from src.services import Service


class OnDemandRouteService(Service):
    """
    Represents an on-demand transportation service - in this case simulating bike-sharing.
    Currently the assumption is that the service represents all vehicles under the same service.
    Attributes:
        name (str): Name of the service.
        stops (List[Hex]): List of Hex objects representing the stops.
        stop_hex_lookup (Dict[Hex, int]): Mapping from Hex to its index in stops.
        vehicles (List[FixedRouteVehicle]): List of vehicles operating on this route.
        capacity (float): Maximum capacity of each vehicle.
        stopping_time (timedelta): Time spent at each stop.
        travel_time (timedelta): Time taken to travel between hexes.
    """

    def __init__(
        self, name, vehicles: List["OnDemandVehicle"], capacity: float, network
    ):  # Network is required here unlike FixedRouteService
        super().__init__(name)
        self.vehicles = vehicles  # List of OnDemandVehicle instances
        self.capacity = capacity
        self.network = network

    def get_fare(self, start_hex, end_hex, time=None) -> float:
        """
        Get fare for a trip from start_hex to end_hex.

        Args:
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            time (datetime, optional): Time of the trip. Defaults to None.
        Returns:
            float: Fare amount.
        """
        base_fare = 3.0  # Base fare for the first 30 minutes for on-demand service
        time_rate_per_minute = 0.1  # Rate per minute of travel
        drive_time = self.compute_drive_time(start_hex, end_hex)
        total_minutes = drive_time.total_seconds() / 60
        if total_minutes <= 30:
            total_fare = base_fare
        else:
            total_fare = base_fare + ((total_minutes - 30) * time_rate_per_minute)

        return total_fare

    def _load_on_demand_speed_from_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/config.json"
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("on_demand_speed", 35.0)  # Default to 35.0 if not specified

    def compute_drive_time(self, start_hex: Hex, end_hex: Hex) -> timedelta:
        """
        Compute the drive time between two hexes based on on-demand speed.

        Args:
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.

        Returns:
            timedelta: Estimated drive time.
        """
        distance = self.network.get_distance(start_hex, end_hex)
        on_demand_speed = self._load_on_demand_speed_from_config()  # in hexes per hour
        hours = distance / on_demand_speed
        return timedelta(hours=hours)

    def get_route(self, unit, start_time: datetime, start_hex: Hex, end_hex: Hex) -> Route:
        """
        Get a Route object representing the trip from start_hex to end_hex.
        Args:
            unit (float): Number of units to be transported.
            start_time (datetime): When the trip starts.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
        Returns:
            Ride: A Ride action representing the trip.
        """
        # For On-Demand, we assume immediate availability for simplicity
        drive_time = self.compute_drive_time(start_hex, end_hex)
        arrival_time = start_time + drive_time

        ride_action = Ride(start_time, arrival_time, start_hex, end_hex, unit, service=self)

        return ride_action


class OnDemandRouteServiceDocked(OnDemandRouteService):
    """
    Represents an on-demand transportation service with docking stations.
    Inherits from OnDemandRouteService.
    """

    def __init__(
        self,
        name,
        vehicles: List["OnDemandVehicle"],
        capacity: float,
        network,
        docking_stations: List["DockingStation"],
    ):
        super().__init__(name, vehicles, capacity, network)
        self.docking_stations = (
            docking_stations  # List of DockingStation objects representing docking stations
        )


class DockingStation:
    """
    Represents a docking station for docked on-demand vehicles.

    Attributes:
        station_id (str): Unique identifier for the docking station.
        location (Hex): Location of the docking station.
        capacity (int): Maximum number of vehicles the station can hold.
    """

    def __init__(self, station_id: str, location: Hex, capacity: int):
        """
        Initialize a DockingStation.

        Args:
            station_id (str): Unique identifier for the docking station.
            location (Hex): Location of the docking station.
            capacity (int): Maximum number of vehicles the station can hold.
        """
        self.station_id = station_id
        self.location = location
        self.capacity = capacity
        self.current_vehicles: List[OnDemandVehicle] = (
            []
        )  # Vehicles currently docked at the station

    def check_availability(self) -> bool:
        """
        Check if there is available space at the docking station.

        Returns:
            bool: True if there is space, False otherwise.
        """
        return len(self.current_vehicles) < self.capacity

    def take_vehicle(self) -> "OnDemandVehicle":
        """
        Take a vehicle from the docking station if available.

        Returns:
            OnDemandVehicle: The vehicle taken from the station.

        Raises:
            Exception: If no vehicles are available.
        """
        if self.current_vehicles:
            return self.current_vehicles.pop()
        else:
            raise Exception("No vehicles available at this docking station.")

    def dock_vehicle(self, vehicle: "OnDemandVehicle"):
        """
        Dock a vehicle at the docking station if there is space.

        Args:
            vehicle (OnDemandVehicle): The vehicle to be docked.
        Raises:
            Exception: If the docking station is full.
        """
        if self.check_availability():
            self.current_vehicles.append(vehicle)
        else:
            raise Exception("Docking station is full.")


# note i'm adding here first: vehicles are only "docked" to a docking station after the ride is over and it has been made available again


class OnDemandRouteServiceDockless(OnDemandRouteService):
    """
    Represents an on-demand transportation service that simply doesn't have docking stations.
    Inherits from OnDemandRouteService.
    """

    def __init__(self, name, vehicles: List["OnDemandVehicle"], capacity: float, network):
        super().__init__(name, vehicles, capacity, network)
        # No docking stations for dockless service


class OnDemandVehicle:
    """
    Represents a vehicle in the on-demand service.

    Attributes:
        vehicle_id (str): Unique identifier for the vehicle.
        current_location (Hex): Current location of the vehicle.
        capacity (float): Maximum capacity of the vehicle.
    """

    def __init__(self, vehicle_id: str, current_location: Hex, capacity: int = 1):
        """
        Initialize an OnDemandVehicle.

        Args:
            vehicle_id (str): Unique identifier for the vehicle.
            current_location (Hex): Current location of the vehicle.
            capacity (float): Maximum capacity of the vehicle.
        """
        self.vehicle_id = vehicle_id
        self.current_location = current_location
        self.capacity = capacity
        self.available_time = datetime.min  # Initially available

    def is_available(self, request_time: datetime) -> bool:
        """
        Check if the vehicle is available at the requested time.

        Args:
            request_time (datetime): The time to check availability for.
        Returns:
            bool: True if the vehicle is available, False otherwise.
        """
        return request_time >= self.available_time
