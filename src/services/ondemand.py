from datetime import datetime, timedelta

from src.services import Service
from src.route import Route
from src.actions import Wait, Ride
from src.hex import Hex
from src.network import Network
from typing import List, Tuple, Dict, OrderedDict as TypingOrderedDict
import json
import os



class OnDemandRouteService(Service):
    '''
    Represents a fixed-route transportation service.
    Attributes:
        name (str): Name of the service.
        stops (List[Hex]): List of Hex objects representing the stops.
        stop_hex_lookup (Dict[Hex, int]): Mapping from Hex to its index in stops.
        vehicles (List[FixedRouteVehicle]): List of vehicles operating on this route.
        capacity (float): Maximum capacity of each vehicle.
        stopping_time (timedelta): Time spent at each stop.
        travel_time (timedelta): Time taken to travel between hexes.
    '''

    def __init__(self, name, 
                 location: Hex,
                 capacity: float, 
                 network: Network):   # Network is required here unlike FixedRouteService
        super().__init__(name)
        self.location = location
        self.capacity = capacity        
        self.network = network
        
    
    def get_fare(self, start_hex, end_hex, time = None) -> float:
        '''
        Get fare for a trip from start_hex to end_hex.
        
        Args:
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            time (datetime, optional): Time of the trip. Defaults to None.
        Returns:
            float: Fare amount.
        '''
        base_fare = 6.2  # Base fare for on-demand service
        distance = self.network.get_distance(start_hex, end_hex)
        per_hex_rate = 1.0  # Rate per hexagon traveled
        total_fare = base_fare + (distance * per_hex_rate)
        return total_fare
    
    def _load_on_demand_speed_from_config(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('on_demand_speed', 35.0)  # Default to 35.0 if not specified
    
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
        '''
        Get a Route object representing the trip from start_hex to end_hex.
        Args:
            unit (float): Number of units to be transported.
            start_time (datetime): When the trip starts.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
        Returns:
            Ride: A Ride action representing the trip.
        '''
        # For On-Demand, we assume immediate availability for simplicity
        drive_time = self.compute_drive_time(start_hex, end_hex)
        arrival_time = start_time + drive_time

        ride_action = Ride(
            start_time,
            arrival_time,
            start_hex,
            end_hex,
            unit,
            service=self
        )

        return ride_action