"""
FixedRouteService class representing a fixed route public transportation service.
"""

from datetime import datetime, timedelta, time
from typing import List, Dict
import json


class Vehicle:
    """
    Represents a single vehicle operating on a fixed route.
    
    Attributes:
        timetable (List[datetime]): List of arrival times at each stop.
        current_capacity (int): Current number of passengers on board.
        max_capacity (int): Maximum capacity of this vehicle.
        active (bool): Whether this vehicle is currently operating.
    """
    
    def __init__(self, timetable: List[datetime], max_capacity: int):
        """
        Initialize a Vehicle object.
        
        Args:
            timetable (List[datetime]): List of arrival times at each stop.
            max_capacity (int): Maximum capacity of this vehicle.
        """
        self.timetable = timetable
        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.active = False
    
    def is_at_stop(self, current_time: datetime, stop_index: int) -> bool:
        """
        Check if this vehicle is at a specific stop at the given time.
        
        Since we operate with HH:MM format (1 minute precision), we check for exact match.
        
        Args:
            current_time (datetime): Current simulation time.
            stop_index (int): Index of the stop to check.
            
        Returns:
            bool: True if vehicle is at the stop at the exact time, False otherwise.
        """
        if stop_index >= len(self.timetable):
            return False
        
        arrival_time = self.timetable[stop_index]
        
        # Normalize both times to minute precision (ignore seconds and microseconds)
        current_normalized = current_time.replace(second=0, microsecond=0)
        arrival_normalized = arrival_time.replace(second=0, microsecond=0)
        
        # Vehicle is at stop if current time exactly matches the timetable time
        return current_normalized == arrival_normalized
    
    def update_active_status(self, current_time: datetime):
        """
        Update the active status based on current time.
        
        Args:
            current_time (datetime): Current simulation time.
        """
        if not self.timetable:
            self.active = False
            return
        
        # Normalize current time to minute precision
        current_normalized = current_time.replace(second=0, microsecond=0)
        
        first_arrival = self.timetable[0].replace(second=0, microsecond=0)
        last_arrival = self.timetable[-1].replace(second=0, microsecond=0)
        
        # Vehicle is active if current time is between first and last stop (inclusive)
        self.active = first_arrival <= current_normalized <= last_arrival
    
    def get_current_stop_index(self, current_time: datetime) -> int:
        """
        Get the index of the stop where this vehicle currently is.
        
        Args:
            current_time (datetime): Current simulation time.
            
        Returns:
            int: Index of current stop, or -1 if not at any stop.
        """
        for i, arrival_time in enumerate(self.timetable):
            if self.is_at_stop(current_time, i):
                return i
        return -1
    
    def __repr__(self):
        first_time = self.timetable[0].strftime("%H:%M") if self.timetable else "N/A"
        return f"Vehicle(start={first_time}, capacity={self.current_capacity}/{self.max_capacity}, active={self.active})"


class FixedRouteService:
    """
    Represents a fixed route public transportation service (bus, metro, etc.).
    
    Attributes:
        name (str): Name of the service (e.g., "Bus 1", "Metro Line 1").
        stops (List[int]): List of hex IDs representing stops in order.
        frequency (int): Frequency in minutes (how often vehicles appear at stops).
        capacity (int): Maximum capacity per vehicle.
        vehicles (List[Vehicle]): List of all vehicles operating on this route.
        route_start_time (datetime): When the first vehicle starts (default 0:00).
    """
    
    def __init__(self, name: str, stops: List[int], frequency: int, capacity: int, 
                 route_start_time: datetime = None, time_per_stop: int = 2):
        """
        Initialize a FixedRouteService object.
        
        Args:
            name (str): Name of the service.
            stops (List[int]): List of hex IDs for stops.
            frequency (int): Frequency in minutes.
            capacity (int): Maximum capacity per vehicle.
            route_start_time (datetime, optional): When route starts. Defaults to 0:00 today.
            time_per_stop (int): Minutes to travel between consecutive stops. Default 2 minutes.
        """
        self.name = name
        self.stops = stops
        self.frequency = frequency
        self.capacity = capacity
        self.time_per_stop = time_per_stop
        
        # Default route start time to 0:00 (midnight)
        if route_start_time is None:
            self.route_start_time = datetime.combine(datetime.today(), time(0, 0))
        else:
            self.route_start_time = route_start_time
        
        self.vehicles = []
        self._spawn_vehicles()
    
    def _spawn_vehicles(self):
        """
        Spawn all vehicles for the day based on frequency.
        Vehicles are spawned so that one arrives at each stop every 'frequency' minutes.
        """
        # Calculate how many vehicles we need for a full day (24 hours = 1440 minutes)
        minutes_per_day = 24 * 60
        num_vehicles = (minutes_per_day // self.frequency) + 1  # +1 to ensure coverage
        
        # Spawn vehicles starting at route_start_time, then every 'frequency' minutes
        for i in range(num_vehicles):
            vehicle_start_time = self.route_start_time + timedelta(minutes=i * self.frequency)
            
            # Create timetable for this vehicle
            timetable = []
            current_time = vehicle_start_time
            
            for stop_index in range(len(self.stops)):
                timetable.append(current_time)
                # Move to next stop (add time_per_stop minutes)
                if stop_index < len(self.stops) - 1:
                    current_time = current_time + timedelta(minutes=self.time_per_stop)
            
            # Create vehicle with this timetable
            vehicle = Vehicle(timetable, self.capacity)
            self.vehicles.append(vehicle)
    
    def get_vehicles_at_stop(self, stop_hex_id: int, current_time: datetime) -> List[Vehicle]:
        """
        Get all vehicles currently at a specific stop.
        
        Args:
            stop_hex_id (int): Hex ID of the stop.
            current_time (datetime): Current simulation time.
            
        Returns:
            List[Vehicle]: List of vehicles at the stop.
        """
        if stop_hex_id not in self.stops:
            return []
        
        stop_index = self.stops.index(stop_hex_id)
        vehicles_at_stop = []
        
        for vehicle in self.vehicles:
            vehicle.update_active_status(current_time)
            if vehicle.is_at_stop(current_time, stop_index):
                vehicles_at_stop.append(vehicle)
        
        return vehicles_at_stop
    
    def get_next_vehicle_arrival(self, stop_hex_id: int, current_time: datetime) -> datetime:
        """
        Get the arrival time of the next vehicle at a specific stop.
        
        Args:
            stop_hex_id (int): Hex ID of the stop.
            current_time (datetime): Current simulation time.
            
        Returns:
            datetime: Arrival time of next vehicle, or None if no more vehicles.
        """
        if stop_hex_id not in self.stops:
            return None
        
        # Normalize current time to minute precision
        current_normalized = current_time.replace(second=0, microsecond=0)
        
        stop_index = self.stops.index(stop_hex_id)
        next_arrival = None
        
        for vehicle in self.vehicles:
            if stop_index < len(vehicle.timetable):
                arrival_time = vehicle.timetable[stop_index].replace(second=0, microsecond=0)
                # Only consider vehicles that arrive AFTER current time
                if arrival_time > current_normalized:
                    if next_arrival is None or arrival_time < next_arrival:
                        next_arrival = vehicle.timetable[stop_index]  # Return original datetime
        
        return next_arrival
    
    @classmethod
    def from_json(cls, service_data: Dict, route_start_time: datetime = None, 
                  time_per_stop: int = None) -> 'FixedRouteService':
        """
        Create a FixedRouteService from JSON data.
        
        Args:
            service_data (Dict): Dictionary with keys: name, stops, frequency, capacity.
            route_start_time (datetime, optional): When route starts. Defaults to 0:00.
            time_per_stop (int, optional): Minutes between stops. If None, uses frequency.
            
        Returns:
            FixedRouteService: New FixedRouteService instance.
        """
        # If time_per_stop not specified, use frequency (travel time between stops equals frequency)
        if time_per_stop is None:
            time_per_stop = service_data['frequency']
        
        return cls(
            name=service_data['name'],
            stops=service_data['stops'],
            frequency=service_data['frequency'],
            capacity=service_data['capacity'],
            route_start_time=route_start_time,
            time_per_stop=time_per_stop
        )
    
    @classmethod
    def load_from_file(cls, file_path: str, route_start_time: datetime = None,
                      time_per_stop: int = None) -> List['FixedRouteService']:
        """
        Load multiple FixedRouteService objects from a JSON file.
        
        Args:
            file_path (str): Path to JSON file.
            route_start_time (datetime, optional): When routes start. Defaults to 0:00.
            time_per_stop (int, optional): Minutes between stops. If None, uses frequency for each service.
            
        Returns:
            List[FixedRouteService]: List of FixedRouteService instances.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        services = []
        for service_data in data.get('services', []):
            service = cls.from_json(service_data, route_start_time, time_per_stop)
            services.append(service)
        
        return services
    
    def __repr__(self):
        return f"FixedRouteService(name={self.name}, stops={len(self.stops)}, frequency={self.frequency}min, vehicles={len(self.vehicles)})"

