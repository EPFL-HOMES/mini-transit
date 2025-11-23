"""
Ride class representing riding action on a fixed route service vehicle.
"""

from datetime import datetime
from .action import Action

class Ride(Action):
    """
    Represents a riding action on a fixed route service vehicle.
    
    Attributes:
        name (str): Name of the fixed route service (e.g., "Bus 1").
        vehicle_index (int): Index in the list of vehicles for this fixed route service.
                           Allows quick lookup to update vehicle capacity.
        start_time (datetime): When the unit boards the vehicle (should match vehicle's timetable).
        end_time (datetime): When the unit finishes travel (should match vehicle's timetable).
        start_hex (int): Hex ID of start stop (should match a stop in the fixedRouteService).
        end_hex (int): Hex ID of end stop (should match a stop in the fixedRouteService).
    """
    
    def __init__(self, start_time: datetime, end_time: datetime, name: str, 
                 vehicle_index: int, start_hex: int, end_hex: int):
        """
        Initialize a Ride action.
        
        Args:
            start_time (datetime): When the unit boards the vehicle.
            end_time (datetime): When the unit finishes travel.
            name (str): Name of the fixed route service.
            vehicle_index (int): Index in the vehicles list for quick lookup.
            start_hex (int): Hex ID of start stop.
            end_hex (int): Hex ID of end stop.
        """
        super().__init__(start_time)
        self.name = name
        self.vehicle_index = vehicle_index
        self.end_time = end_time
        self.start_hex = start_hex
        self.end_hex = end_hex
    
    def __repr__(self):
        start_time_str = self.start_time.strftime("%H:%M") if self.start_time else "None"
        end_time_str = self.end_time.strftime("%H:%M") if self.end_time else "None"
        return f"Ride(name={self.name}, vehicle_index={self.vehicle_index}, start_time={start_time_str}, end_time={end_time_str}, start_hex={self.start_hex}, end_hex={self.end_hex})"

