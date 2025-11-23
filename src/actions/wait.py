"""
Wait class representing waiting action at a stop for a fixed route service vehicle.
"""

from datetime import datetime
from .action import Action

class Wait(Action):
    """
    Represents a waiting action at a stop for a fixed route service vehicle.
    
    Attributes:
        unit (int): Amount of units waiting.
        position (int): Hex ID where the unit is waiting.
        start_time (datetime): When the wait starts.
        end_time (datetime): When the wait ends (vehicle arrives).
    """
    
    def __init__(self, start_time: datetime, position: int, unit: int, end_time: datetime = None):
        """
        Initialize a Wait action.
        
        Args:
            start_time (datetime): When the wait starts.
            position (int): Hex ID where the unit is waiting.
            unit (int): Amount of units waiting.
            end_time (datetime, optional): When the wait ends (vehicle arrives).
                                        If None, must be set later.
        """
        super().__init__(start_time)
        self.position = position
        self.unit = unit
        self.end_time = end_time
    
    def set_end_time(self, end_time: datetime):
        """
        Set the end time when the vehicle arrives.
        
        Args:
            end_time (datetime): When the wait ends.
        """
        self.end_time = end_time
    
    def __repr__(self):
        end_time_str = self.end_time.strftime("%H:%M") if self.end_time else "None"
        return f"Wait(start_time={self.start_time.strftime('%H:%M')}, end_time={end_time_str}, position={self.position}, unit={self.unit})"

