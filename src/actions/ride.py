from datetime import datetime, timedelta
from src.action import Action
from src.hex import Hex
from src.services import Service
import json
import os

class Ride(Action):
    """
    Represents a ride action between two hexagons.
    
    Attributes:
        start_hex (Hex): Starting hexagon.
        end_hex (Hex): Destination hexagon.
        units (float): Number of units being transported.
    """
    
    def __init__(self, start_time: datetime, end_time: datetime, start_hex: Hex, end_hex: Hex, units: float, service = Service):
        """
        Initialize a Ride action.
        
        Args:
            start_time (datetime): When the ride starts.
            end_time (datetime): When the ride ends.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            units (float): Number of units being transported.
        """
        super().__init__(start_time, end_time, units)
        self.end_time = end_time
        self.start_hex = start_hex
        self.end_hex = end_hex
        self.units = units
        self.service = service
