"""
TakeService class representing taking service action in a route.
"""

from datetime import datetime
from .action import Action
from .hex import Hex
from .service import Service

class TakeService(Action):
    """
    Represents taking a service action between two hexagons.
    
    Attributes:
        service_id (str): Identifier for the service being taken.
        start_hex (Hex): Starting hexagon.
        end_hex (Hex): Destination hexagon.
    """
    
    def __init__(self, start_time: datetime, service: Service, start_hex: Hex, end_hex: Hex):
        """
        Initialize a TakeService action.
        
        Args:
            start_time (datetime): When the service starts.
            service: instance of Service being taken.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
        """
        super().__init__(start_time)
        self.service = service
        self.start_hex = start_hex
        self.end_hex = end_hex