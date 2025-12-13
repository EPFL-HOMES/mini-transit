import json
import os
from datetime import datetime, timedelta

from src.actions.action import Action
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

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        start_hex: Hex,
        end_hex: Hex,
        unit: int,
        service: Service,
        vehicle=None, # for OnDemand Vehicles in particular
    ):
        """
        Initialize a Ride action.

        Args:
            start_time (datetime): When the ride starts.
            end_time (datetime): When the ride ends.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            units (float): Number of units being transported.
        """
        super().__init__(start_time, end_time, unit=unit)
        self.start_hex = start_hex
        self.end_hex = end_hex
        self.service = service
        self.vehicle = vehicle
        self.fare = self.service.get_fare(start_hex, end_hex, start_time)
