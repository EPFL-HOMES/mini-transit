"""
OnDemandRide class representing an on-demand ride action (e.g., bike share).
Unlike regular Ride actions, on-demand rides don't share capacity - if a vehicle
is taken, it's exclusively used by that unit until the ride ends.
"""

from datetime import datetime

from ..primitives.hex import Hex
from ..services import Service
from .action import Action


class OnDemandRide(Action):
    """
    Represents an on-demand ride action between two hexagons.

    For on-demand services (like bike share), each vehicle is taken exclusively
    by one unit. Unlike fixed-route services where multiple units can share
    a vehicle, on-demand vehicles are checked for availability at pickup locations
    but don't track capacity during the ride.
    Attributes:
        start_hex (Hex): Starting hexagon.
        end_hex (Hex): Destination hexagon.
        unit (int): Number of units (typically 1 for on-demand).
        service (Service): The on-demand service being used.
        vehicle: The vehicle this ride is associated with.
    """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        start_hex: Hex,
        end_hex: Hex,
        unit: int,
        service: Service,
        vehicle=None,
    ):
        """
        Initialize an OnDemandRide action.
        Args:
            start_time (datetime): When the ride starts.
            end_time (datetime): When the ride ends.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            unit (int): Number of units being transported (typically 1 for bikes).
            service (Service): The on-demand service.
            vehicle: The vehicle this ride is associated with.
        """
        super().__init__(start_time, end_time, unit=unit)
        self.start_hex = start_hex
        self.end_hex = end_hex
        self.service = service
        self.vehicle = vehicle  # Store vehicle reference
        self.fare = self.service.get_fare(start_hex, end_hex, start_time)
