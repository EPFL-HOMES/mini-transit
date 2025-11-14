"""
Wait class representing waiting action in a route.
"""

from datetime import datetime, timedelta

from .action import Action
from .hex import Hex


class Wait(Action):
    """
    Represents a waiting action at a hexagon.

    Attributes:
        location_hex (Hex): Hexagon where the wait occurs.
        duration (timedelta): Duration of the wait.
    """

    def __init__(self, start_time: datetime, location_hex, duration: timedelta):
        """
        Initialize a Wait action.

        Args:
            start_time (datetime): When the wait starts.
            location_hex (Hex): Hexagon where the wait occurs.
            duration (timedelta): Duration of the wait.
        """
        super().__init__(start_time)
        self.location_hex = location_hex
        self.duration = duration
        self.end_time = self.start_time + self.duration
