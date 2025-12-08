"""
Demand class representing a specific travel demand or request.
"""

from datetime import datetime

from .hex import Hex


class Demand:
    """
    Represents a specific travel demand or request.

    Attributes:
        hour (int): The hour at which the demand occurs.
        start_hex (Hex): The starting hexagonal cell for the demand.
        end_hex (Hex): The destination hexagonal cell for the demand.
        unit (float): A numerical value associated with the demand (e.g., number of passengers, weight).
    """

    def __init__(self, time: datetime, start_hex: Hex, end_hex: Hex, unit: int):
        """
        Initialize a Demand object.

        Args:
            time (datetime): The time at which the demand occurs.
            start_hex (Hex): The starting hexagonal cell for the demand.
            end_hex (Hex): The destination hexagonal cell for the demand.
            unit (int): A numerical value associated with the demand.
        """
        self.time = time
        self.start_hex = start_hex
        self.end_hex = end_hex
        self.unit = unit

    def __repr__(self):
        return f"Demand(time={self.time}, start_hex={self.start_hex}, end_hex={self.end_hex}, unit={self.unit})"

    def __eq__(self, other):
        if not isinstance(other, Demand):
            return False
        return (
            self.hour == other.hour
            and self.start_hex == other.start_hex
            and self.end_hex == other.end_hex
        )
