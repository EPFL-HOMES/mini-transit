from datetime import datetime

from ..primitives.hex import Hex
from .action import Action


class Wait(Action):
    def __init__(self, start_time: datetime, end_time: datetime, location: Hex, unit: int):
        super().__init__(start_time, end_time, unit=unit)
        self.location = location
        self.fare = 0.0  # Waiting has no fare
