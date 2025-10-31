from datetime import datetime
from src.action import Action
from src.hex import Hex

class Wait(Action):
    def __init__(self, start_time: datetime, end_time: datetime, location: Hex):
        super().__init__(start_time, end_time)
        self.location = location