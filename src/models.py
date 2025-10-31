from pydantic import BaseModel
from typing import List, Tuple, TypingOrderedDict, Dict
from datetime import datetime, timedelta
from src.hex import Hex

class FixedRouteServiceModel(BaseModel):
    name: str
    stops: List[Hex]
    capacity: float
    stopping_time: timedelta
    travel_time: timedelta
    vehicles: List[TypingOrderedDict[int, Tuple[datetime, datetime]]]  # List of dicts mapping stop index to (arrival_time, departure_time)

