from pydantic import BaseModel
from typing import List, Tuple, OrderedDict
from datetime import datetime, timedelta

class HexModel(BaseModel):
    hex_id: str

class FixedRouteServiceModel(BaseModel):
    name: str
    stops: List[HexModel]
    capacity: float
    stopping_time: timedelta
    travel_time: timedelta
    vehicles: List[OrderedDict[int, Tuple[datetime, datetime]]]  # List of dicts mapping stop index to (arrival_time, departure_time)



