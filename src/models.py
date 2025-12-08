from datetime import datetime, timedelta
from typing import List, OrderedDict, Tuple

import networkx as nx
from pydantic import BaseModel, ConfigDict

from src.hex import Hex


class HexModel(BaseModel):
    hex_id: str


class FixedRouteServiceModel(BaseModel):
    name: str
    stops: List[Hex]
    capacity: float
    stopping_time: timedelta
    travel_time: timedelta
    vehicles: List[
        OrderedDict[int, Tuple[datetime, datetime]]
    ]  # List of dicts mapping stop index to (arrival_time, departure_time)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DemandModel(BaseModel):
    time: datetime
    start_hex: Hex
    end_hex: Hex
    unit: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FixedRouteServiceInput(BaseModel):
    name: str
    stops: List[Hex]
    capacity: int
    stopping_time: timedelta
    travel_time: timedelta
    freq_period: List[Tuple[datetime, datetime, timedelta]]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class NetworkModel(BaseModel):
    graph: nx.Graph
    services: List[FixedRouteServiceModel]
    routes_taken: list

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DemandInput(BaseModel):
    hour: int
    start_hex: Hex
    end_hex: Hex
    unit: int
    model_config = ConfigDict(arbitrary_types_allowed=True)
