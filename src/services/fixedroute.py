from datetime import datetime, timedelta

from src.services import Service
from src.route import Route
from src.actions import Wait, Ride
from src.hex import Hex
from src.network import Network
from typing import List, Tuple, Dict, OrderedDict as TypingOrderedDict
from collections import OrderedDict



class FixedRouteService(Service):

    def __init__(self, name, 
                 stops: List[Hex],
                 capacity: float, 
                 stopping_time: timedelta, 
                 travel_time: timedelta, 
                 vehicles: List[TypingOrderedDict[int, Tuple[datetime,datetime]]],):   
        super().__init__(name)
        self.stops = stops
        self.stop_hex_lookup = {stop: index for index, stop in enumerate(stops)}
        self.vehicles = [FixedRouteVehicle(self, timetable=t) for t in vehicles]  # List of OrderedDicts mapping stop index to (arrival_time, departure_time)
        self.capacity = capacity        
        self.stopping_time = stopping_time
        self.travel_time = travel_time


    def __get_next_departure(self, current_time: datetime, stop_index: int):
        for vehicle in self.vehicles:
            if stop_index in vehicle.timetable:
                _, departure_time = vehicle.timetable[stop_index]
                if departure_time >= current_time:
                    return vehicle
        raise ValueError("No available departures from this stop after the given time")
        
    
    def get_fare(self, start_hex, end_hex, time = None) -> float:
        return 2.40  # Fixed route service
    
    def get_route(self, unit, start_time: datetime, start_hex, end_hex) -> Route:
        if start_hex not in self.stop_hex_lookup or end_hex not in self.stop_hex_lookup:
            raise ValueError("Start or end hex not in stops")
        
        start_index = self.stop_hex_lookup[start_hex]
        end_index = self.stop_hex_lookup[end_hex]


        vehicle = self.__get_next_departure(start_time, start_index)
        _, next_departure = vehicle.timetable[start_index]
        arrival_time, _ = vehicle.timetable[end_index]

        wait_action = Wait(start_time, next_departure, start_hex, )
        ride_action = Ride(
            next_departure,
            arrival_time,
            start_hex,
            end_hex,
            unit,
        )

        return Route([wait_action, ride_action])


class FixedRouteVehicle:
    def __init__(self, service: FixedRouteService, timetable: TypingOrderedDict[int, Tuple[datetime, datetime]]):
        self.service = service
        self.stopping_time = service.stopping_time
        self.current_load = 0
        self.capacity = service.capacity
        self.timetable = timetable  # OrderedDict mapping stop index to (arrival_time, departure_time)
    
    def __verify_timetable(self):
        pass

    
    def load_passengers(self, unit: float):
        if self.current_load + unit > self.capacity:
            raise ValueError("Exceeding vehicle capacity")
        self.current_load += unit
    
    def unload_passengers(self, unit: float):
        if self.current_load - unit < 0:
            raise ValueError("Cannot unload more passengers than currently loaded")
        self.current_load -= unit