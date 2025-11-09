from datetime import datetime, timedelta

from src.services import Service
from src.route import Route
from src.actions import Wait, Ride
from src.hex import Hex
from src.network import Network
from typing import List, Tuple, Dict, OrderedDict as TypingOrderedDict



class FixedRouteService(Service):
    '''
    Represents a fixed-route transportation service.
    Attributes:
        name (str): Name of the service.
        stops (List[Hex]): List of Hex objects representing the stops.
        stop_hex_lookup (Dict[Hex, int]): Mapping from Hex to its index in stops.
        vehicles (List[FixedRouteVehicle]): List of vehicles operating on this route.
        capacity (float): Maximum capacity of each vehicle.
        stopping_time (timedelta): Time spent at each stop.
        travel_time (timedelta): Time taken to travel between hexes.
        network (Network): The transportation network the service operates on.
    '''

    def __init__(self, name, 
                 stops: List[Hex],
                 capacity: float, 
                 stopping_time: timedelta, 
                 travel_time: timedelta, 
                 vehicles: List[TypingOrderedDict[int, Tuple[datetime,datetime]]],
                 network: Network = None):   
        super().__init__(name)
        self.stops = stops
        self.stop_hex_lookup = {stop: index for index, stop in enumerate(stops)}
        self.vehicles = [FixedRouteVehicle(self, timetable=t) for t in vehicles]  # List of OrderedDicts mapping stop index to (arrival_time, departure_time)
        self.capacity = capacity        
        self.stopping_time = stopping_time
        self.travel_time = travel_time
        self.network = network


    def __get_next_departure(self, current_time: datetime, stop_index: int):
        '''
        Get the next available vehicle departing from the given stop after current_time.
        Args:
            current_time (datetime): The time after which to find the next departure.
            stop_index (int): Index of the stop in the route.
        Returns:
            FixedRouteVehicle: The next available vehicle.
        '''
        for vehicle in self.vehicles:
            if stop_index in vehicle.timetable:
                _, departure_time = vehicle.timetable[stop_index]
                if departure_time >= current_time:
                    return vehicle
        raise ValueError("No available departures from this stop after the given time")
        
    
    def get_fare(self, start_hex, end_hex, time = None) -> float:
        return 2.40  # Fixed route service
    
    def get_route(self, unit, start_time: datetime, start_hex: Hex, end_hex: Hex) -> Route:
        '''
        Get a Route object representing the trip from start_hex to end_hex.
        Args:
            unit (float): Number of units to be transported.
            start_time (datetime): When the trip starts.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
        Returns:
            Tuple[Wait, Ride]: A tuple containing the Wait action and Ride action.
        '''
        #TODO: I don't even know why I wrote TODO here, might figure it out later
        if start_hex not in self.stop_hex_lookup or end_hex not in self.stop_hex_lookup:
            raise ValueError("Start or end hex not in stops")
        
        start_index = self.stop_hex_lookup[start_hex]
        end_index = self.stop_hex_lookup[end_hex]

        # Get the next available vehicle departing from start_hex
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
            service=self
        )

        return wait_action, ride_action


class FixedRouteVehicle:
    '''
    Represents a vehicle operating on a fixed-route service.
    Attributes:
        service (FixedRouteService): The service this vehicle operates on.
        stopping_time (timedelta): Time spent at each stop.
        current_load (float): Current number of units on board.
        capacity (float): Maximum capacity of the vehicle.
        timetable (OrderedDict[int, Tuple[datetime, datetime]]): Mapping from stop index to (arrival_time, departure_time).
    '''
    def __init__(self, service: FixedRouteService, timetable: TypingOrderedDict[int, Tuple[datetime, datetime]]):
        self.service = service
        self.stopping_time = service.stopping_time
        self.current_load = 0
        self.capacity = service.capacity
        self.timetable = timetable  # OrderedDict mapping stop index to (arrival_time, departure_time)
    
    def __verify_timetable(self):
        '''
        Verify that the timetable is consistent with the service's travel and stopping times.
        Raises:
            ValueError: If the timetable is inconsistent.
        '''
        for i in range(len(self.timetable) - 1):
            _, departure_time = self.timetable[i]
            next_arrival_time, _ = self.timetable[i + 1]
            # Expected travel time between stops
            # This requires access to the network to calculate distance
            start_stop = self.service.stop_hex_lookup[i]
            end_stop = self.service.stop_hex_lookup[i + 1]
            distance = self.service.network.get_distance(start_stop, end_stop)
            expected_travel_time = self.service.travel_time * distance + self.stopping_time
            if next_arrival_time - departure_time < expected_travel_time:
                raise ValueError("Inconsistent timetable between stops {} and {}".format(start_stop, end_stop))

    
    def load_passengers(self, unit: float):
        if self.current_load + unit > self.capacity:
            raise ValueError("Exceeding vehicle capacity")
        self.current_load += unit
    
    def unload_passengers(self, unit: float):
        if self.current_load - unit < 0:
            raise ValueError("Cannot unload more passengers than currently loaded")
        self.current_load -= unit