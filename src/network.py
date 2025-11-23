"""
Network class representing the transportation network of a city.
"""

import networkx as nx
import sys
import os
import json
from datetime import datetime, timedelta

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from src.route import Route
from src.hex import Hex
from src.actions.walk import Walk
from src.actions.wait import Wait
from src.actions.ride import Ride

class Network:
    """
    Represents the transportation network of a city.
    
    Attributes:
        graph (nx.Graph): A graph object from the networkx library, where nodes are Hex objects.
        services (list): A list of Service objects available in the network (e.g., public transport, ride-sharing).
        routes_taken (list): A list of Route objects, representing the routes that have been taken during a simulation.
    """
    
    def __init__(self, geojson_file_path: str):
        """
        Initialize a Network object.
        
        Args:
            geojson_file_path (str): Path to the GeoJSON file for the city.
        """
        self.graph = utils.construct_graph(geojson_file_path)
        self.services = []  # Will be populated later
        self.routes_taken = []  # Will be populated during simulation
        self.fixed_route_services = []  # List of FixedRouteService objects
    
    def get_optimal_route(self, demand, start_datetime=None):
        """
        Get the optimal route for a given demand, considering walking and fixed route services.
        
        Args:
            demand: A Demand object representing the travel request.
            start_datetime (datetime, optional): Simulation reference time for the route.
            
        Returns:
            Route: The optimal route to fulfill the demand.
        """
        try:
            start_node = demand.start_hex.hex_id
            end_node = demand.end_hex.hex_id
            
            # Check if nodes exist in graph
            if start_node not in self.graph.nodes() or end_node not in self.graph.nodes():
                return None
            
            # Create start time (simulation start time)
            if start_datetime is None:
                start_time = datetime.now().replace(second=0, microsecond=0)
            else:
                start_time = start_datetime.replace(second=0, microsecond=0)
            
            # Calculate walking-only route
            walk_route = self._calculate_walk_route(demand, start_time)
            routes = [walk_route] if walk_route else []
            
            # Calculate routes using fixed route services
            for service in self.fixed_route_services:
                service_route = self._calculate_fixed_route_service_route(demand, service, start_time)
                if service_route:
                    routes.append(service_route)
            
            # Return the fastest route (shortest total time)
            if not routes:
                return None
            
            # Find route with minimum total time
            best_route = min(routes, key=lambda r: self._get_route_total_time(r))
            return best_route
            
        except Exception as e:
            print(f"Error in get_optimal_route: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_walk_speed_from_config(self):
        """Load walking speed from config.json."""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('walk_speed_hex_per_hour', 10.0)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return 10.0
    
    def push_route(self, route):
        """
        Add a route to the network's taken routes.
        
        Args:
            route: A Route object to add to the network.
        """
        if route is not None:
            self.routes_taken.append(route)
    
    def clear(self):
        """
        Clear all taken routes from the network.
        """
        self.routes_taken.clear()
    
    def _calculate_walk_route(self, demand, start_time):
        """Calculate walking-only route."""
        start_node = demand.start_hex.hex_id
        end_node = demand.end_hex.hex_id
        
        try:
            path = nx.shortest_path(self.graph, start_node, end_node)
        except nx.NetworkXNoPath:
            return None
        
        distance = len(path) - 1
        walk_speed = self._load_walk_speed_from_config()
        total_time_hours = distance / walk_speed
        total_time_minutes = total_time_hours * 60
        
        walk_action_data = {
            'type': 'Walk',
            'start_time': start_time,
            'end_time': start_time + timedelta(minutes=total_time_minutes),
            'start_hex': demand.start_hex,
            'end_hex': demand.end_hex,
            'walk_speed': walk_speed,
            'distance': distance,
            'duration_minutes': total_time_minutes
        }
        
        return Route(unit=demand.unit, actions=[walk_action_data])
    
    def _calculate_fixed_route_service_route(self, demand, service, start_time):
        """Calculate route using a fixed route service."""
        start_hex_id = demand.start_hex.hex_id
        end_hex_id = demand.end_hex.hex_id
        
        # Find closest stops
        boarding_stop = self._find_closest_stop(start_hex_id, service.stops)
        alighting_stop = self._find_closest_stop(end_hex_id, service.stops)
        
        if boarding_stop is None or alighting_stop is None:
            return None
        
        boarding_stop_index = service.stops.index(boarding_stop)
        alighting_stop_index = service.stops.index(alighting_stop)
        
        # Must travel in the direction of the route
        if boarding_stop_index >= alighting_stop_index:
            return None  # Can't go backwards on a fixed route
        
        actions = []
        current_time = start_time
        
        # 1. Walk to boarding stop (if needed)
        if start_hex_id != boarding_stop:
            walk_time = self._calculate_walk_time(start_hex_id, boarding_stop)
            walk_end_time = current_time + timedelta(minutes=walk_time)
            walk_action = Walk(
                start_time=current_time,
                start_hex=demand.start_hex,
                end_hex=Hex(boarding_stop)
            )
            walk_action.end_time = walk_end_time
            actions.append(walk_action)
            current_time = walk_end_time
        
        # 2. Wait for vehicle (if needed)
        next_arrival = service.get_next_vehicle_arrival(boarding_stop, current_time)
        if next_arrival is None:
            return None  # No more vehicles
        
        wait_time = (next_arrival - current_time).total_seconds() / 60.0
        if wait_time > 0:
            wait_action = Wait(
                start_time=current_time,
                position=boarding_stop,
                unit=demand.unit,
                end_time=next_arrival
            )
            actions.append(wait_action)
            current_time = next_arrival
        
        # 3. Find the vehicle that arrives at boarding stop
        vehicle_index = None
        for i, vehicle in enumerate(service.vehicles):
            if boarding_stop_index < len(vehicle.timetable):
                arrival_time = vehicle.timetable[boarding_stop_index].replace(second=0, microsecond=0)
                if arrival_time == current_time.replace(second=0, microsecond=0):
                    vehicle_index = i
                    break
        
        if vehicle_index is None:
            return None  # Couldn't find matching vehicle
        
        # 4. Ride on vehicle
        vehicle = service.vehicles[vehicle_index]
        if alighting_stop_index >= len(vehicle.timetable):
            return None
        
        ride_end_time = vehicle.timetable[alighting_stop_index]
        ride_action = Ride(
            start_time=current_time,
            end_time=ride_end_time,
            name=service.name,
            vehicle_index=vehicle_index,
            start_hex=boarding_stop,
            end_hex=alighting_stop
        )
        actions.append(ride_action)
        current_time = ride_end_time
        
        # 5. Walk from alighting stop to destination (if needed)
        if alighting_stop != end_hex_id:
            walk_time = self._calculate_walk_time(alighting_stop, end_hex_id)
            walk_end_time = current_time + timedelta(minutes=walk_time)
            walk_action = Walk(
                start_time=current_time,
                start_hex=Hex(alighting_stop),
                end_hex=demand.end_hex
            )
            walk_action.end_time = walk_end_time
            actions.append(walk_action)
        
        return Route(unit=demand.unit, actions=actions)
    
    def _find_closest_stop(self, hex_id, stops):
        """Find the closest stop to a hex ID."""
        if not stops or hex_id not in self.graph.nodes():
            return None
        
        if hex_id in stops:
            return hex_id  # Already at a stop
        
        min_distance = float('inf')
        closest_stop = None
        
        for stop in stops:
            if stop not in self.graph.nodes():
                continue
            try:
                distance = nx.shortest_path_length(self.graph, hex_id, stop)
                if distance < min_distance:
                    min_distance = distance
                    closest_stop = stop
            except nx.NetworkXNoPath:
                continue
        
        return closest_stop
    
    def _calculate_walk_time(self, start_hex_id, end_hex_id):
        """Calculate walking time in minutes between two hex IDs."""
        if start_hex_id == end_hex_id:
            return 0.0
        
        if start_hex_id not in self.graph.nodes() or end_hex_id not in self.graph.nodes():
            return float('inf')
        
        try:
            distance = nx.shortest_path_length(self.graph, start_hex_id, end_hex_id)
            walk_speed = self._load_walk_speed_from_config()
            total_time_hours = distance / walk_speed
            return total_time_hours * 60.0
        except nx.NetworkXNoPath:
            return float('inf')
    
    def _get_route_total_time(self, route):
        """Get total time of a route in minutes."""
        if not route.actions:
            return float('inf')
        
        start_times = []
        end_times = []
        
        for action in route.actions:
            if hasattr(action, 'start_time'):
                start_times.append(action.start_time)
                if hasattr(action, 'end_time') and action.end_time:
                    end_times.append(action.end_time)
            elif isinstance(action, dict):
                if 'start_time' in action:
                    start_times.append(action['start_time'])
                if 'end_time' in action and action['end_time']:
                    end_times.append(action['end_time'])
        
        if not start_times or not end_times:
            return float('inf')
        
        total_time = max(end_times) - min(start_times)
        return total_time.total_seconds() / 60.0
    
    def __repr__(self):
        return f"Network(graph_nodes={len(self.graph.nodes())}, graph_edges={len(self.graph.edges())}, routes_taken={len(self.routes_taken)})"
