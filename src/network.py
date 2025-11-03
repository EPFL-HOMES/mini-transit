"""
Network class representing the transportation network of a city.
"""

import networkx as nx
import sys
import os
import json
from src.actions.walk import Walk
from src.services.fixedroute import FixedRouteService

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Simulation classes will be imported dynamically to avoid circular imports

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
        
    def get_distance(self, start_hex, end_hex) -> int:
        """
        Get the distance in hexagons between two hexes using the network graph.
        
        Args:
            start_hex (Hex): The starting hexagon.
            end_hex (Hex): The destination hexagon.
        Returns:
            int: Distance in hexagons.
        """
        try:
            length = nx.shortest_path_length(self.graph, start_hex.hex_id, end_hex.hex_id)
            return length
        except nx.NetworkXNoPath:
            return float('inf')  # No path exists
        
        
    def compute_walk_time(self, graph, from_hex, to_hex, walk_speed):
        """Returns walk time (in minutes) and path if exists, else (inf, None)
        Specially reserved for isolated computation cases outside of Walk class"""
        try:
            distance = nx.shortest_path_length(graph, source=from_hex, target=to_hex, weight='length')
            time_hours = distance / walk_speed 
            return time_hours, nx.shortest_path(graph, source=from_hex, target=to_hex)
        except nx.NetworkXNoPath:
            return float('inf'), None


    def find_closest_stop(self, graph, hex_id, service, walk_speed):
        best_stop = None
        best_time = float('inf')
        for stop in service.stops:
            walk_time, _ = self.compute_walk_time(graph, hex_id, stop, walk_speed)
            if walk_time < best_time:
                best_time = walk_time
                best_stop = stop
        return best_stop, best_time
    
    
    def get_optimal_route(self, demand):
        """
        Get the optimal route for a given demand using shortest path algorithm.
        
        Args:
            demand: A Demand object representing the travel request.
            
        Returns:
            Route: The optimal route to fulfill the demand.
        """
        from datetime import datetime, timedelta
        
        # Dynamic imports to avoid circular import issues
        try:
            from .route import Route
        except ImportError:
            from src.route import Route

        walk_speed = self._load_walk_speed_from_config()
        # Find shortest path using NetworkX
        start = demand.start_hex.hex_id
        end = demand.end_hex.hex_id
        demand_hour = demand.hour
        
        # 1. Try direct walk
        walk_time, walk_path = self.compute_walk_time(self.graph, start, end, walk_speed)
        if walk_time < float('inf'):
            walk_action = Walk(
                start_time=demand_hour,
                start_hex=start,
                end_hex=end,
                unit=demand.unit,
                walk_speed=walk_speed,
                end_time=demand_hour + timedelta(hours=walk_time)
            )

            walk_route = Route(unit=demand.unit, actions=[walk_action])
        else:
            walk_route = None

        # 2. Try all combinations of services
        best_route = walk_route
        best_time = walk_route.time_taken if walk_route else float('inf')
        # TODO: for now implementation is for only possbilities where at most one transfer occurs

        # Current algorithm: try all pairs of services (including same service twice)
        for s1 in self.services:
            for s2 in self.services:
                if not isinstance(s1, FixedRouteService) or not isinstance(s2, FixedRouteService):
                    continue

                s1_start, s1_walk_time = self.find_closest_stop(self.graph, start, s1, walk_speed)
                s2_end, s2_walk_time = self.find_closest_stop(self.graph, end, s2, walk_speed)

                if s1_start is None or s2_end is None:
                    continue

                # CASE 1: Same service
                if s1 == s2 and s1_start in s1.stops and s2_end in s1.stops:
                    idx1 = s1.stops.index(s1_start)
                    idx2 = s1.stops.index(s2_end)
                    if idx1 < idx2:  # Ensure forward direction along the line's actual direction   
                        walk1 = Walk(demand_hour, demand_hour + timedelta(hours=s1_walk_time), start, s1_start, unit=demand.unit, walk_speed=walk_speed)
                        wait, ride = s1.get_route(demand.unit, walk1.end_time, s1_start, s2_end)
                        walk2 = Walk(ride.end_time, ride.end_time + timedelta(hours=s2_walk_time), s2_end, end, unit=demand.unit, walk_speed=walk_speed)

                        route = Route(unit=demand.unit, actions=[walk1, wait, ride, walk2])

                        if route.time_taken < best_time:
                            best_time = route.time_taken
                            best_route = route

                # CASE 2: Try transfer via common stop
                else:
                    common_stops = set(s1.stops) & set(s2.stops)
                    for transfer_stop in common_stops:
                        if (s1_start in s1.stops and transfer_stop in s1.stops and
                            transfer_stop in s2.stops and s2_end in s2.stops):
                            # Ensure valid directions
                            if s1.stops.index(s1_start) < s1.stops.index(transfer_stop) and \
                            s2.stops.index(transfer_stop) < s2.stops.index(s2_end):

                                walk1 = Walk(demand_hour, demand_hour + timedelta(hours=s1_walk_time), start, s1_start, unit=demand.unit, walk_speed=walk_speed)
                                wait1, ride1 = s1.get_route(demand.unit, walk1.end_time, s1_start, transfer_stop)
                                wait2, ride2 = s2.get_route(demand.unit, ride1.end_time, transfer_stop, s2_end)
                                walk2 = Walk(ride.end_time, ride.end_time + timedelta(hours=s2_walk_time), s2_end, end, unit=demand.unit, walk_speed=walk_speed)

                                route = Route(unit=demand.unit, actions=[walk1, wait1, ride1, wait2, ride2, walk2])

                                if route.time_taken < best_time:
                                    best_time = route.time_taken
                                    best_route = route
                
                # TODO: future cases:
                # CASE 3: Different services with no common stop (not implemented yet) aka literally walk the gap between services
                # CASE 4: More than 2 services transfer (not implemented yet)

        return best_route
    
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
    
    def __repr__(self):
        return f"Network(graph_nodes={len(self.graph.nodes())}, graph_edges={len(self.graph.edges())}, routes_taken={len(self.routes_taken)})"
