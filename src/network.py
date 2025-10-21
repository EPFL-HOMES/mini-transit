"""
Network class representing the transportation network of a city.
"""

import networkx as nx
import sys
import os
import json

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
        
        try:
            # Find shortest path using NetworkX
            start_node = demand.start_hex.hex_id
            end_node = demand.end_hex.hex_id
            
            # Check if nodes exist in graph
            if start_node not in self.graph.nodes() or end_node not in self.graph.nodes():
                return None
            
            # Find shortest path
            try:
                path = nx.shortest_path(self.graph, start_node, end_node)
            except nx.NetworkXNoPath:
                # No path exists between start and end
                return None
            
            # Calculate total distance (number of edges in path)
            distance = len(path) - 1  # Number of edges = number of nodes - 1
            
            # Load walking speed from config
            walk_speed = self._load_walk_speed_from_config()
            
            # Calculate total time
            total_time_hours = distance / walk_speed
            total_time_minutes = total_time_hours * 60
            
            # Create start time (simulation start time)
            start_time = datetime.now()  # This will be updated with actual simulation time
            
            # Create a simple walk action dictionary (avoiding Walk class import issues)
            walk_action_data = {
                'type': 'Walk',
                'start_time': start_time,
                'end_time': start_time + timedelta(minutes=total_time_minutes),
                'start_hex': demand.start_hex,
                'end_hex': demand.end_hex,
                'walk_speed': walk_speed,
                'distance': distance
            }
            
            # Create route with simple action data
            route = Route(
                unit=demand.unit,
                actions=[walk_action_data]
            )
            
            return route
            
        except Exception as e:
            print(f"Error in get_optimal_route: {e}")
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
    
    def __repr__(self):
        return f"Network(graph_nodes={len(self.graph.nodes())}, graph_edges={len(self.graph.edges())}, routes_taken={len(self.routes_taken)})"
