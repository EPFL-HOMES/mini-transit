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

            demand_hour = demand.hour
            
            # Check if nodes exist in graph
            if start_node not in self.graph.nodes() or end_node not in self.graph.nodes():
                return None
            

            # --- Option 1: Direct Walk ---
            # Find shortest path
            #TODO: clean up code for walk time calculation
            #also clean up what to do with all the Nones
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

            start_time = datetime.now()  # This will be updated with actual simulation time (that's what the other guy wanted to do anyway but we'll see)
            #TODO: STANDARDIZE TIME HANDLING, discuss with the other devs
            
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
            walk_route = Route(
                unit=demand.unit,
                actions=[walk_action_data]
            )
            
            # --- Option 2: Use Service ---
            best_service_route = None
            min_total_time = float('inf')

            #iterating through all services to find best one
            for service in self.services:
                if not isinstance(service, FixedRouteService):
                    continue

                for i in range(len(service.stops) - 1):
                    service_start = service.stops[i]
                    service_end = service.stops[i + 1]

                    # FixedRouteService.vehicles format: List[TypingOrderedDict[int, Tuple[datetime,datetime]]],

                    walk1 = Walk(
                        start_time=demand_hour,
                        start_hex=demand.start_hex,
                        end_hex=service_start,
                        walk_speed=walk_speed,
                        graph=self.graph
                    )

                    walk1.end_time  # this is just to note that end_time should've already been calculated in Walk

                    total_actions = [walk1]

                    #TODO: VERY IMPORTANT: implement route transfer logic based on whether the service can accommodate the demand at the given time and whether the service route includes the start and end hexes. This will likely involve checking the service's timetable and vehicle availability, as well as calculating wait times and ride times for the service.
                    # get the wait and ride actions for ONE particular service route
                    wait, ride = service.get_route(demand.unit, walk1.end_time, service_start, service_end)
                    total_actions.extend([wait, ride])

                    walk2 = Walk(
                        start_time=demand_hour,
                        start_hex=service_end,
                        end_hex=demand.end_hex,
                        walk_speed=walk_speed,
                        graph=self.graph,
                    )
                    total_actions.append(walk2)

                    route = Route(unit=demand.unit, actions=total_actions)

                    # time_taken calculation should already be handled by the Route class
                    route.total_fare = ride.fare # Assuming fare is calculated in the Ride action


                    if route.time_taken < min_total_time:
                        min_total_time = route.time_taken
                        best_service_route = route

            # --- Compare & Return ---
            if walk_route.time_taken < min_total_time:
                return walk_route
            else:
                return best_service_route if best_service_route else walk_route
            
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
