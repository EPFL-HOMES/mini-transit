"""
Network class representing the transportation network of a city.
"""

import json
import os
import sys

import networkx as nx

from src.actions.walk import Walk
from src.hex import Hex
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
            return float("inf")  # No path exists

    def compute_walk_time(self, graph, from_hex, to_hex, walk_speed):
        """Returns walk time (in minutes) and path if exists, else (inf, None)
        Specially reserved for isolated computation cases outside of Walk class"""
        try:
            distance = nx.shortest_path_length(
                graph, source=from_hex, target=to_hex, weight="length"
            )
            time_hours = distance / walk_speed
            return time_hours, nx.shortest_path(graph, source=from_hex, target=to_hex)
        except nx.NetworkXNoPath:
            return float("inf"), None

    def find_closest_stop(self, graph, hex_id, service, walk_speed):
        best_stop = None
        best_time = float("inf")
        for stop in service.stops:
            walk_time, _ = self.compute_walk_time(graph, hex_id, stop.hex_id, walk_speed)
            if walk_time < best_time:
                best_time = walk_time
                best_stop = stop
        return best_stop, best_time

    def get_optimal_route(self, demand):
        """
        Get the optimal route for a given demand using shortest path algorithm.
        Searches through all fixed route services and finds the fastest route with at most one transfer.

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
        demand_time = demand.time

        # 1. Try direct walk
        walk_time, walk_path = self.compute_walk_time(self.graph, start, end, walk_speed)
        if walk_time < float("inf"):
            walk_action = Walk(
                start_time=demand_time,
                end_time=demand_time + timedelta(hours=walk_time),
                start_hex=demand.start_hex,
                end_hex=demand.end_hex,
                unit=demand.unit,
                graph=self.graph,
                walk_speed=walk_speed,
            )
            walk_route = Route(unit=demand.unit, actions=[walk_action])
        else:
            walk_route = None

        best_route = walk_route
        best_time = walk_route.time_taken if walk_route else float("inf")

        # Get all fixed route services
        fixed_services = [s for s in self.services if isinstance(s, FixedRouteService)]

        if not fixed_services:
            return best_route

        # 2. Try direct service routes (walk to stop, ride service, walk to destination)
        for service in fixed_services:
            try:
                # Find closest stops to start and end
                start_stop, start_walk_time = self.find_closest_stop(
                    self.graph, start, service, walk_speed
                )
                end_stop, end_walk_time = self.find_closest_stop(
                    self.graph, end, service, walk_speed
                )

                if start_stop is None or end_stop is None:
                    continue

                # Check if both stops are in the service
                if start_stop not in service.stops or end_stop not in service.stops:
                    continue

                # Try the route - get_route will find the appropriate vehicle/direction
                try:
                    # Walk to start stop
                    walk1 = Walk(
                        demand_time,
                        demand_time + timedelta(hours=start_walk_time),
                        Hex(start),
                        start_stop,
                        unit=demand.unit,
                        graph=self.graph,
                        walk_speed=walk_speed,
                    )

                    # Ride service (get_route handles finding the right direction/vehicle)
                    wait, ride = service.get_route(
                        demand.unit, walk1.end_time, start_stop, end_stop
                    )

                    # Walk to destination
                    walk2 = Walk(
                        ride.end_time,
                        ride.end_time + timedelta(hours=end_walk_time),
                        end_stop,
                        Hex(end),
                        unit=demand.unit,
                        graph=self.graph,
                        walk_speed=walk_speed,
                    )

                    route = Route(unit=demand.unit, actions=[walk1, wait, ride, walk2])

                    if route.time_taken < best_time:
                        best_time = route.time_taken
                        best_route = route
                except (ValueError, RuntimeError):
                    # Service route not available at this time, skip
                    continue
            except Exception:
                continue

        # 3. Try routes with one transfer (walk to stop1, ride service1, transfer, ride service2, walk to destination)
        for service1 in fixed_services:
            for service2 in fixed_services:
                if service1 == service2:
                    continue  # Already handled in direct routes

                try:
                    # Find closest stops
                    start_stop1, start_walk_time = self.find_closest_stop(
                        self.graph, start, service1, walk_speed
                    )
                    end_stop2, end_walk_time = self.find_closest_stop(
                        self.graph, end, service2, walk_speed
                    )

                    if start_stop1 is None or end_stop2 is None:
                        continue

                    if start_stop1 not in service1.stops or end_stop2 not in service2.stops:
                        continue

                    # Find common stops for transfer
                    common_stops = set(service1.stops) & set(service2.stops)

                    if not common_stops:
                        continue

                    # Try each common stop as transfer point
                    for transfer_stop in common_stops:
                        try:
                            # Try the transfer route - get_route will find the appropriate vehicle/direction
                            # We just need to ensure we're not trying to go from a stop to itself
                            if start_stop1 == transfer_stop or transfer_stop == end_stop2:
                                continue

                            # Walk to first service stop
                            walk1 = Walk(
                                demand_time,
                                demand_time + timedelta(hours=start_walk_time),
                                Hex(start),
                                start_stop1,
                                unit=demand.unit,
                                graph=self.graph,
                                walk_speed=walk_speed,
                            )

                            # Ride first service to transfer stop
                            wait1, ride1 = service1.get_route(
                                demand.unit, walk1.end_time, start_stop1, transfer_stop
                            )

                            # Ride second service from transfer stop
                            wait2, ride2 = service2.get_route(
                                demand.unit, ride1.end_time, transfer_stop, end_stop2
                            )

                            # Walk to destination
                            walk2 = Walk(
                                ride2.end_time,
                                ride2.end_time + timedelta(hours=end_walk_time),
                                end_stop2,
                                Hex(end),
                                unit=demand.unit,
                                graph=self.graph,
                                walk_speed=walk_speed,
                            )

                            route = Route(
                                unit=demand.unit,
                                actions=[walk1, wait1, ride1, wait2, ride2, walk2],
                            )

                            if route.time_taken < best_time:
                                best_time = route.time_taken
                                best_route = route
                        except (ValueError, RuntimeError):
                            # Transfer route not available at this time, skip
                            continue
                except Exception:
                    continue

        return best_route

    def _load_walk_speed_from_config(self):
        """Load walking speed from config.json."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "config.json"
            )
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("walk_speed_hex_per_hour", 10.0)
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
