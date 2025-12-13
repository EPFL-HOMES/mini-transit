"""
Network class representing the transportation network of a city.
"""

import json
import os
import sys

import networkx as nx

from src.actions.walk import Walk
from src.actions.ride import Ride
from src.services.fixedroute import FixedRouteService
from src.services.ondemand import *
import numpy as np
from datetime import datetime, timedelta
# Dynamic imports to avoid circular import issues
try:
    from .route import Route
except ImportError:
    from src.route import Route

#TODO: actually find a way to input this to be customizable

MAX_INTERMEDIARIES = 1

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
    
    def find_closest_ondemand_vehicle(self, graph, hex_id, service, walk_speed, demand_time, radius=2):
        import math
        best_vehicle = None
        best_time = float('inf')
        vehicle_within_radius_count = 0
        for vehicle in service.vehicles:
            # count if vehicle is within radius
            distance = nx.shortest_path_length(graph, source=hex_id, target=vehicle.current_location.hex_id)
            if distance <= radius:
                if not vehicle.is_available(demand_time):
                    continue  # Vehicle is not available
                vehicle_within_radius_count += 1
                vehicle_location = vehicle.current_location.hex_id
                # compute walk time (for vehicle choosing purposes only)
                walk_time, _ = self.compute_walk_time(graph, hex_id, vehicle_location, walk_speed)
                area = (1 + (radius-1)*6 + (radius-2)*(radius-1)/2) # total hexes within the radius
                if walk_time < best_time: # we pick the vehicle based on the actual walk time
                    best_vehicle = vehicle
                # actual metric chosen to calculate "walk time" for the purpose or route calculation:
                walk_time_metric = (1/(2*walk_speed)) * math.sqrt(area/vehicle_within_radius_count)  # in hours? maybe??? idk
                
        return best_vehicle, walk_time_metric
    
    def find_closest_ondemand_dock(self, graph, hex_id, service, walk_speed, radius=2):
        best_dock = None
        best_time = float('inf')
        for dock in service.docking_stations:
            # count if dock is within radius
            distance = nx.shortest_path_length(graph, source=hex_id, target=dock.location.hex_id)
            if distance <= radius:
                # we don't actually care about the availability of the dock since the assumption is the demand will only find out once they actually reach the dock
                walk_time, _ = self.compute_walk_time(graph, hex_id, dock.location.hex_id, walk_speed)
                if walk_time < best_time:
                    best_time = walk_time
                    best_dock = dock
        return best_dock, best_time

    
    def find_service_chains(self, s_start, s_end, all_services, max_intermediaries):
        """
        Produce all valid service chains from s_start → ... → s_end,
        with at most max_intermediaries middle services.
        """
        chains = []
        visited = set()

        def dfs(current, chain, depth):
            # depth counts *middle* services used so far
            if depth > max_intermediaries:
                return

            if current == s_end:
                chains.append(chain[:])
                return

            visited.add(current)

            for svc in all_services:
                if svc in visited:
                    continue
                if not isinstance(svc, FixedRouteService):
                    continue

                # must have a feasible connection (walk or shared stop)
                if self.services_can_link(current, svc):
                    chain.append(svc)
                    dfs(svc, chain, depth + 1)
                    chain.pop()

            visited.remove(current)

        dfs(s_start, [s_start], 0)
        return chains
    
    def services_can_link(self, sa, sb):
        """
        Returns dict with:
            {
                'ok': bool,
                'transfer_pairs': [ (stop_A, stop_B, walk_time_hours), ... ]
            }
        """

        walk_speed = self._load_walk_speed_from_config()

        # 1. Shared stops → zero walking transfer
        shared = set(sa.stops) & set(sb.stops)
        if shared:
            return {
                'ok': True,
                'transfer_pairs': [(x, x, 0) for x in shared]
            }

        # 2. Otherwise walkable transfer between ANY stop pair (in THIS case we would want the one with the least amount of walking time)
        best = []
        inf = float('inf')
        best_time = inf
        for a in sa.stops:
            for b in sb.stops:
                walk_time, _ = self.compute_walk_time(self.graph, a.hex_id, b.hex_id, walk_speed)
                if walk_time < best_time:
                    best_time = walk_time
                    best = [(a, b, walk_time)]
                elif walk_time == best_time:
                    best.append((a, b, walk_time))

        if best:
            return {
                'ok': True,
                'transfer_pairs': best
            }

        return {'ok': False, 'transfer_pairs': []}
    
    def build_route_for_chain(self, chain, start, end, demand_time, demand, graph, walk_speed):
        """
        chain: [s1, sm1, sm2, ..., s2]
        Returns: Route or raises Exception if something invalid.
        """

        actions = []
        current_time = demand_time
        num_transfers = len(chain) - 1

        # Step 1: walk from origin to the start service’s nearest stop
        s_first = chain[0]
        s1_start, s1_walk_time = self.find_closest_stop(graph, start, s_first, walk_speed)
        if s1_start is None:
            raise Exception("No feasible access stop for first service")

        walk_to_start = Walk(
            start_time=current_time,
            end_time=current_time + timedelta(hours=s1_walk_time),
            start_hex=start,
            end_hex=s1_start,
            walk_speed=walk_speed,
            unit=demand.unit
        )
        actions.append(walk_to_start)
        current_time = walk_to_start.end_time

        # Traverse service pairs
        for i in range(len(chain) - 1):
            sa = chain[i]
            sb = chain[i + 1]

            link = self.services_can_link(sa, sb, walk_speed)
            _, _, walk_time_hours = link['transfer_pairs'][0] # should either be the pair with the best walking time or one that doesn't need walking at all
            if walk_time_hours != 0:
                num_transfers += 1 # extra transfer for walking transfer
            # at this point we SHOULD'VE already checked that they can link in the find_service_chains step
            if not link['ok']:
                raise Exception("Invalid link in chain")

            # choose the *first* feasible pair for simplicity
            (a_stop, b_stop, walk_t) = link['transfer_pairs'][0]

            # ride on service sa from its nearest start to a_stop
            wait_sa, ride_sa = sa.get_route(
                unit=demand.unit,
                start_time=current_time,
                start_hex=s1_start if i == 0 else prev_b,
                end_hex=a_stop
            )
            actions.extend([wait_sa, ride_sa])
            current_time = ride_sa.end_time

            # walk transfer if needed
            if a_stop != b_stop:
                transfer_walk = Walk(
                    start_time=current_time,
                    end_time=current_time + timedelta(hours=walk_t),
                    start_hex=a_stop,
                    end_hex=b_stop,
                    walk_speed=walk_speed,
                    unit=demand.unit
                )
                actions.append(transfer_walk)
                current_time = transfer_walk.end_time

            prev_b = b_stop

        # Step 3: final ride on last service to closest end stop
        s_last = chain[-1]
        s2_end, s2_walk_time = self.find_closest_stop(graph, end, s_last, walk_speed)
        if s2_end is None:
            raise Exception("No feasible end stop for last service")

        wait_last, ride_last = s_last.get_route(
            unit=demand.unit,
            start_time=current_time,
            start_hex=prev_b,
            end_hex=s2_end
        )
        actions.extend([wait_last, ride_last])
        current_time = ride_last.end_time

        # Step 4: final walk to destination
        final_walk = Walk(
            start_time=current_time,
            end_time=current_time + timedelta(hours=s2_walk_time),
            start_hex=s2_end,
            end_hex=end,
            walk_speed=walk_speed,
            unit=demand.unit
        )
        actions.append(final_walk)

        #TODO: calculate transfers properly in the form of including transfers BETWEEN walking and ride actions etc
        # still leaving todo here in case i'm missing something
        return Route(unit=demand.unit, actions=actions, transfers=num_transfers)

    
    def get_optimal_route(self, demand, second_try=False):
        """
        Get the optimal route for a given demand using shortest path algorithm.

        Args:
            demand: A Demand object representing the travel request.
            second_try: A boolean indicating if this is a second attempt to find a route after a prior attempt failed (e.g. discovering no available vehicles after arriving at a dock). (For now) mainly used to indicate that the user isn't willing to take docked on-demand services anymore.

        Returns:
            Route: The optimal route to fulfill the demand.
        """
        
        walk_speed = self._load_walk_speed_from_config()
        # Find shortest path using NetworkX
        start = demand.start_hex.hex_id
        end = demand.end_hex.hex_id
        demand_time = demand.time

        walk_best_route = None
        walk_best_cost = -float('inf')
        walk_fixed_best_route = None
        walk_fixed_best_cost = -float('inf')
        #ondemand_fixed_best_route = None
        #ondemand_fixed_best_cost = float('inf')
        ondemanddocked_best_route = None
        ondemanddockless_best_route = None
        ondemanddocked_best_cost = walk_route.total_cost if walk_route else -float('inf')
        
        # 1. Try direct walk
        walk_time, walk_path = self.compute_walk_time(self.graph, start, end, walk_speed)
        if walk_time < float("inf"):
            walk_action = Walk(
                start_time=demand_time,
                start_hex=start,
                end_hex=end,
                unit=demand.unit,
                walk_speed=walk_speed,
                end_time=demand_time + timedelta(hours=walk_time),
            )

            walk_route = Route(unit=demand.unit, actions=[walk_action], transfers=0)
        else:
            walk_route = None

        walk_best_route = walk_route
        walk_best_cost = walk_route.total_cost if walk_route else float('inf')

        # 2. Try other combinations of services

        # Current algorithm: try all pairs of services (including same service twice)
        for s1 in self.services:
            # 2.1.1 implement for all OnDemandRouteServiceDocked directly
            if isinstance(s1, OnDemandRouteServiceDocked):
                if second_try:
                    continue  # skip docked services altogether on second try

                best_start_dock, vehicle_walk_time = self.find_closest_ondemand_dock(self.graph, start, s1, walk_speed, radius=2)
                if best_start_dock is None: # aka literally no docks at all within the radius
                    continue
                best_end_dock, off_vehicle_walk_time = self.find_closest_ondemand_dock(self.graph, end, s1, walk_speed, radius=2)
                # Walk to vehicle
                walk_to_vehicle = Walk(
                    start_time=demand_time,
                    end_time=demand_time + timedelta(hours=vehicle_walk_time),
                    start_hex=start,
                    end_hex=best_start_dock.location,
                    unit=demand.unit,
                    walk_speed=walk_speed
                )
                vehicle, _ = best_start_dock.take_vehicle()
                if vehicle is None: # no available vehicle at the dock
                    # try to find another route without docked OnDemand service
                    demand.time = walk_to_vehicle.end_time
                    return self.get_optimal_route(demand, second_try=True)
                # below case for when there IS an available vehicle
                # Ride with vehicle
                drive_time = s1.compute_drive_time(best_start_dock.location, end)
                arrival_time = walk_to_vehicle.end_time + drive_time
                ride_action = Ride(
                    start_time=walk_to_vehicle.end_time,
                    end_time=arrival_time,
                    start_hex=best_start_dock.location,
                    end_hex=best_end_dock.location,
                    unit=demand.unit,
                    service=s1,
                    vehicle=vehicle
                )
                walk_from_vehicle = Walk(
                    start_time=ride_action.end_time,
                    end_time=ride_action.end_time + timedelta(hours=off_vehicle_walk_time),
                    start_hex=best_end_dock.location,
                    end_hex=end,
                    unit=demand.unit,
                    walk_speed=walk_speed
                )
                ondemand_route = Route(unit=demand.unit, actions=[walk_to_vehicle, ride_action, walk_from_vehicle], transfers=0)
                if ondemand_route.total_cost < ondemanddocked_best_cost:
                    ondemanddocked_best_cost = ondemand_route.total_cost
                    ondemanddocked_best_route = ondemand_route
            elif isinstance(s1, OnDemandRouteServiceDockless):
                best_vehicle, vehicle_walk_time = self.find_closest_ondemand_vehicle(self.graph, start, s1, walk_speed, demand_time, radius=2)
                if best_vehicle is None: # aka literally no vehicles at all within the radius
                    continue
                # Walk to vehicle
                walk_to_vehicle = Walk(
                    start_time=demand_time,
                    end_time=demand_time + timedelta(hours=vehicle_walk_time),
                    start_hex=start,
                    end_hex=best_vehicle.current_location,
                    unit=demand.unit,
                    walk_speed=walk_speed
                )
                # Ride with vehicle
                drive_time = s1.compute_drive_time(best_vehicle.current_location, end)
                arrival_time = walk_to_vehicle.end_time + drive_time
                ride_action = Ride(
                    start_time=walk_to_vehicle.end_time,
                    end_time=arrival_time,
                    start_hex=best_vehicle.current_location,
                    end_hex=end,
                    unit=demand.unit,
                    service=s1,
                    vehicle=best_vehicle
                )
                ondemand_route = Route(unit=demand.unit, actions=[walk_to_vehicle, ride_action], transfers=0)
                if ondemand_route.total_cost < ondemanddockless_best_cost:
                    ondemanddockless_best_cost = ondemand_route.total_cost
                    ondemanddockless_best_route = ondemand_route
            else: 
            # 2.2 FixedRouteService cases
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
                            walk1 = Walk(demand_time, demand_time + timedelta(hours=s1_walk_time), start, s1_start, unit=demand.unit, walk_speed=walk_speed)
                            wait, ride = s1.get_route(demand.unit, walk1.end_time, s1_start, s2_end)
                            walk2 = Walk(ride.end_time, ride.end_time + timedelta(hours=s2_walk_time), s2_end, end, unit=demand.unit, walk_speed=walk_speed)

                            route = Route(unit=demand.unit, actions=[walk1, wait, ride, walk2], transfers=0)

                            if route.total_cost < walk_fixed_best_cost:
                                walk_fixed_best_cost = route.total_cost
                                walk_fixed_best_route = route

                    # CASE 2: Try transfer via common stop
                    else:
                        common_stops = set(s1.stops) & set(s2.stops)
                        if common_stops:
                            for transfer_stop in common_stops:
                                if (s1_start in s1.stops and transfer_stop in s1.stops and
                                    transfer_stop in s2.stops and s2_end in s2.stops):
                                    # Ensure valid directions
                                    if s1.stops.index(s1_start) < s1.stops.index(transfer_stop) and \
                                    s2.stops.index(transfer_stop) < s2.stops.index(s2_end):

                                        walk1 = Walk(demand_time, demand_time + timedelta(hours=s1_walk_time), start, s1_start, unit=demand.unit, walk_speed=walk_speed)
                                        wait1, ride1 = s1.get_route(demand.unit, walk1.end_time, s1_start, transfer_stop)
                                        wait2, ride2 = s2.get_route(demand.unit, ride1.end_time, transfer_stop, s2_end)
                                        walk2 = Walk(ride.end_time, ride.end_time + timedelta(hours=s2_walk_time), s2_end, end, unit=demand.unit, walk_speed=walk_speed)

                                        route = Route(unit=demand.unit, actions=[walk1, wait1, ride1, wait2, ride2, walk2], transfers=1)

                                        if route.total_cost < walk_fixed_best_cost:
                                            walk_fixed_best_cost = route.total_cost
                                            walk_fixed_best_route = route
                    
                        # CASE 3: No shared stops - try to bridge s1 and s2 via at least one intermediary service
                        else:
                            chains = self.find_service_chains(s1, s2, self.services, MAX_INTERMEDIARIES)

                            for chain in chains:
                                try:
                                    route = self.build_route_for_chain(
                                        chain=chain,
                                        start=start,
                                        end=end,
                                        demand_time=demand_time,
                                        demand=demand,
                                        graph=self.graph,
                                        walk_speed=walk_speed
                                    )

                                    if route.total_cost < walk_fixed_best_cost:
                                        walk_fixed_best_cost = route.total_cost
                                        walk_fixed_best_route = route

                                except Exception:
                                    continue
                        
        # After evaluating all options, determine route probabilities via softmax
        # foolproofing for None routes
        choices = [walk_best_route]
        logits = [walk_best_cost]
        if walk_fixed_best_route is not None:
            logits.append(walk_fixed_best_cost)
            choices.append(walk_fixed_best_route)
        if ondemanddocked_best_route is not None:
            logits.append(ondemanddocked_best_cost)
            choices.append(ondemanddocked_best_route)
        if ondemanddockless_best_route is not None:
            logits.append(ondemanddockless_best_cost)
            choices.append(ondemanddockless_best_route)
        exp_logits = np.exp(np.array(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Select route based on probabilities
        choice = np.random.choice(choices, p=probabilities)

        # if chosen route has OnDemand, set the vehicle to be used/unavailable
        # until end_time where the vehicle is free again at the new location
        for action in choice.actions:
            if (isinstance(action, Ride) and isinstance(action.service, OnDemandRouteServiceDocked)) or (isinstance(action, Ride) and isinstance(action.service, OnDemandRouteServiceDockless)):
                # Find the vehicle that is closest to the start_hex at start_time. Should reasonably be the same one used in the Ride action.
                vehicle = action.vehicle
                if vehicle is not None:
                    # Update vehicle's location
                    vehicle.current_location = action.end_hex
                    # simulate being unavailable until the end of the action
                    vehicle.available_time = action.end_time
                    
        return choice
        
    
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
