"""
Network class representing the transportation network of a city.
"""

from dataclasses import dataclass
from datetime import timedelta

import traceback

import networkx as nx
from collections import defaultdict
import numpy as np

from .actions.ondemand_ride import OnDemandRide
from .actions.ride import Ride
from .actions.walk import Walk
from .primitives.route import RouteConfig
from .services.fixedroute import FixedRouteService
from .services.ondemand import *

# Dynamic imports to avoid circular import issues
try:
    from .primitives.route import Route
except ImportError:
    from .primitives.route import Route


from .graph import construct_graph

# Simulation classes will be imported dynamically to avoid circular imports


@dataclass
class NetworkConfig(RouteConfig):
    walk_speed: float = 10.0  # hexagons per hour


class Network:
    """
    Represents the transportation network of a city.

    Attributes:
        graph (nx.Graph): A graph object from the networkx library, where nodes are Hex objects.
        services (list): A list of Service objects available in the network (e.g., public transport, ride-sharing).
        routes_taken (list): A list of Route objects, representing the routes that have been taken during a simulation.
    """

    def __init__(self, geojson_file_path: str, config=NetworkConfig()):
        """
        Initialize a Network object.

        Args:
            geojson_file_path (str): Path to the GeoJSON file for the city.
        """
        self.config = config
        self.graph = construct_graph(geojson_file_path)
        self.fixedroute_graph = None  # Will be built later
        self.services = []  # Will be populated later
        self.routes_taken = []  # Will be populated during simulation
        self.fixedroute_lookup = {}  # Mapping from route name to FixedRouteService
        self.component_distance_table = None  # Will be built later
        self.path_lookup = defaultdict(dict)  # (from_hex_id, to_hex_id) -> (path, time)
        #self.construct_path_lookup()

    #def construct_path_lookup(self):
        """
        Constructs an empty lookup table for all the obtained fastest path in the format (List, time (in float)) between any two nodes in self.graph. 
        """
        #for u in self.graph.nodes:
            #for v in self.graph.nodes:
                #self.path_lookup[u][v] = (None, float("inf"))
        

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
        """Returns walk time (in hours) and path if exists, else (inf, None)
        Specially reserved for isolated computation cases outside of Walk class"""
        try:
            distance = nx.shortest_path_length(
                graph, source=from_hex, target=to_hex, weight="length"
            )
            time_hours = distance / walk_speed
            return time_hours, nx.shortest_path(graph, source=from_hex, target=to_hex)
        except nx.NetworkXNoPath:
            return float("inf"), None

    def find_closest_stop(self, graph, hex_id, walk_speed):
        best_stop = None
        best_time = float("inf")
        for stop in self.fixedroute_graph.nodes:
            walk_time, _ = self.compute_walk_time(graph, hex_id, stop, walk_speed)
            if walk_time < best_time:
                best_time = walk_time
                best_stop = stop
        return best_stop, best_time

    def find_closest_ondemand_vehicle(
        self, graph, hex_id, service, walk_speed, demand_time, radius=2
    ):
        import math

        best_vehicle = None
        best_time = float("inf")
        vehicle_within_radius_count = 0
        walk_time_metric = float("inf")
        for vehicle in service.vehicles:
            # count if vehicle is within radius
            distance = nx.shortest_path_length(
                graph, source=hex_id, target=vehicle.current_location.hex_id
            )
            if distance <= radius:
                if not vehicle.is_available(demand_time):
                    continue  # Vehicle is not available
                vehicle_within_radius_count += 1
                vehicle_location = vehicle.current_location.hex_id
                # compute walk time (for vehicle choosing purposes only)
                walk_time, _ = self.compute_walk_time(graph, hex_id, vehicle_location, walk_speed)
                area = (
                    1 + (radius - 1) * 6 + (radius - 2) * (radius - 1) / 2
                )  # total hexes within the radius
                if walk_time < best_time:  # we pick the vehicle based on the actual walk time
                    best_time = walk_time
                    best_vehicle = vehicle
                # actual metric chosen to calculate "walk time" for the purpose or route calculation:
                walk_time_metric = (1 / (2 * walk_speed)) * math.sqrt(
                    area / vehicle_within_radius_count
                )  # in hours? maybe??? idk

        return best_vehicle, walk_time_metric

    def find_closest_ondemand_dock(self, graph, hex_id, service, walk_speed, radius=2):
        best_dock = None
        best_time = float("inf")
        for dock in service.docking_stations:
            # count if dock is within radius
            distance = nx.shortest_path_length(graph, source=hex_id, target=dock.location.hex_id)
            if distance <= radius:
                # we don't actually care about the availability of the dock since the assumption is the demand will only find out once they actually reach the dock
                walk_time, _ = self.compute_walk_time(
                    graph, hex_id, dock.location.hex_id, walk_speed
                )
                if walk_time < best_time:
                    best_time = walk_time
                    best_dock = dock
        return best_dock, best_time


    def build_fixedroute_graph(self, fixed_routes):
        """
        Builds self.fixedroute_graph using all FixedRouteService objects.
        """
        G = nx.MultiDiGraph()

        for route in fixed_routes:
            if route.name in self.fixedroute_lookup:
                print(f"Warning: Duplicate route name detected: {route.name}")
            self.fixedroute_lookup[route.name] = route
            # Assume all vehicles along the same direction on the route have the same timetable
            # Should change if we ever use another scheduling method
            if route.bidirectional:
                vehicles = route.vehicles[:2]  # the first two vehicles represent both directions
            else:
                vehicles = route.vehicles[:1]
            for vehicle in vehicles:
                timetable = vehicle.timetable  # OrderedDict[int, (arr, dep)]
                stop_indices = list(timetable.keys())

                for i in range(len(stop_indices) - 1):
                    a = route.stops[stop_indices[i]]
                    b = route.stops[stop_indices[i + 1]]

                    dep_time = timetable[stop_indices[i]][1] # departure time of stop i
                    arr_time = timetable[stop_indices[i + 1]][0] # arrival time of stop i+1

                    minutes = (arr_time - dep_time).total_seconds() / 60

                    G.add_edge(
                        a.hex_id,
                        b.hex_id,
                        weight=minutes,
                        route=route.name,
                        #vehicle_id=id(vehicle),
                    )
        self.fixedroute_graph = G


    def build_component_distance_table(self):
        """
        Builds:
        - component id per node
        - min walking distance between every component pair
        - which nodes realize that min distance
        """
        G = self.fixedroute_graph
        components = list(nx.weakly_connected_components(G))
        node_to_component = {}

        for cid, comp in enumerate(components):
            for node in comp:
                node_to_component[node] = cid

        table = defaultdict(dict)

        for i, comp_a in enumerate(components):
            for j, comp_b in enumerate(components):
                if i >= j:
                    continue

                best = (float("inf"), None, None)

                for a in comp_a:
                    for b in comp_b:
                        d = nx.shortest_path_length(
                            self.graph, a, b, weight="weight"
                        )
                        if d < best[0]:
                            best = (d, a, b)

                walk_speed = self.config.walk_speed
                best_distance = best[0] / walk_speed  * 60 # in minutes

                table[i][j] = {
                    "distance": best_distance,
                    "from": best[1],
                    "to": best[2],
                }

                table[j][i] = {
                    "distance": best_distance,
                    "from": best[2],
                    "to": best[1],
                }
        
        component_graph = nx.Graph()
        for a in table:
            for b, data in table[a].items():
                component_graph.add_edge(a, b, weight=data["distance"])

        self.component_distance_table = {
            "node_to_component": node_to_component,
            "table": table,
            "graph": component_graph,
        }

    def route_within_component(self, start, end):
        """
        Returns:
        path: [hex_id, ...]
        annotated_steps: [(hex_id, route_name), ...]
        """
        # if the combination already exists in the path_lookup
        if start in self.path_lookup and end in self.path_lookup[start]:
            #print("Cached path is actually working")
            #print("---", self.path_lookup[start][end])
            return self.path_lookup[start][end]
        
        G = self.fixedroute_graph

        path = nx.shortest_path(G, start, end, weight="weight")

        steps = []
        prev_route = None

        # calculate total minutes
        minutes = 0

        for u, v in zip(path[:-1], path[1:]):
            # consistently pick the edge with the smallest weight if multiple edges exist
            edge_data = G.get_edge_data(u, v)
            if edge_data is None:
                continue
            # edge_data is a dict of edge keys to edge attribute dicts
            min_weight = float('inf')
            min_edge = None
            for key, attr in edge_data.items():
                if attr.get("weight", float('inf')) < min_weight:
                    min_weight = attr["weight"]
                    min_edge = attr

            route = min_edge["route"]

            if route != prev_route:
                steps.append((u, route))
                prev_route = route

            minutes += min_weight
        self.path_lookup[start][end] = ([steps], minutes)
        return ([steps], minutes)
    

    def route_across_components_shortest_k(self, start_hex_id, end_hex_id, k=2):
        """
        Returns:
        single best [(hex_id, route_name), ...] from among k shortest paths across components (encased in a list)
        multiple paths [[(hex_id, route_name), ...], ...] if multiple found
        """
        node_to_comp = self.component_distance_table["node_to_component"]

        s = start_hex_id
        e = end_hex_id

        cs = node_to_comp[s]
        ce = node_to_comp[e]
        # if both in the same component
        if cs == ce:
            steps_list, _ = self.route_within_component(s, e)
            #print("---------------", steps_list)
            steps = steps_list[0].copy()
            steps.append((e, None))
            return [steps]
        
        # otherwise if the combination already exists in the path_lookup
        if s in self.path_lookup and e in self.path_lookup[s]:
            #print("Cached path is actually working level 2")
            return self.path_lookup[s][e][0]
        
        best_minutes = float("inf")

        CG = self.component_distance_table["graph"]
        comp_paths = list(
            nx.shortest_simple_paths(CG, cs, ce, weight="weight")
        )[:k]

        all_chains = []

        for comp_path in comp_paths:
            full_steps = []

            for c_from, c_to in zip(comp_path[:-1], comp_path[1:]):
                path_minutes = 0
                bridge = self.component_distance_table["table"][c_from][c_to]

                a = bridge["from"]
                b = bridge["to"]
                walk_minutes = bridge["distance"]

                # Route inside component c_from
                steps_a_list, minutes = self.route_within_component(s, a)
                steps_a = steps_a_list[0]
                full_steps.extend(steps_a)

                path_minutes += minutes
                # Walking transfer
                full_steps.append((a, "WALK"))
                #full_steps.append((b, "WALK"))

                s = b  # continue from next component
                path_minutes += walk_minutes

            # Final component
            steps_end_list, minutes_end = self.route_within_component(s, e)
            steps_end = steps_end_list[0]
            full_steps.extend(steps_end)
            full_steps.append((e, None))

            if path_minutes + minutes_end < best_minutes:
                best_minutes = path_minutes + minutes_end
                all_chains = [full_steps]
            elif path_minutes + minutes_end == best_minutes:
                all_chains.append(full_steps)

        self.path_lookup[start_hex_id][end_hex_id] = (all_chains, best_minutes)

        return all_chains
    

    def build_route_for_chain(self, chain, start, end, demand_time, demand, graph, walk_speed):
        """
        build_route_for_chain using the chain obtained by the new logic obtained from route_across_components
        start: Hex
        end: Hex
        chain: [(hex_id, route_name), ...]
        Returns: Route or raises Exception if something invalid.
        """
        # start and end SHOULD be already on the fixed route graph
        actions = []
        current_time = demand_time
        num_transfers = 0
        # chain input example:
        # [
            #(start_ride_hex_id, "Route_A"),
            #(hex_17, "Route_B"),      # transfer
            #(hex_42, "WALK"),      # end of component
            #(hex_88, "Route_C"),
            #(end_ride_hex_id, None),
        # ]

        # Step 1: walk from origin to the start service’s nearest stop
        s1_start, s_first_route = chain[0]
        s1_walk_time = self.compute_walk_time(graph, start.hex_id, s1_start, walk_speed)[0]
        if s1_start is None:
            raise Exception("No feasible access stop for first service")

        walk_to_start = Walk(
            start_time=current_time,
            end_time=current_time + timedelta(hours=s1_walk_time),
            start_hex=start,
            end_hex=Hex(s1_start),
            walk_speed=walk_speed,
            unit=demand.unit,
            graph=self.graph,
        )
        actions.append(walk_to_start)
        current_time = walk_to_start.end_time

        prev_hex_id = s1_start

        # Traverse the chain
        for i in range(len(chain) - 1):
            
            curr_hex_id, curr_route = chain[i]
            next_hex_id, next_route = chain[i + 1]

            if curr_route == "WALK":
                # Walking transfer
                walk_time, _ = self.compute_walk_time(graph, curr_hex_id, next_hex_id, walk_speed)
                transfer_walk = Walk(
                    start_time=current_time,
                    end_time=current_time + timedelta(hours=walk_time),
                    start_hex=Hex(curr_hex_id),
                    end_hex=Hex(next_hex_id),
                    walk_speed=walk_speed,
                    unit=demand.unit,
                    graph=self.graph,
                )
                actions.append(transfer_walk)
                current_time = transfer_walk.end_time
                num_transfers += 1
            else:
                # Ride on service
                service = self.fixedroute_lookup.get(curr_route, None)
                if service is None:
                    raise Exception(f"Service {curr_route} not found in network.")

                wait_ride = service.get_route(
                    unit=demand.unit,
                    start_time=current_time,
                    start_hex=Hex(curr_hex_id),
                    end_hex=Hex(next_hex_id),
                )
                actions.extend(wait_ride)
                current_time = wait_ride[-1].end_time  # end_time of the last action in wait_ride
                prev_hex_id = next_hex_id
                num_transfers += 1
                
                
        num_transfers -= 1  # Adjust for final walk not being a transfer
        # Step 3: final walk to destination
        final_walk_time, _ = self.compute_walk_time(graph, prev_hex_id, end.hex_id, walk_speed)
        final_walk = Walk(
            start_time=current_time,
            end_time=current_time + timedelta(hours=final_walk_time),
            start_hex=Hex(prev_hex_id),
            end_hex=end,
            walk_speed=walk_speed,
            unit=demand.unit,
            graph=self.graph,
        )
        actions.append(final_walk)
        current_time = final_walk.end_time

        return Route(unit=demand.unit, actions=actions, transfers=num_transfers, config=self.config)
    


    def get_optimal_route(self, demand, second_try=False):
        """
        Get the optimal route for a given demand using shortest path algorithm.
        Searches through all fixed route services and finds the fastest route with at most one transfer.

        Args:
            demand: A Demand object representing the travel request.
            second_try: A boolean indicating if this is a second attempt to find a route after a prior attempt failed (e.g. discovering no available vehicles after arriving at a dock). (For now) mainly used to indicate that the user isn't willing to take docked on-demand services anymore.

        Returns:
            Route: The optimal route to fulfill the demand.
        """

        walk_speed = self.config.walk_speed
        # Find shortest path using NetworkX
        start = demand.start_hex
        end = demand.end_hex
        demand_time = demand.time

        walk_best_route = None
        walk_best_cost = -float("inf")
        walk_fixed_best_route = None
        walk_fixed_best_cost = -float("inf")
        # ondemand_fixed_best_route = None
        # ondemand_fixed_best_cost = float('inf')
        ondemanddocked_best_route = None
        ondemanddockless_best_route = None
        ondemanddocked_best_cost = -float("inf")
        ondemanddockless_best_cost = -float("inf")

        # 1. Try direct walk
        walk_time, walk_path = self.compute_walk_time(
            self.graph, start.hex_id, end.hex_id, walk_speed
        )
        if walk_time < float("inf"):
            walk_action = Walk(
                start_time=demand_time,
                start_hex=start,
                end_hex=end,
                unit=demand.unit,
                graph=self.graph,
                walk_speed=walk_speed,
                end_time=demand_time + timedelta(hours=walk_time),
            )

            walk_route = Route(
                unit=demand.unit, actions=[walk_action], transfers=0, config=self.config
            )
        else:
            walk_route = None

        walk_best_route = walk_route
        walk_best_cost = walk_route.total_cost if walk_route else -float("inf")  # total_cost is utility (higher is better)

        # Get all different route services
        fixed_services = [s for s in self.services if isinstance(s, FixedRouteService)]
        ondemandservices_docked = [
            s for s in self.services if isinstance(s, OnDemandRouteServiceDocked)
        ]
        ondemandservices_dockless = [
            s for s in self.services if isinstance(s, OnDemandRouteServiceDockless)
        ]

        # 2. FixedRouteService options
        
        try:
            start_stop, start_walk_time = self.find_closest_stop(
                self.graph, start.hex_id, walk_speed
            )
            end_stop, end_walk_time = self.find_closest_stop(
                self.graph, end.hex_id, walk_speed
            )

            chains = self.route_across_components_shortest_k(start_stop, end_stop, k=2)
            #print("Trying chains:", chains)
            for chain in chains:
                try:
                    fixedroute_chain_route = self.build_route_for_chain(
                        chain, start, end, demand_time, demand, self.graph, walk_speed
                    )
                    if fixedroute_chain_route.total_cost > walk_fixed_best_cost:  # total_cost is utility (higher is better)
                        walk_fixed_best_cost = fixedroute_chain_route.total_cost
                        walk_fixed_best_route = fixedroute_chain_route
                except Exception: # Any case where the chain returned is a "false" positive e.g. no available vehicles on the route is dealt with here
                    pass
                    traceback.print_exc()
                    raise
        except Exception:
            pass
            traceback.print_exc()
            raise

        # 3. OnDemandService options

        for service in ondemandservices_docked:
            try:
                best_start_dock, vehicle_walk_time = self.find_closest_ondemand_dock(
                    self.graph, start.hex_id, service, walk_speed, radius=2
                )
                if best_start_dock is None:  # aka literally no docks at all within the radius
                    continue
                best_end_dock, off_vehicle_walk_time = self.find_closest_ondemand_dock(
                    self.graph, end.hex_id, service, walk_speed, radius=2
                )
                if best_end_dock is None:  # No dock at destination
                    continue
                # Walk to vehicle
                walk_to_vehicle = Walk(
                    start_time=demand_time,
                    end_time=demand_time + timedelta(hours=vehicle_walk_time),
                    start_hex=start,
                    end_hex=best_start_dock.location,
                    unit=demand.unit,
                    graph=self.graph,
                    walk_speed=walk_speed,
                )
                vehicle = best_start_dock.take_vehicle()
                # below case for when there IS an available vehicle
                # Ride with vehicle
                drive_time = service.compute_drive_time(best_start_dock.location, end)
                arrival_time = walk_to_vehicle.end_time + drive_time
                ride_action = OnDemandRide(
                    start_time=walk_to_vehicle.end_time,
                    end_time=arrival_time,
                    start_hex=best_start_dock.location,
                    end_hex=best_end_dock.location,
                    unit=demand.unit,
                    service=service,
                    vehicle=vehicle,
                )
                walk_from_vehicle = Walk(
                    start_time=ride_action.end_time,
                    end_time=ride_action.end_time + timedelta(hours=off_vehicle_walk_time),
                    start_hex=best_end_dock.location,
                    end_hex=end,
                    unit=demand.unit,
                    graph=self.graph,
                    walk_speed=walk_speed,
                )
                ondemand_route = Route(
                    unit=demand.unit,
                    actions=[walk_to_vehicle, ride_action, walk_from_vehicle],
                    transfers=0,
                    config=self.config,
                )
                if ondemand_route.total_cost > ondemanddocked_best_cost:  # total_cost is utility (higher is better)
                    ondemanddocked_best_cost = ondemand_route.total_cost
                    ondemanddocked_best_route = ondemand_route
            except Exception:
                continue

        for service in ondemandservices_dockless:
            best_vehicle, vehicle_walk_time = self.find_closest_ondemand_vehicle(
                self.graph, start.hex_id, service, walk_speed, demand_time, radius=2
            )
            if best_vehicle is None:  # aka literally no vehicles at all within the radius
                continue
            try:
                # Walk to vehicle
                walk_to_vehicle = Walk(
                    start_time=demand_time,
                    end_time=demand_time + timedelta(hours=vehicle_walk_time),
                    start_hex=start,
                    end_hex=best_vehicle.current_location,
                    unit=demand.unit,
                    graph=self.graph,
                    walk_speed=walk_speed,
                )
                # Ride with vehicle
                drive_time = service.compute_drive_time(best_vehicle.current_location, end)
                arrival_time = walk_to_vehicle.end_time + drive_time
                ride_action = OnDemandRide(
                    start_time=walk_to_vehicle.end_time,
                    end_time=arrival_time,
                    start_hex=best_vehicle.current_location,
                    end_hex=end,
                    unit=demand.unit,
                    service=service,
                    vehicle=best_vehicle,
                )
                ondemand_route = Route(
                    unit=demand.unit,
                    actions=[walk_to_vehicle, ride_action],
                    transfers=0,
                    config=self.config,
                )
                if ondemand_route.total_cost > ondemanddockless_best_cost:  # total_cost is utility (higher is better)
                    ondemanddockless_best_cost = ondemand_route.total_cost
                    ondemanddockless_best_route = ondemand_route
            except Exception:
                continue


        # After evaluating all options, determine route probabilities via softmax
        # foolproofing for None routes
        choices = [walk_best_route]
        logits = [walk_best_cost]
        #print(walk_best_cost, walk_fixed_best_cost, ondemanddocked_best_cost, ondemanddockless_best_cost)
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
            if (
                isinstance(action, Ride) and isinstance(action.service, OnDemandRouteServiceDocked)
            ) or (
                isinstance(action, Ride)
                and isinstance(action.service, OnDemandRouteServiceDockless)
            ):
                # Find the vehicle that is closest to the start_hex at start_time. Should reasonably be the same one used in the Ride action.
                vehicle = action.vehicle
                if vehicle is not None:
                    # Update vehicle's location
                    vehicle.current_location = action.end_hex
                    # simulate being unavailable until the end of the action
                    vehicle.available_time = action.end_time

        return choice

    def push_route(self, route):
        """
        Add a route to the network's taken routes.

        Args:
            route: A Route object to add to the network.
        """
        if route is not None:
            self.routes_taken.append(route)

    def clear_routes(self):
        """
        Clear all taken routes from the network.
        """
        self.routes_taken.clear()

    def __repr__(self):
        return f"Network(graph_nodes={len(self.graph.nodes())}, graph_edges={len(self.graph.edges())}, routes_taken={len(self.routes_taken)})"
