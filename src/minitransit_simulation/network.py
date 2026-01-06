"""
Network class representing the transportation network of a city.
"""

from dataclasses import dataclass
from datetime import timedelta

import networkx as nx
import numpy as np

from .primitives.route import RouteConfig

from .actions.ride import Ride
from .actions.walk import Walk
from .services.fixedroute import FixedRouteService
from .services.ondemand import *

# Dynamic imports to avoid circular import issues
try:
    from .primitives.route import Route
except ImportError:
    from .primitives.route import Route

# TODO: actually find a way to input this to be customizable

MAX_INTERMEDIARIES = 1

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

    def __init__(self, geojson_file_path: str, config = NetworkConfig()):
        """
        Initialize a Network object.

        Args:
            geojson_file_path (str): Path to the GeoJSON file for the city.
        """
        self.config = config
        self.graph = construct_graph(geojson_file_path)
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

    def find_closest_ondemand_vehicle(
        self, graph, hex_id, service, walk_speed, demand_time, radius=2
    ):
        import math

        best_vehicle = None
        best_time = float("inf")
        vehicle_within_radius_count = 0
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

        walk_speed = self.config.walk_speed

        # 1. Shared stops → zero walking transfer
        shared = set(sa.stops) & set(sb.stops)
        if shared:
            return {"ok": True, "transfer_pairs": [(x, x, 0) for x in shared]}

        # 2. Otherwise walkable transfer between ANY stop pair (in THIS case we would want the one with the least amount of walking time)
        best = []
        inf = float("inf")
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
            return {"ok": True, "transfer_pairs": best}

        return {"ok": False, "transfer_pairs": []}

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
            unit=demand.unit,
        )
        actions.append(walk_to_start)
        current_time = walk_to_start.end_time

        # Traverse service pairs
        for i in range(len(chain) - 1):
            sa = chain[i]
            sb = chain[i + 1]

            link = self.services_can_link(sa, sb, walk_speed)
            _, _, walk_time_hours = link["transfer_pairs"][
                0
            ]  # should either be the pair with the best walking time or one that doesn't need walking at all
            if walk_time_hours != 0:
                num_transfers += 1  # extra transfer for walking transfer
            # at this point we SHOULD'VE already checked that they can link in the find_service_chains step
            if not link["ok"]:
                raise Exception("Invalid link in chain")

            # choose the *first* feasible pair for simplicity
            (a_stop, b_stop, walk_t) = link["transfer_pairs"][0]

            # ride on service sa from its nearest start to a_stop
            wait_sa, ride_sa = sa.get_route(
                unit=demand.unit,
                start_time=current_time,
                start_hex=s1_start if i == 0 else prev_b,
                end_hex=a_stop,
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
                    unit=demand.unit,
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
            unit=demand.unit, start_time=current_time, start_hex=prev_b, end_hex=s2_end
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
            unit=demand.unit,
        )
        actions.append(final_walk)

        # TODO: calculate transfers properly in the form of including transfers BETWEEN walking and ride actions etc
        # still leaving todo here in case i'm missing something
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
        walk_time, walk_path = self.compute_walk_time(self.graph, start, end, walk_speed)
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

            walk_route = Route(unit=demand.unit, actions=[walk_action], transfers=0, config=self.config)
        else:
            walk_route = None

        walk_best_route = walk_route
        walk_best_cost = walk_route.total_cost if walk_route else float("inf")

        # Get all different route services
        fixed_services = [s for s in self.services if isinstance(s, FixedRouteService)]
        ondemandservices_docked = [
            s for s in self.services if isinstance(s, OnDemandRouteServiceDocked)
        ]
        ondemandservices_dockless = [
            s for s in self.services if isinstance(s, OnDemandRouteServiceDockless)
        ]

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
                        start,
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
                        end,
                        unit=demand.unit,
                        graph=self.graph,
                        walk_speed=walk_speed,
                    )

                    route = Route(unit=demand.unit, actions=[walk1, wait, ride, walk2], transfers=0, config=self.config)

                    if route.total_cost < walk_fixed_best_cost:
                        walk_fixed_best_cost = route.total_cost
                        walk_fixed_best_route = route
                except Exception:
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

                    if common_stops:
                        # Try each common stop as transfer point
                        for transfer_stop in common_stops:
                            # Try the transfer route - get_route will find the appropriate vehicle/direction
                            # We just need to ensure we're not trying to go from a stop to itself
                            if start_stop1 == transfer_stop or transfer_stop == end_stop2:
                                continue
                            try:
                                # Walk to first service stop
                                walk1 = Walk(
                                    demand_time,
                                    demand_time + timedelta(hours=start_walk_time),
                                    start,
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
                                    end,
                                    unit=demand.unit,
                                    graph=self.graph,
                                    walk_speed=walk_speed,
                                )

                                route = Route(
                                    unit=demand.unit,
                                    actions=[walk1, wait1, ride1, wait2, ride2, walk2],
                                    transfers=1,
                                    config=self.config
                                )

                                if route.total_cost < walk_fixed_best_cost:
                                    walk_fixed_best_cost = route.total_cost
                                    walk_fixed_best_route = route
                            except Exception:
                                # Service route not available at this time, skip
                                continue
                    else:
                        chains = self.find_service_chains(
                            service1, service2, fixed_services, MAX_INTERMEDIARIES
                        )

                        for chain in chains:
                            try:
                                route = self.build_route_for_chain(
                                    chain=chain,
                                    start=start,
                                    end=end,
                                    demand_time=demand_time,
                                    demand=demand,
                                    graph=self.graph,
                                    walk_speed=walk_speed,
                                )

                                if route.total_cost < walk_fixed_best_cost:
                                    walk_fixed_best_cost = route.total_cost
                                    walk_fixed_best_route = route

                            except Exception:
                                continue
                except Exception:
                    continue

        if not second_try:
            for service in ondemandservices_docked:
                try:
                    best_start_dock, vehicle_walk_time = self.find_closest_ondemand_dock(
                        self.graph, start, service, walk_speed, radius=2
                    )
                    if best_start_dock is None:  # aka literally no docks at all within the radius
                        continue
                    best_end_dock, off_vehicle_walk_time = self.find_closest_ondemand_dock(
                        self.graph, end, service, walk_speed, radius=2
                    )
                    # Walk to vehicle
                    walk_to_vehicle = Walk(
                        start_time=demand_time,
                        end_time=demand_time + timedelta(hours=vehicle_walk_time),
                        start_hex=start,
                        end_hex=best_start_dock.location,
                        unit=demand.unit,
                        walk_speed=walk_speed,
                    )
                    vehicle, _ = best_start_dock.take_vehicle()
                    if vehicle is None:  # no available vehicle at the dock
                        # try to find another route without docked OnDemand service
                        demand.time = walk_to_vehicle.end_time
                        return self.get_optimal_route(demand, second_try=True)
                    # below case for when there IS an available vehicle
                    # Ride with vehicle
                    drive_time = service.compute_drive_time(best_start_dock.location, end)
                    arrival_time = walk_to_vehicle.end_time + drive_time
                    ride_action = Ride(
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
                        walk_speed=walk_speed,
                    )
                    ondemand_route = Route(
                        unit=demand.unit,
                        actions=[walk_to_vehicle, ride_action, walk_from_vehicle],
                        transfers=0,
                        config=self.config
                    )
                    if ondemand_route.total_cost < ondemanddocked_best_cost:
                        ondemanddocked_best_cost = ondemand_route.total_cost
                        ondemanddocked_best_route = ondemand_route
                except Exception:
                    continue

        for service in ondemandservices_dockless:
            best_vehicle, vehicle_walk_time = self.find_closest_ondemand_vehicle(
                self.graph, start, service, walk_speed, demand_time, radius=2
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
                    walk_speed=walk_speed,
                )
                # Ride with vehicle
                drive_time = service.compute_drive_time(best_vehicle.current_location, end)
                arrival_time = walk_to_vehicle.end_time + drive_time
                ride_action = Ride(
                    start_time=walk_to_vehicle.end_time,
                    end_time=arrival_time,
                    start_hex=best_vehicle.current_location,
                    end_hex=end,
                    unit=demand.unit,
                    service=service,
                    vehicle=best_vehicle,
                )
                ondemand_route = Route(
                    unit=demand.unit, actions=[walk_to_vehicle, ride_action], transfers=0, config=self.config
                )
                if ondemand_route.total_cost < ondemanddockless_best_cost:
                    ondemanddockless_best_cost = ondemand_route.total_cost
                    ondemanddockless_best_route = ondemand_route
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
