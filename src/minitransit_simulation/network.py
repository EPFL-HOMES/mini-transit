"""
Network class representing the transportation network of a city.
"""

import math
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import networkx as nx
import numpy as np

from .actions.ondemand_ride import OnDemandRide
from .actions.ride import Ride
from .actions.walk import Walk
from .primitives.route import RouteConfig
from .services.fixedroute import FixedRouteService
from .services.ondemand import *

try:
    from .primitives.route import Route
except ImportError:
    from .primitives.route import Route

from .graph import construct_graph


@dataclass
class NetworkConfig(RouteConfig):
    walk_speed: float = 15.0  # hexagons per hour
    bike_speed: float = 35.0  # hexagons per hour


class Network:
    """
    Represents the transportation network of a city.
    """

    def __init__(self, geojson_file_path: str, config=NetworkConfig()):
        self.config = config
        self.graph = construct_graph(geojson_file_path)
        self.fixedroute_graph = None
        self.services = []
        self.routes_taken = []
        self.fixedroute_lookup = {}
        self.component_distance_table = None
        self.path_lookup = defaultdict(dict)
        self.closest_stop_lookup = {}

        # --- PERFORMANCE OPTIMIZATION: CACHING ---
        # Cache for shortest path lengths to avoid redundant Dijkstra runs
        self._distance_cache = {}
        self._walk_time_path_cache = {}

    # =========================================================================
    # BULLETPROOF HELPERS
    # =========================================================================
    def _safe_id(self, obj):
        """Safely extract hex_id whether the object is an int, str, or a Hex object."""
        return getattr(obj, "hex_id", obj)

    def _safe_hex(self, obj):
        """Safely ensure the output is a Hex wrapper for actions."""
        if isinstance(obj, (int, float, str)):
            return Hex(obj)
        return obj

    def _get_available_bike(self, dock, service, demand_time):
        """Safely find an available bike whether mapped directly to dock or listed globally."""
        dock_id = self._safe_id(getattr(dock, "location", dock))

        if hasattr(dock, "vehicles") and dock.vehicles:
            for v in dock.vehicles:
                if getattr(v, "is_available", lambda t: True)(demand_time):
                    return v

        if hasattr(service, "vehicles"):
            for v in service.vehicles:
                v_loc = self._safe_id(
                    getattr(v, "current_location", getattr(v, "initial_location", None))
                )
                if v_loc == dock_id and getattr(v, "is_available", lambda t: True)(demand_time):
                    return v
        return None

    # =========================================================================
    # HIGH-PERFORMANCE GRAPH & DISTANCE UTILITIES
    # =========================================================================
    def get_cached_shortest_path_length(self, source_id, target_id, weight=None):
        """Retrieves shortest path length from cache or computes it if not found."""
        cache_key = (source_id, target_id, weight)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        try:
            dist = nx.shortest_path_length(self.graph, source_id, target_id, weight=weight)
        except nx.NetworkXNoPath:
            dist = float("inf")
        self._distance_cache[cache_key] = dist
        return dist

    def get_distance(self, start_hex, end_hex) -> int:
        return self.get_cached_shortest_path_length(
            self._safe_id(start_hex), self._safe_id(end_hex)
        )

    def get_walk_shortest_path(self, start_hex, end_hex):
        # We also cache the actual path sequence since it's frequently requested
        cache_key = (self._safe_id(start_hex), self._safe_id(end_hex))
        if cache_key in self._walk_time_path_cache:
            return self._walk_time_path_cache[cache_key][1]
        try:
            path = nx.shortest_path(self.graph, cache_key[0], cache_key[1])
            # Compute distance by summing edge weights along the returned path so
            # we avoid a second full Dijkstra via shortest_path_length.
            distance = sum(
                self.graph[u][v].get("length", 1)
                for u, v in zip(path[:-1], path[1:])
            )
            self._walk_time_path_cache[cache_key] = (distance, path)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._walk_time_path_cache[cache_key] = (float("inf"), None)
            return None

    def compute_walk_time(self, graph, from_hex, to_hex, walk_speed):
        source_id = self._safe_id(from_hex)
        target_id = self._safe_id(to_hex)
        cache_key = (source_id, target_id)

        if cache_key in self._walk_time_path_cache:
            dist, path = self._walk_time_path_cache[cache_key]
            return dist / walk_speed if dist != float("inf") else float("inf"), path

        try:
            distance = nx.shortest_path_length(
                graph, source=source_id, target=target_id, weight="length"
            )
            path = nx.shortest_path(graph, source=source_id, target=target_id)
            self._walk_time_path_cache[cache_key] = (distance, path)
            return distance / walk_speed, path
        except nx.NetworkXNoPath:
            self._walk_time_path_cache[cache_key] = (float("inf"), None)
            return float("inf"), None

    def find_closest_stop(self, graph, hex_id, walk_speed):
        h_id = self._safe_id(hex_id)
        if hasattr(self, "closest_stop_lookup") and h_id in self.closest_stop_lookup:
            best_stop, distance = self.closest_stop_lookup[h_id]
            if best_stop is None or distance == float("inf"):
                return None, float("inf")
            return best_stop, distance / walk_speed
        return None, float("inf")

    def find_closest_ondemand_vehicle(
        self, graph, hex_id, service, walk_speed, demand_time, radius=3
    ):
        best_vehicle = None
        best_time = float("inf")
        vehicle_within_radius_count = 0
        walk_time_metric = float("inf")
        source_id = self._safe_id(hex_id)

        # PERFORMANCE FIX: Use single_source_shortest_path_length with cutoff!
        # This explores ONLY the immediate 3-grid radius instead of checking the whole city.
        try:
            nearby_hexes = nx.single_source_shortest_path_length(
                graph, source=source_id, cutoff=radius
            )
        except nx.NetworkXNoPath:
            return None, float("inf")

        for vehicle in service.vehicles:
            target_id = self._safe_id(vehicle.current_location)
            # Only proceed if the vehicle's hex is in our pre-filtered nearby radius
            if target_id in nearby_hexes:
                if not vehicle.is_available(demand_time):
                    continue
                vehicle_within_radius_count += 1
                walk_time, _ = self.compute_walk_time(graph, source_id, target_id, walk_speed)
                # CORRECTED: Standard formula for the number of hexagons within a radius R (center + R rings)
                area = 1 + 3 * radius * (radius + 1)
                if walk_time < best_time:
                    best_time = walk_time
                    best_vehicle = vehicle
                walk_time_metric = (1 / (2 * walk_speed)) * math.sqrt(
                    area / vehicle_within_radius_count
                )

        return best_vehicle, walk_time_metric

    def find_closest_ondemand_dock(self, graph, hex_id, service, walk_speed, radius=3):
        best_dock = None
        best_time = float("inf")
        source_id = self._safe_id(hex_id)

        # PERFORMANCE FIX: Bounded BFS cutoff
        try:
            nearby_hexes = nx.single_source_shortest_path_length(
                graph, source=source_id, cutoff=radius
            )
        except nx.NetworkXNoPath:
            return None, float("inf")

        for dock in service.docking_stations:
            target_id = self._safe_id(dock.location)
            if target_id in nearby_hexes:
                walk_time, _ = self.compute_walk_time(graph, source_id, target_id, walk_speed)
                if walk_time < best_time:
                    best_time = walk_time
                    best_dock = dock
        return best_dock, best_time

    # =========================================================================
    # MULTIMODAL ACCESS LOGIC (BIKE + BUS)
    # =========================================================================
    def get_best_bike_to_bus_access(
        self, graph, start_hex, walk_speed, bike_speed, service, demand_time, radius=3
    ):
        best_time = float("inf")
        best_res = (None, float("inf"), None, None)
        start_docks = []
        source_id = self._safe_id(start_hex)

        # PERFORMANCE FIX: Bounded BFS cutoff
        try:
            nearby_hexes = nx.single_source_shortest_path_length(
                graph, source=source_id, cutoff=radius
            )
        except nx.NetworkXNoPath:
            return best_res

        for dock in service.docking_stations:
            dock_id = self._safe_id(dock.location)
            if dock_id in nearby_hexes:
                dist = nearby_hexes[dock_id]
                if self._get_available_bike(dock, service, demand_time) is not None:
                    start_docks.append((dock, dist))

        for dock_s, dist_s in start_docks:
            walk_time_s = dist_s / walk_speed
            dock_s_id = self._safe_id(dock_s.location)
            for dock_d in service.docking_stations:
                dock_d_id = self._safe_id(dock_d.location)
                if dock_s_id == dock_d_id:
                    continue

                capacity = getattr(dock_d, "capacity", 100)
                dock_vehicles = getattr(dock_d, "current_vehicles", [])
                if len(dock_vehicles) >= capacity:
                    continue

                b_stop, b_dist = self.closest_stop_lookup.get(dock_d_id, (None, float("inf")))
                if b_stop is None:
                    continue

                walk_time_b = b_dist / walk_speed

                # PERFORMANCE FIX: Use cached distance!
                bike_dist = self.get_cached_shortest_path_length(dock_s_id, dock_d_id)
                if bike_dist == float("inf"):
                    continue

                bike_time = bike_dist / bike_speed
                total_time = walk_time_s + bike_time + walk_time_b
                if total_time < best_time:
                    best_time = total_time
                    best_res = (b_stop, total_time, dock_s, dock_d)

        return best_res

    def get_best_bus_to_bike_egress(
        self, graph, end_hex, walk_speed, bike_speed, service, demand_time, radius=3
    ):
        best_time = float("inf")
        best_res = (None, float("inf"), None, None)
        end_docks = []
        target_id = self._safe_id(end_hex)

        # PERFORMANCE FIX: Bounded BFS cutoff
        try:
            # Note: We compute distance *from* the dock *to* the target, but since undirected grid, reverse is fine
            nearby_hexes = nx.single_source_shortest_path_length(
                graph, source=target_id, cutoff=radius
            )
        except nx.NetworkXNoPath:
            return best_res

        for dock in service.docking_stations:
            dock_id = self._safe_id(dock.location)
            if dock_id in nearby_hexes:
                dist = nearby_hexes[dock_id]
                capacity = getattr(dock, "capacity", 100)
                dock_vehicles = getattr(dock, "vehicles", [])
                if len(dock_vehicles) < capacity:
                    end_docks.append((dock, dist))

        for dock_e, dist_e in end_docks:
            walk_time_e = dist_e / walk_speed
            dock_e_id = self._safe_id(dock_e.location)
            for dock_p in service.docking_stations:
                dock_p_id = self._safe_id(dock_p.location)
                if dock_e_id == dock_p_id:
                    continue
                if self._get_available_bike(dock_p, service, demand_time) is None:
                    continue

                b_stop, b_dist = self.closest_stop_lookup.get(dock_p_id, (None, float("inf")))
                if b_stop is None:
                    continue

                walk_time_b = b_dist / walk_speed

                # PERFORMANCE FIX: Use cached distance!
                bike_dist = self.get_cached_shortest_path_length(dock_p_id, dock_e_id)
                if bike_dist == float("inf"):
                    continue

                bike_time = bike_dist / bike_speed
                total_time = walk_time_b + bike_time + walk_time_e
                if total_time < best_time:
                    best_time = total_time
                    best_res = (b_stop, total_time, dock_p, dock_e)

        return best_res

    # =========================================================================
    # PUBLIC TRANSIT ROUTING LOGIC
    # =========================================================================
    def build_fixedroute_graph(self, fixed_routes):
        G = nx.MultiDiGraph()
        for route in fixed_routes:
            if route.name in self.fixedroute_lookup:
                print(f"Warning: Duplicate route name detected: {route.name}")
            self.fixedroute_lookup[route.name] = route
            vehicles = route.vehicles[:2] if route.bidirectional else route.vehicles[:1]
            for vehicle in vehicles:
                timetable = vehicle.timetable
                stop_indices = list(timetable.keys())
                for i in range(len(stop_indices) - 1):
                    a_id = self._safe_id(route.stops[stop_indices[i]])
                    b_id = self._safe_id(route.stops[stop_indices[i + 1]])
                    dep_time = timetable[stop_indices[i]][1]
                    arr_time = timetable[stop_indices[i + 1]][0]
                    minutes = (arr_time - dep_time).total_seconds() / 60
                    G.add_edge(a_id, b_id, weight=minutes, route=route.name)
        self.fixedroute_graph = G
        self.build_closest_stops_table()

    def build_closest_stops_table(self):
        self.closest_stop_lookup = {}
        if not self.fixedroute_graph or len(self.fixedroute_graph.nodes) == 0:
            return
        all_stops = set(self.fixedroute_graph.nodes)
        try:
            distances, paths = nx.multi_source_dijkstra(
                self.graph, sources=all_stops, weight="length"
            )
            for node, dist in distances.items():
                self.closest_stop_lookup[node] = (paths[node][0], dist)
        except Exception:
            pass
        for node in self.graph.nodes:
            if node not in self.closest_stop_lookup:
                self.closest_stop_lookup[node] = (None, float("inf"))

    def build_component_distance_table(self):
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
                        d = nx.shortest_path_length(self.graph, a, b, weight="weight")
                        if d < best[0]:
                            best = (d, a, b)
                best_distance = best[0] / self.config.walk_speed * 60
                table[i][j] = {"distance": best_distance, "from": best[1], "to": best[2]}
                table[j][i] = {"distance": best_distance, "from": best[2], "to": best[1]}
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
        if start in self.path_lookup and end in self.path_lookup[start]:
            return self.path_lookup[start][end]
        G = self.fixedroute_graph
        path = nx.shortest_path(G, start, end, weight="weight")
        steps, prev_route, minutes = [], None, 0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data is None:
                continue
            min_edge = min(edge_data.values(), key=lambda attr: attr.get("weight", float("inf")))
            route = min_edge["route"]
            if route != prev_route:
                steps.append((u, route))
                prev_route = route
            minutes += min_edge["weight"]
        self.path_lookup[start][end] = ([steps], minutes)
        return ([steps], minutes)

    def best_bridge_from_node(self, source_node, target_component):
        best = (float("inf"), None)
        for node, comp in self.component_distance_table["node_to_component"].items():
            if comp != target_component:
                continue
            d = nx.shortest_path_length(self.graph, source_node, node, weight="weight")
            if d < best[0]:
                best = (d, node)
        return best

    def route_across_components_shortest_k(self, start_hex_id, end_hex_id, k=2):
        node_to_comp = self.component_distance_table["node_to_component"]
        s, e = self._safe_id(start_hex_id), self._safe_id(end_hex_id)

        if s not in node_to_comp or e not in node_to_comp:
            return []

        cs, ce = node_to_comp[s], node_to_comp[e]
        if cs == ce:
            steps_list, _ = self.route_within_component(s, e)
            steps = steps_list[0].copy()
            steps.append((e, None))
            return [steps]
        if s in self.path_lookup and e in self.path_lookup[s]:
            return self.path_lookup[s][e][0]

        best_minutes = float("inf")
        CG = self.component_distance_table["graph"]
        try:
            comp_paths = list(nx.shortest_simple_paths(CG, cs, ce, weight="weight"))[:k]
        except nx.NetworkXNoPath:
            return []

        all_chains = []
        for comp_path in comp_paths:
            used_fixed_route = False
            s = self._safe_id(start_hex_id)
            direct_walk_start = s
            full_steps, path_minutes = [], 0

            for c_from, c_to in zip(comp_path[:-1], comp_path[1:]):
                bridge = self.component_distance_table["table"][c_from][c_to]
                a = bridge["from"]
                if s != a:
                    full_steps.append((direct_walk_start, "WALK"))
                    direct_walk_start = a
                    steps_a_list, minutes = self.route_within_component(s, a)
                    if minutes > 0:
                        used_fixed_route = True
                    full_steps.extend(steps_a_list[0])
                    path_minutes += minutes
                    full_steps.append((a, "WALK"))
                    walk_minutes = bridge["distance"]
                    b = bridge["to"]
                else:
                    walk_minutes, b = self.best_bridge_from_node(s, c_to)
                    full_steps.append((s, "WALK"))
                path_minutes += walk_minutes
                s = b
            if s != e:
                steps_end_list, minutes_end = self.route_within_component(s, e)
                full_steps.append((direct_walk_start, "WALK"))
                full_steps.extend(steps_end_list[0])
                full_steps.append((e, None))
            else:
                minutes_end = 0
                if not used_fixed_route:
                    continue
            if path_minutes + minutes_end < best_minutes:
                best_minutes = path_minutes + minutes_end
                all_chains = [full_steps]
            elif path_minutes + minutes_end == best_minutes:
                all_chains.append(full_steps)

        self.path_lookup[self._safe_id(start_hex_id)][self._safe_id(end_hex_id)] = (
            all_chains,
            best_minutes,
        )
        return all_chains

    def build_route_for_chain(self, chain, start, end, demand_time, demand, graph, walk_speed):
        bike_speed = getattr(self.config, "bike_speed", walk_speed * 3)
        return self.build_multimodal_chain_route(
            chain,
            start,
            end,
            demand_time,
            demand,
            graph,
            walk_speed,
            bike_speed,
            None,
            "walk",
            "walk",
            None,
            None,
        )

    # =========================================================================
    # MULTIMODAL BUILDER
    # =========================================================================
    def build_multimodal_chain_route(
        self,
        chain,
        start,
        end,
        demand_time,
        demand,
        graph,
        walk_speed,
        bike_speed,
        docked_service,
        acc_mode,
        eg_mode,
        acc_data,
        eg_data,
    ):
        actions = []
        current_time = demand_time
        num_transfers = 0
        s1_start = self._safe_id(chain[0][0])
        start_obj = self._safe_hex(start)
        end_obj = self._safe_hex(end)

        # Access Leg
        if acc_mode == "bike":
            b_stop, _, dock_s, dock_d = acc_data
            dock_s_obj, dock_d_obj = self._safe_hex(dock_s.location), self._safe_hex(
                dock_d.location
            )
            walk_time_s, walk_path_s = self.compute_walk_time(
                graph, start_obj, dock_s_obj, walk_speed
            )
            walk1 = Walk(
                start_time=current_time,
                end_time=current_time + timedelta(hours=walk_time_s),
                start_hex=start_obj,
                end_hex=dock_s_obj,
                walk_speed=walk_speed,
                unit=demand.unit,
                graph=graph,
                walk_path=walk_path_s,
            )
            actions.append(walk1)
            current_time = walk1.end_time

            vehicle = self._get_available_bike(dock_s, docked_service, current_time)
            if not vehicle:
                raise Exception("Bike became unavailable")

            bike_dist = self.get_cached_shortest_path_length(
                self._safe_id(dock_s_obj), self._safe_id(dock_d_obj)
            )
            drive_time = bike_dist / bike_speed
            ride_path_1 = self.get_walk_shortest_path(dock_s_obj, dock_d_obj)
            ride1 = OnDemandRide(
                start_time=current_time,
                end_time=current_time + timedelta(hours=drive_time),
                start_hex=dock_s_obj,
                end_hex=dock_d_obj,
                unit=demand.unit,
                service=docked_service,
                vehicle=vehicle,
                ride_path=ride_path_1,
            )
            actions.append(ride1)
            current_time = ride1.end_time
            num_transfers += 1

            walk_time_b, walk_path_b = self.compute_walk_time(
                graph, dock_d_obj, self._safe_hex(s1_start), walk_speed
            )
            walk2 = Walk(
                start_time=current_time,
                end_time=current_time + timedelta(hours=walk_time_b),
                start_hex=dock_d_obj,
                end_hex=self._safe_hex(s1_start),
                walk_speed=walk_speed,
                unit=demand.unit,
                graph=graph,
                walk_path=walk_path_b,
            )
            actions.append(walk2)
            current_time = walk2.end_time
        else:
            s1_walk_time, walk_path = self.compute_walk_time(
                graph, start_obj, self._safe_hex(s1_start), walk_speed
            )
            if self._safe_id(start_obj) != s1_start:
                walk_to_start = Walk(
                    start_time=current_time,
                    end_time=current_time + timedelta(hours=s1_walk_time),
                    start_hex=start_obj,
                    end_hex=self._safe_hex(s1_start),
                    walk_speed=walk_speed,
                    unit=demand.unit,
                    graph=graph,
                    walk_path=walk_path,
                )
                actions.append(walk_to_start)
                current_time = walk_to_start.end_time

        prev_hex_id = s1_start

        # Main Transit Leg
        for i in range(len(chain) - 1):
            curr_hex_id, curr_route = self._safe_id(chain[i][0]), chain[i][1]
            next_hex_id, next_route = self._safe_id(chain[i + 1][0]), chain[i + 1][1]
            if curr_route == "WALK":
                walk_time, walk_path = self.compute_walk_time(
                    graph, curr_hex_id, next_hex_id, walk_speed
                )
                t_walk = Walk(
                    start_time=current_time,
                    end_time=current_time + timedelta(hours=walk_time),
                    start_hex=self._safe_hex(curr_hex_id),
                    end_hex=self._safe_hex(next_hex_id),
                    walk_speed=walk_speed,
                    unit=demand.unit,
                    graph=graph,
                    walk_path=walk_path,
                )
                actions.append(t_walk)
                current_time = t_walk.end_time
                num_transfers += 1
            else:
                service = self.fixedroute_lookup.get(curr_route, None)
                wait_ride = service.get_route(
                    unit=demand.unit,
                    start_time=current_time,
                    start_hex=self._safe_hex(curr_hex_id),
                    end_hex=self._safe_hex(next_hex_id),
                )
                actions.extend(wait_ride)
                current_time = wait_ride[-1].end_time
                prev_hex_id = next_hex_id
                num_transfers += 1

        if acc_mode == "walk" and eg_mode == "walk":
            num_transfers -= 1

        # Egress Leg
        if eg_mode == "bike":
            b_stop, _, dock_p, dock_e = eg_data
            dock_p_obj, dock_e_obj = self._safe_hex(dock_p.location), self._safe_hex(
                dock_e.location
            )
            walk_time_p, walk_path_p = self.compute_walk_time(
                graph, prev_hex_id, dock_p_obj, walk_speed
            )
            walk3 = Walk(
                start_time=current_time,
                end_time=current_time + timedelta(hours=walk_time_p),
                start_hex=self._safe_hex(prev_hex_id),
                end_hex=dock_p_obj,
                walk_speed=walk_speed,
                unit=demand.unit,
                graph=graph,
                walk_path=walk_path_p,
            )
            actions.append(walk3)
            current_time = walk3.end_time

            vehicle = self._get_available_bike(dock_p, docked_service, current_time)
            if not vehicle:
                raise Exception("Bike became unavailable at egress")
            bike_dist = self.get_cached_shortest_path_length(
                self._safe_id(dock_p_obj), self._safe_id(dock_e_obj)
            )
            drive_time = bike_dist / bike_speed
            _, ride_path_2 = self.compute_walk_time(
                graph, dock_p_obj, dock_e_obj, walk_speed
            )
            ride2 = OnDemandRide(
                start_time=current_time,
                end_time=current_time + timedelta(hours=drive_time),
                start_hex=dock_p_obj,
                end_hex=dock_e_obj,
                unit=demand.unit,
                service=docked_service,
                vehicle=vehicle,
                ride_path=ride_path_2,
            )
            actions.append(ride2)
            current_time = ride2.end_time
            num_transfers += 1

            walk_time_e, walk_path_e = self.compute_walk_time(
                graph, dock_e_obj, end_obj, walk_speed
            )
            walk4 = Walk(
                start_time=current_time,
                end_time=current_time + timedelta(hours=walk_time_e),
                start_hex=dock_e_obj,
                end_hex=end_obj,
                walk_speed=walk_speed,
                unit=demand.unit,
                graph=graph,
                walk_path=walk_path_e,
            )
            actions.append(walk4)
        else:
            final_walk_time, walk_path = self.compute_walk_time(
                graph, prev_hex_id, end_obj, walk_speed
            )
            if prev_hex_id != self._safe_id(end_obj):
                final_walk = Walk(
                    start_time=current_time,
                    end_time=current_time + timedelta(hours=final_walk_time),
                    start_hex=self._safe_hex(prev_hex_id),
                    end_hex=end_obj,
                    walk_speed=walk_speed,
                    unit=demand.unit,
                    graph=graph,
                    walk_path=walk_path,
                )
                actions.append(final_walk)

        return Route(unit=demand.unit, actions=actions, transfers=num_transfers, config=self.config)

    def get_optimal_route(self, demand, second_try=False):
        """Get the optimal route for a given demand using shortest path and logit selection."""
        walk_speed = self.config.walk_speed
        bike_speed = getattr(self.config, "bike_speed", 30.0)
        start = self._safe_hex(demand.start_hex)
        end = self._safe_hex(demand.end_hex)
        demand_time = demand.time

        walk_best_route = None
        walk_best_cost = -float("inf")
        walk_fixed_best_route = None
        walk_fixed_best_cost = -float("inf")
        ondemanddocked_best_route = None
        ondemanddockless_best_route = None
        ondemanddocked_best_cost = -float("inf")
        ondemanddockless_best_cost = -float("inf")
        multimodal_best_route = None
        multimodal_best_cost = -float("inf")

        # 1. Option: Pure Walk
        walk_time, walk_path = self.compute_walk_time(self.graph, start, end, walk_speed)
        if walk_time < float("inf"):
            walk_action = Walk(
                start_time=demand_time,
                start_hex=start,
                end_hex=end,
                unit=demand.unit,
                graph=self.graph,
                walk_speed=walk_speed,
                walk_path=walk_path,
                end_time=demand_time + timedelta(hours=walk_time),
            )
            walk_route = Route(
                unit=demand.unit, actions=[walk_action], transfers=0, config=self.config
            )
            walk_best_route = walk_route
            walk_best_cost = walk_route.total_cost

        fixed_services = [s for s in self.services if isinstance(s, FixedRouteService)]
        ondemandservices_docked = [
            s for s in self.services if isinstance(s, OnDemandRouteServiceDocked)
        ]
        ondemandservices_dockless = [
            s for s in self.services if isinstance(s, OnDemandRouteServiceDockless)
        ]

        # 2. Option: Walk + FixedRouteService (Bus/Metro)
        try:
            start_stop, start_walk_time = self.find_closest_stop(self.graph, start, walk_speed)
            end_stop, end_walk_time = self.find_closest_stop(self.graph, end, walk_speed)

            chains = (
                self.route_across_components_shortest_k(start_stop, end_stop, k=2)
                if start_stop and end_stop
                else []
            )
            for chain in chains:
                if chain is None or len(chain) == 0:
                    continue
                try:
                    fixedroute_chain_route = self.build_route_for_chain(
                        chain, start, end, demand_time, demand, self.graph, walk_speed
                    )
                    if fixedroute_chain_route.total_cost > walk_fixed_best_cost:
                        walk_fixed_best_cost = fixedroute_chain_route.total_cost
                        walk_fixed_best_route = fixedroute_chain_route
                except Exception:
                    pass
        except Exception:
            pass

        # 3. Option: OnDemandService options (Docked bikes)
        for service in ondemandservices_docked:
            # PERFORMANCE FIX: Bounded BFS cutoff for starting docks!
            start_docks_with_bikes = []
            try:
                nearby_start_hexes = nx.single_source_shortest_path_length(
                    self.graph, source=self._safe_id(start), cutoff=3
                )
                for dock in service.docking_stations:
                    dock_id = self._safe_id(dock.location)
                    if dock_id in nearby_start_hexes:
                        if self._get_available_bike(dock, service, demand_time) is not None:
                            start_docks_with_bikes.append((dock, nearby_start_hexes[dock_id]))
            except nx.NetworkXNoPath:
                pass

            if not start_docks_with_bikes:
                continue

            # PERFORMANCE FIX: Bounded BFS cutoff for ending docks!
            end_docks_with_space = []
            try:
                # Undirected graph, so distance from end to dock == dock to end
                nearby_end_hexes = nx.single_source_shortest_path_length(
                    self.graph, source=self._safe_id(end), cutoff=3
                )
                for dock in service.docking_stations:
                    dock_id = self._safe_id(dock.location)
                    if dock_id in nearby_end_hexes:
                        if len(getattr(dock, "vehicles", [])) < getattr(dock, "capacity", 100):
                            end_docks_with_space.append((dock, nearby_end_hexes[dock_id]))
            except nx.NetworkXNoPath:
                pass

            for s_dock, s_dist in start_docks_with_bikes:
                for e_dock, e_dist in end_docks_with_space:
                    try:
                        vehicle = self._get_available_bike(s_dock, service, demand_time)
                        if vehicle is None:
                            continue

                        b_start_loc = self._safe_hex(s_dock.location)
                        b_end_loc = self._safe_hex(e_dock.location)
                        if self._safe_id(b_start_loc) == self._safe_id(b_end_loc):
                            continue

                        actions = []
                        if self._safe_id(start) != self._safe_id(b_start_loc):
                            vehicle_walk_time = s_dist / walk_speed
                            walk_path = self.get_walk_shortest_path(start, b_start_loc)
                            walk_to_vehicle = Walk(
                                start_time=demand_time,
                                end_time=demand_time + timedelta(hours=vehicle_walk_time),
                                start_hex=start,
                                end_hex=b_start_loc,
                                unit=demand.unit,
                                graph=self.graph,
                                walk_speed=walk_speed,
                                walk_path=walk_path,
                            )
                            actions.append(walk_to_vehicle)

                        # PERFORMANCE FIX: Cached distance
                        bike_dist = self.get_cached_shortest_path_length(
                            self._safe_id(b_start_loc), self._safe_id(b_end_loc)
                        )
                        if bike_dist == float("inf"):
                            continue
                        drive_time = bike_dist / bike_speed
                        arrival_time = (
                            actions[-1].end_time + timedelta(hours=drive_time)
                            if actions
                            else demand_time + timedelta(hours=drive_time)
                        )

                        ride_path_local = self.get_walk_shortest_path(b_start_loc, b_end_loc)
                        ride_action = OnDemandRide(
                            start_time=actions[-1].end_time if actions else demand_time,
                            end_time=arrival_time,
                            start_hex=b_start_loc,
                            end_hex=b_end_loc,
                            unit=demand.unit,
                            service=service,
                            vehicle=vehicle,
                            ride_path=ride_path_local,
                        )
                        actions.append(ride_action)

                        if self._safe_id(b_end_loc) != self._safe_id(end):
                            off_vehicle_walk_time = e_dist / walk_speed
                            walk_path = self.get_walk_shortest_path(b_end_loc, end)
                            walk_from_vehicle = Walk(
                                start_time=ride_action.end_time,
                                end_time=ride_action.end_time
                                + timedelta(hours=off_vehicle_walk_time),
                                start_hex=b_end_loc,
                                end_hex=end,
                                unit=demand.unit,
                                graph=self.graph,
                                walk_speed=walk_speed,
                                walk_path=walk_path,
                            )
                            actions.append(walk_from_vehicle)

                        ondemand_route = Route(
                            unit=demand.unit, actions=actions, transfers=0, config=self.config
                        )
                        if ondemand_route.total_cost > ondemanddocked_best_cost:
                            ondemanddocked_best_cost = ondemand_route.total_cost
                            ondemanddocked_best_route = ondemand_route
                    except Exception:
                        continue

        # Option: OnDemandService options (Dockless bikes)
        for service in ondemandservices_dockless:
            best_vehicle, vehicle_walk_time = self.find_closest_ondemand_vehicle(
                self.graph, start, service, walk_speed, demand_time, radius=3
            )
            if best_vehicle is None:
                continue
            try:
                actions = []
                v_loc = self._safe_hex(best_vehicle.current_location)
                if self._safe_id(start) != self._safe_id(v_loc):
                    walk_path = self.get_walk_shortest_path(start, v_loc)
                    walk_to_vehicle = Walk(
                        start_time=demand_time,
                        end_time=demand_time + timedelta(hours=vehicle_walk_time),
                        start_hex=start,
                        end_hex=v_loc,
                        unit=demand.unit,
                        graph=self.graph,
                        walk_path=walk_path,
                        walk_speed=walk_speed,
                    )
                    actions.append(walk_to_vehicle)

                # PERFORMANCE FIX: Cached distance
                bike_dist = self.get_cached_shortest_path_length(
                    self._safe_id(v_loc), self._safe_id(end)
                )
                if bike_dist == float("inf"):
                    continue

                drive_time = bike_dist / bike_speed
                arrival_time = (
                    actions[-1].end_time + timedelta(hours=drive_time)
                    if actions
                    else demand_time + timedelta(hours=drive_time)
                )

                ride_path_local = self.get_walk_shortest_path(v_loc, end)
                ride_action = OnDemandRide(
                    start_time=actions[-1].end_time if actions else demand_time,
                    end_time=arrival_time,
                    start_hex=v_loc,
                    end_hex=end,
                    unit=demand.unit,
                    service=service,
                    vehicle=best_vehicle,
                    ride_path=ride_path_local,
                )
                actions.append(ride_action)

                ondemand_route = Route(
                    unit=demand.unit, actions=actions, transfers=0, config=self.config
                )
                if ondemand_route.total_cost > ondemanddockless_best_cost:
                    ondemanddockless_best_cost = ondemand_route.total_cost
                    ondemanddockless_best_route = ondemand_route
            except Exception:
                continue

        # 4. Multimodal Injection (Bike + Bus combos)
        if fixed_services and ondemandservices_docked:
            docked_srv = ondemandservices_docked[0]
            b_acc = self.get_best_bike_to_bus_access(
                self.graph, start, walk_speed, bike_speed, docked_srv, demand_time, radius=3
            )
            b_egr = self.get_best_bus_to_bike_egress(
                self.graph, end, walk_speed, bike_speed, docked_srv, demand_time, radius=3
            )

            combos = []
            if b_acc[0] is not None and end_stop is not None:
                combos.append(("bike", "walk", b_acc[0], end_stop, b_acc, None))
            if start_stop is not None and b_egr[0] is not None:
                combos.append(("walk", "bike", start_stop, b_egr[0], None, b_egr))
            if b_acc[0] is not None and b_egr[0] is not None:
                combos.append(("bike", "bike", b_acc[0], b_egr[0], b_acc, b_egr))

            for acc_mode, eg_mode, B_S, B_E, acc_data, eg_data in combos:
                chains = (
                    self.route_across_components_shortest_k(B_S, B_E, k=1) if B_S and B_E else []
                )
                for chain in chains:
                    if chain is None or len(chain) == 0:
                        continue
                    try:
                        rt = self.build_multimodal_chain_route(
                            chain,
                            start,
                            end,
                            demand_time,
                            demand,
                            self.graph,
                            walk_speed,
                            bike_speed,
                            docked_srv,
                            acc_mode,
                            eg_mode,
                            acc_data,
                            eg_data,
                        )
                        if rt.total_cost > multimodal_best_cost:
                            multimodal_best_cost = rt.total_cost
                            multimodal_best_route = rt
                    except Exception:
                        pass

        # 5. Native Logit Array building & Deduplication
        raw_options = [
            walk_best_route,
            walk_fixed_best_route,
            ondemanddocked_best_route,
            multimodal_best_route,
        ]

        unique_choices = {}
        for route in raw_options:
            if route is not None:
                signature = tuple(
                    (
                        type(a).__name__,
                        self._safe_id(getattr(a, "start_hex", getattr(a, "location", None))),
                        self._safe_id(getattr(a, "end_hex", getattr(a, "location", None))),
                    )
                    for a in route.actions
                )

                if (
                    signature not in unique_choices
                    or route.total_cost > unique_choices[signature].total_cost
                ):
                    unique_choices[signature] = route

        choices = list(unique_choices.values())

        logits = [r.total_cost for r in choices]

        # Numerically-stable softmax: subtract max to avoid overflow/underflow
        logits_np = np.array(logits, dtype=float)
        if logits_np.size == 0:
            return None

        logits_max = np.max(logits_np)
        exp_logits = np.exp(logits_np - logits_max)
        sum_exp = np.sum(exp_logits)

        # If sum_exp is zero (all logits -> -inf after subtraction) or not finite,
        # fall back to a uniform distribution to avoid NaNs in probabilities.
        if not np.isfinite(sum_exp) or sum_exp == 0:
            probabilities = np.full_like(exp_logits, 1.0 / exp_logits.size)
        else:
            probabilities = exp_logits / sum_exp

        choice = np.random.choice(choices, p=probabilities)

        return choice

    def push_route(self, route):
        if route is not None:
            self.routes_taken.append(route)

    def clear_routes(self):
        self.routes_taken.clear()

    def __repr__(self):
        return f"Network(graph_nodes={len(self.graph.nodes())}, graph_edges={len(self.graph.edges())}, routes_taken={len(self.routes_taken)})"
