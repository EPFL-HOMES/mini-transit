"""
APIServer class that talks with frontend and starts the simulation.
"""

import heapq
import json
import os
import pandas as pd
import sys
import traceback
from datetime import datetime, timedelta, time
from itertools import count

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.network import Network
from src.demand import Demand
from src.hex import Hex
from src.route import Route
from src.actions.walk import Walk
from src.actions.ride import Ride
from src.actions.wait import Wait
from src.fixed_route_service import FixedRouteService

class APIServer:
    """
    Main server class that handles simulation requests and manages city data.
    
    Attributes:
        network (Network): Network class object for the chosen city.
        demands (list): List of Demand objects for the chosen city.
    """
    
    def __init__(self):
        """Initialize the APIServer."""
        self.network = None
        self.demands = []
        self.city_name = None
    
    def init_app(self, city_name: str):
        """
        Initialize the application for a given city.
        
        Args:
            city_name (str): Name of the city ('Lausanne' or 'Renens').
        """
        # Validate city name
        if city_name not in ['Lausanne', 'Renens']:
            raise ValueError(f"Invalid city name: {city_name}. Must be 'Lausanne' or 'Renens'.")
        
        # Set up file paths
        geojson_path = f"data/{city_name}/{city_name}.geojson"
        demands_path = f"data/{city_name}/{city_name}_time_dependent_demands.csv"
        fixed_routes_path = f"data/{city_name}/fixed_routes.json"

        # Remember the active city for subsequent calls (e.g. saving results)
        self.city_name = city_name
        
        # Initialize network
        self.network = Network(geojson_path)
        
        # Load and parse demands from CSV
        self.demands = self._load_demands(demands_path)
        
        # Load fixed route services if file exists
        if os.path.exists(fixed_routes_path):
            self.network.fixed_route_services = FixedRouteService.load_from_file(
                fixed_routes_path
            )
            print(f"Loaded {len(self.network.fixed_route_services)} fixed route services")
        else:
            self.network.fixed_route_services = []
            print(f"No fixed route services file found at {fixed_routes_path}")
        
        print(f"Initialized {city_name}: {len(self.demands)} demands, {len(self.network.graph.nodes())} hexagons")
    
    def _load_demands(self, csv_path: str):
        """
        Load demands from CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing time-dependent demands.
            
        Returns:
            list: List of Demand objects.
        """
        demands = []
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Parse each row to create Demand objects
            for _, row in df.iterrows():
                # CSV has columns: departure_hour, start_hex_id, end_hex_id, demands
                hour = int(row.get('departure_hour', 0))
                start_hex_id = int(row.get('start_hex_id', 0))
                end_hex_id = int(row.get('end_hex_id', 0))
                unit = float(row.get('demands', 0.0))
                
                # Create Hex objects
                start_hex = Hex(start_hex_id)
                end_hex = Hex(end_hex_id)
                
                # Create Demand object
                demand = Demand(hour, start_hex, end_hex, unit)
                demands.append(demand)
                
        except Exception as e:
            print(f"Error loading demands from {csv_path}: {e}")
            # Return empty list if there's an error
            return []
        
        return demands
    
    def run_simulation(self, input_json: dict) -> dict:
        """
        Run the simulation with given input parameters.
        
        Args:
            input_json (dict): Input parameters for the simulation.
                Expected keys:
                - hour (int): Hour when simulation starts (filters demands by this hour)
            
        Returns:
            dict: JSON output with simulation results.
        """
        if not self.network or not self.demands:
            return {
                "status": "error",
                "message": "Simulation not initialized. Please select a city first.",
                "routes": []
            }
        
        try:
            # Extract simulation parameters
            simulation_hour = input_json.get('hour', 8)  # Default to hour 8 if not specified
            
            # Clear previous routes
            self.network.clear()

            simulation_start_datetime = datetime.combine(datetime.today(), time(simulation_hour, 0))
            
            # Filter demands by the specified hour
            filtered_demands = [demand for demand in self.demands if demand.hour == simulation_hour]
            
            if not filtered_demands:
                return {
                    "status": "warning",
                    "message": f"No demands found for hour {simulation_hour}",
                    "routes": [],
                    "simulation_time": "0.00s",
                    "demands_processed": 0,
                    "routes_generated": 0
                }
            
            simulation_start_time = datetime.now()

            def get_action_times(action):
                """Return (start, end) times for an action (object or dict)."""
                if isinstance(action, dict):
                    return action.get("start_time"), action.get("end_time")
                return getattr(action, "start_time", None), getattr(action, "end_time", None)

            # Priority queue of demand states sorted by current action end time
            demand_queue = []
            order_counter = count()
            completed_routes = []

            for demand in filtered_demands:
                route = self.network.get_optimal_route(demand, simulation_start_datetime)
                if route is None or not route.actions:
                    continue

                order = next(order_counter)
                first_action = route.actions[0]
                first_start, first_end = get_action_times(first_action)
                if first_start is None or first_end is None:
                    continue

                state = {
                    "demand": demand,
                    "route": route,
                    "current_action_index": 0,
                    "current_action_end_time": first_end,
                    "order": order
                }
                heapq.heappush(demand_queue, (first_end, order, state))


            while demand_queue:
                # Pop the first demand
                first_end_time, first_order, first_state = heapq.heappop(demand_queue)
                
                # Collect all demands with the same end_time
                same_time_demands = [(first_end_time, first_order, first_state)]
                while demand_queue and demand_queue[0][0] == first_end_time:
                    same_time_demands.append(heapq.heappop(demand_queue))
                
                # Phase 1: Update vehicles (process demands finishing Ride actions)
                finishing_ride_demands = []
                other_demands = []
                
                for end_time, order, state in same_time_demands:
                    route = state["route"]
                    actions = route.actions
                    current_index = state["current_action_index"]
                    
                    if current_index >= len(actions) - 1:
                        # Route complete
                        self.network.push_route(route)
                        completed_routes.append((order, route))
                        continue
                    
                    current_action = actions[current_index]
                    
                    # Check if current action is a Ride that's finishing
                    is_ride = (isinstance(current_action, Ride) or 
                              (isinstance(current_action, dict) and current_action.get('type') == 'Ride') or
                              (hasattr(current_action, '__class__') and current_action.__class__.__name__ == 'Ride'))
                    if is_ride:
                        finishing_ride_demands.append((end_time, order, state))
                    else:
                        other_demands.append((end_time, order, state))
                
                # Update vehicle capacities for finishing Ride actions, then move to next action
                for end_time, order, state in finishing_ride_demands:
                    route = state["route"]
                    actions = route.actions
                    current_index = state["current_action_index"]
                    current_action = actions[current_index]
                    
                    # Decrease vehicle capacity
                    if isinstance(current_action, Ride):
                        service = self._get_service_by_name(current_action.name)
                        if service and current_action.vehicle_index < len(service.vehicles):
                            vehicle = service.vehicles[current_action.vehicle_index]
                            vehicle.current_capacity = max(0, vehicle.current_capacity - route.unit)
                    
                    # Move to next action (Ride is finished, continue with remaining actions)
                    next_index = current_index + 1
                    if next_index < len(actions):
                        # Update state to point to next action
                        state["current_action_index"] = next_index
                        next_action = actions[next_index]
                        next_start, next_end = get_action_times(next_action)
                        if next_end:
                            state["current_action_end_time"] = next_end
                            # Add to other_demands to process next action
                            other_demands.append((next_end, order, state))
                        else:
                            # Cannot determine next end time, treat as completed
                            self.network.push_route(route)
                            completed_routes.append((order, route))
                    else:
                        # No more actions, route complete
                        self.network.push_route(route)
                        completed_routes.append((order, route))
                
                # Phase 2: Process demands starting Ride actions
                starting_ride_demands = []
                for end_time, order, state in other_demands:
                    route = state["route"]
                    actions = route.actions
                    current_index = state["current_action_index"]
                    
                    next_index = current_index + 1
                    if next_index < len(actions):
                        next_action = actions[next_index]
                        # Check if next action is a Ride (either object or dict with type 'Ride')
                        is_ride = (isinstance(next_action, Ride) or 
                                  (isinstance(next_action, dict) and next_action.get('type') == 'Ride') or
                                  (hasattr(next_action, '__class__') and next_action.__class__.__name__ == 'Ride'))
                        if is_ride:
                            starting_ride_demands.append((end_time, order, state))
                        else:
                            # Not a Ride action, process normally
                            self._process_demand_state(state, demand_queue, completed_routes, get_action_times)
                    else:
                        # No next action, route complete
                        self.network.push_route(route)
                        completed_routes.append((order, route))
                
                # Process demands starting Ride actions (check capacity and potentially split)
                for end_time, order, state in starting_ride_demands:
                    self._process_ride_demand(state, demand_queue, completed_routes, get_action_times, order_counter)

            # Sort completed routes back to original demand order
            completed_routes.sort(key=lambda item: item[0])
            routes = [route for _, route in completed_routes]
            
            simulation_end_time = datetime.now()
            simulation_duration = (simulation_end_time - simulation_start_time).total_seconds()
            
            # Convert routes to JSON-serializable format
            routes_data = []
            for route in routes:
                route_data = {
                    "unit": route.unit,
                    "time_taken_minutes": route.time_taken_minutes,
                    "total_fare": route.total_fare,
                    "actions": []
                }
                
                for action in route.actions:
                    # Handle both Action objects and dictionary actions
                    if hasattr(action, 'start_time'):
                        # Action object
                        action_data = {
                            "type": action.__class__.__name__,
                            "start_time": action.start_time.strftime("%H:%M"),
                            "end_time": action.end_time.strftime("%H:%M") if action.end_time else None,
                            "duration_minutes": action.duration_minutes
                        }
                        
                        # Add specific fields for Walk actions
                        if isinstance(action, Walk):
                            action_data.update({
                                "start_hex": action.start_hex.hex_id,
                                "end_hex": action.end_hex.hex_id,
                                "walk_speed": action.walk_speed,
                                "distance": action.distance
                            })
                        # Add specific fields for Ride actions
                        elif hasattr(action, 'start_hex') and hasattr(action, 'end_hex') and hasattr(action, 'name'):
                            # This is a Ride action
                            action_data.update({
                                "start_hex": action.start_hex,
                                "end_hex": action.end_hex,
                                "name": action.name,
                                "vehicle_index": action.vehicle_index
                            })
                        # Add specific fields for Wait actions
                        elif hasattr(action, 'position') and hasattr(action, 'unit'):
                            # This is a Wait action
                            action_data.update({
                                "position": action.position,
                                "unit": action.unit
                            })
                    elif isinstance(action, dict):
                        # Dictionary action
                        start_time = action['start_time']
                        end_time = action.get('end_time')
                        if isinstance(start_time, str):
                            start_time_str = start_time
                        else:
                            start_time_str = start_time.strftime("%H:%M")
                        if isinstance(end_time, str) or end_time is None:
                            end_time_str = end_time
                        else:
                            end_time_str = end_time.strftime("%H:%M")
                        duration_minutes = action.get('duration_minutes')
                        if duration_minutes is None and end_time and start_time:
                            duration_minutes = (end_time - start_time).total_seconds() / 60.0
                        action_data = {
                            "type": action.get('type', 'Unknown'),
                            "start_time": start_time_str,
                            "end_time": end_time_str,
                            "duration_minutes": duration_minutes
                        }
                        
                        # Add specific fields for Walk actions
                        if action.get('type') == 'Walk':
                            action_data.update({
                                "start_hex": action['start_hex'].hex_id,
                                "end_hex": action['end_hex'].hex_id,
                                "walk_speed": action['walk_speed'],
                                "distance": action['distance']
                            })
                    
                    route_data["actions"].append(action_data)
                
                routes_data.append(route_data)
            
            # Calculate average fare
            total_fare = sum(route.total_fare for route in routes)
            average_fare = total_fare / len(routes) if len(routes) > 0 else 0.0
            
            result = {
                "status": "success",
                "message": f"Simulation completed successfully for hour {simulation_hour}",
                "routes": routes_data,
                "simulation_time": f"{simulation_duration:.2f}s",
                "simulation_hour": simulation_hour,
                "demands_processed": len(routes),
                "routes_generated": len(routes),
                "total_units": sum(route.unit for route in routes),
                "total_time_minutes": sum(route.time_taken_minutes for route in routes),
                "total_fare": total_fare,
                "average_fare": average_fare,
            }
            
            # Save simulation results
            self._save_simulation_results(result, simulation_hour)
            
            return result
            
        except Exception as e:
            print(f"Error in run_simulation: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Simulation failed: {str(e)}",
                "routes": []
            }
    
    def _get_service_by_name(self, service_name: str):
        """Get a fixed route service by name."""
        if not hasattr(self.network, 'fixed_route_services'):
            return None
        for service in self.network.fixed_route_services:
            if service.name == service_name:
                return service
        return None
    
    def _process_demand_state(self, state, demand_queue, completed_routes, get_action_times):
        """Process a demand state that doesn't involve Ride actions."""
        route = state["route"]
        actions = route.actions
        current_index = state["current_action_index"]
        
        current_action = actions[current_index]
        _, current_end = get_action_times(current_action)
        
        next_index = current_index + 1
        if next_index >= len(actions):
            # Route complete
            self.network.push_route(route)
            completed_routes.append((state["order"], route))
            return
        
        next_action = actions[next_index]
        next_start, next_end = get_action_times(next_action)
        
        if next_start is None or next_end is None:
            # Cannot determine next end time, treat as completed
            self.network.push_route(route)
            completed_routes.append((state["order"], route))
            return
        
        # Ensure the next action starts no earlier than the previous end.
        if next_start < current_end:
            if isinstance(next_action, dict):
                next_action["start_time"] = current_end
            else:
                next_action.start_time = current_end
            next_start = current_end
        
        if next_end < next_start:
            if isinstance(next_action, dict):
                next_action["end_time"] = next_start
            else:
                next_action.end_time = next_start
            next_end = next_start
        
        state["current_action_index"] = next_index
        state["current_action_end_time"] = next_end
        heapq.heappush(demand_queue, (next_end, state["order"], state))
    
    def _process_ride_demand(self, state, demand_queue, completed_routes, get_action_times, order_counter):
        """Process a demand starting a Ride action, check capacity, and potentially split."""
        
        route = state["route"]
        actions = route.actions
        current_index = state["current_action_index"]
        demand = state["demand"]
        
        current_action = actions[current_index]
        _, current_end = get_action_times(current_action)
        
        next_index = current_index + 1
        next_action = actions[next_index]
        next_start, next_end = get_action_times(next_action)
        
        if not isinstance(next_action, Ride):
            # Not a Ride action after all, process normally
            self._process_demand_state(state, demand_queue, completed_routes, get_action_times)
            return
        
        # Get the service and vehicle
        service = self._get_service_by_name(next_action.name)
        if not service or next_action.vehicle_index >= len(service.vehicles):
            # Service not found or invalid vehicle index, skip Ride
            self._process_demand_state(state, demand_queue, completed_routes, get_action_times)
            return
        
        vehicle = service.vehicles[next_action.vehicle_index]
        available_capacity = vehicle.max_capacity - vehicle.current_capacity
        demand_units = route.unit
        
        if available_capacity >= demand_units:
            # Enough capacity, board the vehicle
            vehicle.current_capacity += demand_units
            # Process normally
            if next_start < current_end:
                next_action.start_time = current_end
                next_start = current_end
            
            if next_end < next_start:
                next_action.end_time = next_start
                next_end = next_start
            
            state["current_action_index"] = next_index
            state["current_action_end_time"] = next_end
            heapq.heappush(demand_queue, (next_end, state["order"], state))
        else:
            # Not enough capacity, split the demand
            if available_capacity > 0:
                # Clone 1: Takes available capacity on current vehicle
                clone1_units = available_capacity
                clone1_route = Route(unit=clone1_units, actions=actions.copy())
                # Update the Ride action for clone1
                clone1_ride = clone1_route.actions[next_index]
                if next_start < current_end:
                    clone1_ride.start_time = current_end
                    next_start = current_end
                if next_end < next_start:
                    clone1_ride.end_time = next_start
                    next_end = next_start
                
                vehicle.current_capacity += clone1_units
                
                clone1_state = {
                    "demand": Demand(demand.hour, demand.start_hex, demand.end_hex, clone1_units),
                    "route": clone1_route,
                    "current_action_index": next_index,
                    "current_action_end_time": next_end,
                    "order": state["order"]
                }
                heapq.heappush(demand_queue, (next_end, clone1_state["order"], clone1_state))
            
            # Clone 2: Waits for next vehicle
            clone2_units = demand_units - available_capacity
            next_vehicle_arrival = service.get_next_vehicle_arrival(next_action.start_hex, current_end)
            
            if next_vehicle_arrival is None:
                # No more vehicles, treat as completed (or handle differently)
                clone2_route = Route(unit=clone2_units, actions=actions[:next_index].copy())
                self.network.push_route(clone2_route)
                completed_routes.append((state["order"], clone2_route))
            else:
                # Create new route with Wait action and updated Ride action
                clone2_actions = actions[:next_index].copy()
                
                # Add Wait action
                wait_action = Wait(
                    start_time=current_end,
                    position=next_action.start_hex,
                    unit=clone2_units,
                    end_time=next_vehicle_arrival
                )
                clone2_actions.append(wait_action)
                
                # Find the vehicle for the next arrival
                next_vehicle_index = None
                stop_index = service.stops.index(next_action.start_hex)
                for i, v in enumerate(service.vehicles):
                    if stop_index < len(v.timetable):
                        arrival_time = v.timetable[stop_index].replace(second=0, microsecond=0)
                        if arrival_time == next_vehicle_arrival.replace(second=0, microsecond=0):
                            next_vehicle_index = i
                            break
                
                if next_vehicle_index is not None:
                    # Create new Ride action for next vehicle
                    next_vehicle = service.vehicles[next_vehicle_index]
                    alighting_stop_index = service.stops.index(next_action.end_hex)
                    if alighting_stop_index < len(next_vehicle.timetable):
                        new_ride_end_time = next_vehicle.timetable[alighting_stop_index]
                        new_ride = Ride(
                            start_time=next_vehicle_arrival,
                            end_time=new_ride_end_time,
                            name=service.name,
                            vehicle_index=next_vehicle_index,
                            start_hex=next_action.start_hex,
                            end_hex=next_action.end_hex
                        )
                        clone2_actions.append(new_ride)
                        
                        # Add remaining actions and update their times
                        remaining_actions = []
                        if next_index + 1 < len(actions):
                            # Calculate time shift: original Ride end_time vs new Ride end_time
                            original_ride_end = next_action.end_time
                            time_shift = new_ride_end_time - original_ride_end
                            
                            remaining_actions = actions[next_index + 1:]
                            for remaining_action in remaining_actions:
                                # Create a copy of the action to avoid modifying the original
                                if hasattr(remaining_action, '__class__'):
                                    # Action object - need to create a new instance
                                    action_class = remaining_action.__class__
                                    if action_class.__name__ == 'Walk':
                                        # For Walk, we need to preserve the original duration
                                        # because _calculate_end_time() uses hardcoded distance of 1.0
                                        # but the actual distance might be different
                                        new_action = Walk(
                                            start_time=remaining_action.start_time + time_shift,
                                            start_hex=remaining_action.start_hex,
                                            end_hex=remaining_action.end_hex,
                                            walk_speed=remaining_action.walk_speed
                                        )
                                        # Manually set end_time to preserve the original duration
                                        if remaining_action.end_time:
                                            new_action.end_time = remaining_action.end_time + time_shift
                                    elif action_class.__name__ == 'Wait':
                                        new_action = Wait(
                                            start_time=remaining_action.start_time + time_shift,
                                            position=remaining_action.position,
                                            unit=remaining_action.unit,
                                            end_time=remaining_action.end_time + time_shift if remaining_action.end_time else None
                                        )
                                    elif action_class.__name__ == 'Ride':
                                        new_action = Ride(
                                            start_time=remaining_action.start_time + time_shift,
                                            end_time=remaining_action.end_time + time_shift,
                                            name=remaining_action.name,
                                            vehicle_index=remaining_action.vehicle_index,
                                            start_hex=remaining_action.start_hex,
                                            end_hex=remaining_action.end_hex
                                        )
                                    else:
                                        # Unknown action type, just copy and shift times
                                        new_action = remaining_action
                                        if hasattr(new_action, 'start_time'):
                                            new_action.start_time = remaining_action.start_time + time_shift
                                        if hasattr(new_action, 'end_time') and remaining_action.end_time:
                                            new_action.end_time = remaining_action.end_time + time_shift
                                else:
                                    # Dictionary action - create a copy and shift times
                                    new_action = remaining_action.copy()
                                    if 'start_time' in new_action:
                                        if isinstance(new_action['start_time'], datetime):
                                            new_action['start_time'] = new_action['start_time'] + time_shift
                                    if 'end_time' in new_action and new_action['end_time']:
                                        if isinstance(new_action['end_time'], datetime):
                                            new_action['end_time'] = new_action['end_time'] + time_shift
                                
                                clone2_actions.append(new_action)
                        
                        clone2_route = Route(unit=clone2_units, actions=clone2_actions)
                        # Current action index is the Wait action (just added, at position next_index)
                        # clone2_actions = actions[:next_index] + [wait_action] + [new_ride] + remaining_actions
                        # So wait_action is at index next_index
                        clone2_state = {
                            "demand": Demand(demand.hour, demand.start_hex, demand.end_hex, clone2_units),
                            "route": clone2_route,
                            "current_action_index": next_index,  # Wait action is at this index
                            "current_action_end_time": next_vehicle_arrival,
                            "order": next(order_counter)
                        }
                        heapq.heappush(demand_queue, (next_vehicle_arrival, clone2_state["order"], clone2_state))
                    else:
                        # Invalid stop index, treat as completed
                        clone2_route = Route(unit=clone2_units, actions=clone2_actions)
                        self.network.push_route(clone2_route)
                        completed_routes.append((state["order"], clone2_route))
                else:
                    # Couldn't find next vehicle, treat as completed
                    clone2_route = Route(unit=clone2_units, actions=clone2_actions)
                    self.network.push_route(clone2_route)
                    completed_routes.append((state["order"], clone2_route))
    
    def _save_simulation_results(self, result: dict, simulation_hour: int):
        """
        Save simulation results to a JSON file.
        
        Args:
            result (dict): The simulation result dictionary.
            simulation_hour (int): The hour of the simulation.
        """
        try:
            def safe_print(*args, **kwargs):
                try:
                    print(*args, **kwargs)
                except UnicodeEncodeError:
                    encoding = sys.stdout.encoding or 'ascii'
                    sanitized = [
                        str(arg).encode(encoding, errors='replace').decode(encoding, errors='replace')
                        for arg in args
                    ]
                    print(*sanitized, **kwargs)
            
            city_name = self.city_name or "unknown_city"
            safe_print(f"Starting to save simulation results for {city_name} hour {simulation_hour}")
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'simulation_results')
            safe_print(f"Results directory: {results_dir}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{city_name}_hour{simulation_hour}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            safe_print(f"Saving to: {filepath}")
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            safe_print(f"Simulation results saved successfully to: {filepath}")
            
        except Exception as e:
            print(f"Error saving simulation results: {e}")
            traceback.print_exc()
    
    def get_network_info(self):
        """
        Get information about the current network.
        
        Returns:
            dict: Network information.
        """
        if self.network is None:
            return {"error": "Network not initialized"}
        
        return {
            "nodes": len(self.network.graph.nodes()),
            "edges": len(self.network.graph.edges()),
            "demands": len(self.demands)
        }
    
    def __repr__(self):
        return f"APIServer(network={self.network is not None}, demands={len(self.demands)})"
