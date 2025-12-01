"""
APIServer class that talks with frontend and starts the simulation.
"""

import json
import os
import sys

import pandas as pd

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List

from src.actions.ride import Ride
from src.actions.wait import Wait
from src.actions.walk import Walk
from src.demand import Demand
from src.hex import Hex
from src.models import DemandInput, FixedRouteServiceModel
from src.network import Network
from src.sampler import DemandSampler
from src.simulation import Simulation


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
        self.demand_inputs = []  # Store input demands (DemandInput objects)
        self.city_name = None
        self.config = self._load_config()

    def _load_config(self):
        """
        Load configuration from config.json.

        Returns:
            dict: Configuration dictionary with default values if file not found.
        """
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "config.json"
            )
            print(f"DEBUG: Loading config from: {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"DEBUG: Config loaded successfully: {config}")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config.json: {e}. Using defaults.")
            return {
                "walk_speed": 10.0,
                "unit_sizes": [5],
                "seed": None,
                "sampling": True,
            }

    def init_app(self, city_name: str):
        """
        Initialize the application for a given city.

        Args:
            city_name (str): Name of the city ('Lausanne' or 'Renens').
        """
        # Validate city name
        if city_name not in ["Lausanne", "Renens"]:
            raise ValueError(f"Invalid city name: {city_name}. Must be 'Lausanne' or 'Renens'.")

        # Set up file paths
        geojson_path = f"data/{city_name}/{city_name}.geojson"
        demands_path = f"data/{city_name}/{city_name}_time_dependent_demands.csv"

        # Store city name
        self.city_name = city_name

        # Initialize network
        self.network = Network(geojson_path)

        # Load fixed route services from JSON
        fixed_route_services_path = f"data/{city_name}/fixed_route_services.json"
        self._load_fixed_route_services_from_json(fixed_route_services_path)

        # Load and parse input demands from CSV (as DemandInput objects)
        self.demand_inputs = self._load_demands(demands_path)

        print(
            f"Initialized {city_name}: {len(self.demand_inputs)} input demands, {len(self.network.graph.nodes())} hexagons, {len(self.network.services)} services"
        )

    def _load_fixed_route_services_from_json(self, json_path: str):
        """
        Load fixed-route services from JSON file and add them to the network.

        Args:
            json_path (str): Path to the JSON file containing fixed route services.
        """
        from datetime import datetime, timedelta

        from src.services.fixedroute import FixedRouteService

        # Check if file exists
        if not os.path.exists(json_path):
            print(f"Fixed route services file not found: {json_path}. Skipping service loading.")
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            services_data = data.get("services", [])

            # Default values for service parameters (used if not specified in JSON)
            default_stopping_time_minutes = 1  # 1 minute at each stop
            default_travel_time_minutes = 2  # 2 minutes per hexagon
            default_bidirectional = True

            # Create frequency period for the entire day (0:00 to 23:59)
            # This covers all possible simulation hours
            day_start = datetime(2024, 1, 1, 0, 0, 0)
            day_end = datetime(2024, 1, 1, 23, 59, 59)

            for service_info in services_data:
                name = service_info.get("name", "Unknown Service")
                stops_hex_ids = service_info.get("stops", [])
                frequency_minutes = service_info.get("frequency", 10)  # Default 10 minutes
                capacity = float(service_info.get("capacity", 50))  # Default 50

                # Get stopping_time and travel_time from JSON (in minutes), use defaults if not provided
                stopping_time_minutes = service_info.get(
                    "stopping_time", default_stopping_time_minutes
                )
                travel_time_minutes = service_info.get("travel_time", default_travel_time_minutes)

                # Convert minutes to timedelta
                stopping_time = timedelta(minutes=stopping_time_minutes)
                travel_time = timedelta(minutes=travel_time_minutes)

                # Convert hex IDs to Hex objects
                stops = [Hex(hex_id) for hex_id in stops_hex_ids]

                if not stops:
                    print(f"Warning: Service '{name}' has no stops. Skipping.")
                    continue

                # Create frequency period: (start_time, end_time, frequency)
                frequency_timedelta = timedelta(minutes=frequency_minutes)
                freq_period = [(day_start, day_end, frequency_timedelta)]

                # Create FixedRouteService
                fixed_route_service = FixedRouteService(
                    name=name,
                    stops=stops,
                    capacity=capacity,
                    stopping_time=stopping_time,
                    travel_time=travel_time,
                    network=self.network,
                    freq_period=freq_period,
                    bidirectional=default_bidirectional,
                )

                # Add to network
                self.network.services.append(fixed_route_service)
                print(
                    f"Loaded service: {name} with {len(stops)} stops, frequency {frequency_minutes} min, capacity {capacity}, stopping_time {stopping_time_minutes} min, travel_time {travel_time_minutes} min"
                )

        except Exception as e:
            print(f"Error loading fixed route services from {json_path}: {e}")
            import traceback

            traceback.print_exc()

    def _load_fixedroute_services(self, services: List[FixedRouteServiceModel], *args, **kwargs):
        """
        Load fixed-route services into the network.

        Args:
            *args: Positional arguments for FixedRouteService.
            **kwargs: Keyword arguments for FixedRouteService.
        """
        from src.services.fixedroute import FixedRouteService

        for service_model in services:
            fixed_route_service = FixedRouteService(
                name=service_model.name,
                stops=service_model.stops,
                capacity=service_model.capacity,
                stopping_time=service_model.stopping_time,
                travel_time=service_model.travel_time,
                vehicles=service_model.vehicles,
            )
            self.network.services.append(fixed_route_service)

    def _load_demands(self, csv_path: str):
        """
        Load input demands from CSV file as DemandInput objects.

        Args:
            csv_path (str): Path to the CSV file containing time-dependent demands.

        Returns:
            list: List of DemandInput objects.
        """
        demand_inputs = []

        try:
            # Read CSV file
            df = pd.read_csv(csv_path)

            # Parse each row to create DemandInput objects
            for _, row in df.iterrows():
                # CSV has columns: departure_hour, start_hex_id, end_hex_id, demands
                hour = int(row.get("departure_hour", 0))
                start_hex_id = int(row.get("start_hex_id", 0))
                end_hex_id = int(row.get("end_hex_id", 0))
                unit = int(row.get("demands", 0))

                # Create Hex objects
                start_hex = Hex(start_hex_id)
                end_hex = Hex(end_hex_id)

                # Create DemandInput object
                # The unit field represents λ (lambda) for Poisson sampling
                demand_input = DemandInput(
                    hour=hour,
                    start_hex=start_hex,
                    end_hex=end_hex,
                    unit=unit,
                )
                demand_inputs.append(demand_input)

        except Exception as e:
            print(f"Error loading demands from {csv_path}: {e}")
            # Return empty list if there's an error
            return []

        return demand_inputs

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
        if not self.network or not self.demand_inputs:
            return {
                "status": "error",
                "message": "Simulation not initialized. Please select a city first.",
                "routes": [],
            }

        try:
            from datetime import datetime, timedelta

            # Extract simulation parameters
            simulation_hour = input_json.get("hour", 8)  # Default to hour 8 if not specified

            # Get sampling flag from config.json
            use_sampling = self.config.get("sampling", True)
            print(f"DEBUG: Sampling flag from config: {use_sampling} (type: {type(use_sampling)})")
            print(f"DEBUG: Full config: {self.config}")

            # Clear previous routes
            self.network.clear()

            # Filter input demands by the specified hour
            filtered_demand_inputs = [
                demand_input
                for demand_input in self.demand_inputs
                if demand_input.hour == simulation_hour
            ]

            if not filtered_demand_inputs:
                return {
                    "status": "warning",
                    "message": f"No input demands found for hour {simulation_hour}",
                    "routes": [],
                    "simulation_time": "0.00s",
                    "demands_processed": 0,
                    "routes_generated": 0,
                }

            simulation_start_time = datetime.now()

            # Process demands based on sampling flag
            if use_sampling:
                print(
                    f"DEBUG: Using sampling mode. Input demands count: {len(filtered_demand_inputs)}"
                )
                # Sample demands from input demands using DemandSampler
                unit_sizes = self.config.get("unit_sizes", [5])
                seed = self.config.get("seed", None)  # None means no seed (random)
                print(f"DEBUG: Sampler config - unit_sizes: {unit_sizes}, seed: {seed}")
                sampler = DemandSampler(unit_sizes=unit_sizes, seed=seed)
                processed_demands = sampler.sample_hourly_demand(filtered_demand_inputs)
                print(
                    f"DEBUG: Sampled {len(processed_demands)} demands from {len(filtered_demand_inputs)} input demands"
                )

                if not processed_demands:
                    return {
                        "status": "warning",
                        "message": f"No demands sampled for hour {simulation_hour}",
                        "routes": [],
                        "simulation_time": "0.00s",
                        "demands_processed": 0,
                        "routes_generated": 0,
                    }
            else:
                print(f"DEBUG: Using direct conversion mode (no sampling)")
                # Convert DemandInput to Demand objects directly (old behavior)
                # One Demand per DemandInput, using unit value as-is
                processed_demands = []
                for demand_input in filtered_demand_inputs:
                    # Skip if unit is 0
                    if demand_input.unit <= 0:
                        continue

                    # Create datetime for the demand time
                    demand_time = datetime(2024, 1, 1, demand_input.hour, 0, 0)

                    # Create Demand object with unit value as-is
                    demand = Demand(
                        time=demand_time,
                        start_hex=demand_input.start_hex,
                        end_hex=demand_input.end_hex,
                        unit=demand_input.unit,
                    )
                    processed_demands.append(demand)

                if not processed_demands:
                    return {
                        "status": "warning",
                        "message": f"No demands found for hour {simulation_hour}",
                        "routes": [],
                        "simulation_time": "0.00s",
                        "demands_processed": 0,
                        "routes_generated": 0,
                    }

            # Initialize event-driven simulation
            simulation = Simulation(self.network)

            # Add all processed demands to the simulation
            for demand in processed_demands:
                simulation.add_demand(demand)

            # Run the simulation (processes events in chronological order)
            routes = simulation.run()

            simulation_end_time = datetime.now()
            simulation_duration = (simulation_end_time - simulation_start_time).total_seconds()

            # Convert routes to JSON-serializable format
            routes_data = []
            for route in routes:
                route_data = {
                    "unit": route.unit,
                    "time_taken_minutes": route.time_taken_minutes,
                    "total_fare": route.total_fare,
                    "actions": [],
                }

                for action in route.actions:
                    # Handle both Action objects and dictionary actions
                    if hasattr(action, "start_time"):
                        # Action object
                        action_data = {
                            "type": action.__class__.__name__,
                            "start_time": action.start_time.strftime("%H:%M"),
                            "end_time": (
                                action.end_time.strftime("%H:%M") if action.end_time else None
                            ),
                            "duration_minutes": action.duration_minutes,
                        }

                        # Add specific fields for different action types
                        if isinstance(action, Walk):
                            action_data.update(
                                {
                                    "start_hex": action.start_hex.hex_id,
                                    "end_hex": action.end_hex.hex_id,
                                    "walk_speed": action.walk_speed,
                                    "distance": action.distance,
                                }
                            )
                        elif isinstance(action, Ride):
                            action_data.update(
                                {
                                    "start_hex": action.start_hex.hex_id,
                                    "end_hex": action.end_hex.hex_id,
                                }
                            )
                        elif isinstance(action, Wait):
                            # For Wait, location is the hex where waiting occurs
                            action_data.update(
                                {
                                    "start_hex": action.location.hex_id,
                                    "end_hex": action.location.hex_id,
                                }
                            )
                    elif isinstance(action, dict):
                        # Dictionary action
                        start_time_obj = action["start_time"]
                        end_time_obj = action.get("end_time")
                        action_data = {
                            "type": action.get("type", "Unknown"),
                            "start_time": (
                                start_time_obj.strftime("%H:%M")
                                if hasattr(start_time_obj, "strftime")
                                else str(start_time_obj)
                            ),
                            "end_time": (
                                end_time_obj.strftime("%H:%M")
                                if end_time_obj and hasattr(end_time_obj, "strftime")
                                else (str(end_time_obj) if end_time_obj else None)
                            ),
                            "duration_minutes": (
                                (end_time_obj - start_time_obj).total_seconds() / 60.0
                                if end_time_obj and hasattr(end_time_obj, "__sub__")
                                else 0.0
                            ),
                        }

                        # Add specific fields for different action types
                        action_type = action.get("type")
                        if action_type == "Walk":
                            action_data.update(
                                {
                                    "start_hex": action["start_hex"].hex_id,
                                    "end_hex": action["end_hex"].hex_id,
                                    "walk_speed": action["walk_speed"],
                                    "distance": action["distance"],
                                }
                            )
                        elif action_type == "Ride":
                            if "start_hex" in action and "end_hex" in action:
                                action_data.update(
                                    {
                                        "start_hex": (
                                            action["start_hex"].hex_id
                                            if hasattr(action["start_hex"], "hex_id")
                                            else action["start_hex"]
                                        ),
                                        "end_hex": (
                                            action["end_hex"].hex_id
                                            if hasattr(action["end_hex"], "hex_id")
                                            else action["end_hex"]
                                        ),
                                    }
                                )
                        elif action_type == "Wait":
                            if "location" in action:
                                location = action["location"]
                                location_hex_id = (
                                    location.hex_id if hasattr(location, "hex_id") else location
                                )
                                action_data.update(
                                    {
                                        "start_hex": location_hex_id,
                                        "end_hex": location_hex_id,
                                    }
                                )

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
                "demands_processed": len(processed_demands),
                "input_demands_count": len(filtered_demand_inputs),
                "sampling_enabled": use_sampling,
                "routes_generated": len(routes),
                "total_units": sum(route.unit for route in routes),
                "total_time_minutes": sum(route.time_taken_minutes for route in routes),
                "total_fare": total_fare,
                "average_fare": average_fare,
                "network_routes_taken": len(self.network.routes_taken),
            }

            # Save simulation results to file
            self._save_simulation_results(result, simulation_hour)

            return result

        except Exception as e:
            return {"status": "error", "message": f"Simulation failed: {str(e)}", "routes": []}

    def _save_simulation_results(self, result: dict, simulation_hour: int):
        """
        Save simulation results to a JSON file.

        Args:
            result (dict): The simulation result dictionary.
            simulation_hour (int): The hour of the simulation.
        """
        try:
            import traceback
            from datetime import datetime

            city_name = self.city_name or "Unknown"
            print(f"Starting to save simulation results for {city_name} hour {simulation_hour}")

            # Create results directory if it doesn't exist
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "simulation_results",
            )
            print(f"Results directory: {results_dir}")
            os.makedirs(results_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{city_name}_hour{simulation_hour}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            print(f"Saving to: {filepath}")

            # Save to file
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)

            print(f"Simulation results saved successfully to: {filepath}")

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
            "demand_inputs": len(self.demand_inputs),
        }

    def __repr__(self):
        return f"APIServer(network={self.network is not None}, demand_inputs={len(self.demand_inputs)})"
