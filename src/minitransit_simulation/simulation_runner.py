"""
SimulationRunner module that configures, runs, and serializes transit simulations.
Defines SimulationRunner, its configuration, and input/output data structures.
"""

import json
from dataclasses import asdict, dataclass, field

from .services.ondemand import OnDemandRouteServiceConfig

from .demand import Demand, DemandSampler, demand_input_from_csv
from .network import Network, NetworkConfig
from .serialization import SerializedAction, serialize_action, serialize_action_dict
from .services.fixedroute import FixedRouteServiceConfig
from .services.services_loader import load_services_from_json
from .simulation import Simulation


@dataclass
class SimulationRunnerConfig(NetworkConfig, FixedRouteServiceConfig, OnDemandRouteServiceConfig):
    # We use field(default_factory=...) for lists to avoid
    # the "mutable default argument" error in Python
    unit_sizes: list[int] = field(default_factory=lambda: [5])
    seed: int | None = None
    sampling: bool = True

    @classmethod
    def from_json(cls, file_path: str) -> "SimulationRunnerConfig":
        """
        Load configuration from a json config file.

        Returns:
            "SimulationRunnerConfig": Configuration object with default values if file not found.
        """
        try:
            print(f"DEBUG: Loading config from: {file_path}")
            with open(file_path, "r") as f:
                config = json.load(f)
            print(f"DEBUG: Config loaded successfully: {config}")
            return cls(**config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config from {file_path}: {e}. Using defaults.")
            return cls(
                **{
                    "walk_speed": 10.0,
                    "unit_sizes": [5],
                    "seed": None,
                    "sampling": True,
                }
            )


@dataclass
class SimulationRunnerInput:
    """Input parameters for running the simulation."""

    hour: int  # Hour when simulation starts (filters demands by this hour)


@dataclass
class SimulationRunnerResultRoute:
    """Represents a full journey taken by a demand unit."""

    unit: int
    time_taken_minutes: float
    total_fare: float
    actions: list[SerializedAction]


@dataclass
class SimulationRunnerResult:
    """The complete response shape returned by run_simulation."""

    status: str  # "success", "warning", or "error"
    message: str
    routes: list[SimulationRunnerResultRoute]

    # Timing and count metadata (present in success and warning)
    simulation_time: str | None = None
    routes_generated: int | None = None

    # Extended metrics (only present in "success" status)
    simulation_hour: int | None = None
    demands_processed: int | None = None
    input_demands_count: int | None = None
    sampling_enabled: bool | None = None
    total_units: int | None = None
    total_time_minutes: float | None = None
    total_fare: float | None = None
    average_fare: float | None = None
    network_routes_taken: int | None = None

    def save_to_json(self, path: str):
        """
        Save simulation results to a JSON file.

        Args:
            path (str): Destination file path where the simulation results JSON will be written.
        """
        try:
            import traceback

            print(f"Saving simulation results in {path}")
            # Save to file
            with open(path, "w") as f:
                json.dump(asdict(self), f, indent=2)

            print(f"Simulation results saved successfully to: {path}")

        except Exception as e:
            print(f"Error saving simulation results: {e}")
            traceback.print_exc()


class SimulationRunner:
    """
    Main server class that handles simulation requests and manages city data.

    Attributes:
        network (Network): Network class object for the chosen city.
        demands (list): List of Demand objects for the chosen city.
    """

    def __init__(self, config: SimulationRunnerConfig):
        """Initialize the SimulationRunner with the given configuration."""
        self.network = None
        self.demand_inputs = []  # Store input demands (DemandInput objects)
        self.city_name = None
        self.config = config

    def init_area(
        self,
        geojson_path: str,
        demands_path: str,
        services_json_path: str,
    ):
        """
        Initialize the application for a given city.

        Args:
            geojson_path (str): Path to the GeoJSON file for the city's network.
            demands_path (str): Path to the CSV file containing time-dependent demands.
            services_json_path (str): Path to the JSON file containing both fixed route and on-demand services.
        """
        self.network = Network(geojson_path, self.config)
        self.demand_inputs = demand_input_from_csv(demands_path)
        services = load_services_from_json(services_json_path, self.network)
        self.network.services.extend(services)

    def run_simulation(self, input_json: SimulationRunnerInput) -> SimulationRunnerResult:
        """
        Run the simulation with given input parameters.

        Args:
            input_json (SimulationRunnerInput): Input parameters for the simulation.

        Returns:
            SimulationRunnerResult: JSON output with simulation results.
        """
        if not self.network or not self.demand_inputs:
            return SimulationRunnerResult(
                status="error",
                message="Simulation not initialized. Please select a city first.",
                routes=[],
            )

        try:
            from datetime import datetime, timedelta

            # Extract simulation parameters
            simulation_hour = (
                input_json.hour if hasattr(input_json, "hour") else 8
            )  # Default to hour 8 if not specified

            use_sampling = self.config.sampling if hasattr(self.config, "sampling") else True
            print(f"DEBUG: Sampling flag from config: {use_sampling} (type: {type(use_sampling)})")
            print(f"DEBUG: Full config: {self.config}")

            # Clear previous routes
            self.network.clear_routes()

            # Filter input demands by the specified hour
            filtered_demand_inputs = [
                demand_input
                for demand_input in self.demand_inputs
                if demand_input.hour == simulation_hour
            ]

            if not filtered_demand_inputs:
                return SimulationRunnerResult(
                    status="warning",
                    message=f"No input demands found for hour {simulation_hour}",
                    routes=[],
                    simulation_time="0.00s",
                    routes_generated=0,
                )

            simulation_start_time = datetime.now()

            # Process demands based on sampling flag
            if use_sampling:
                print(
                    f"DEBUG: Using sampling mode. Input demands count: {len(filtered_demand_inputs)}"
                )
                # Sample demands from input demands using DemandSampler
                unit_sizes = self.config.unit_sizes or [5]
                seed = (
                    self.config.seed if hasattr(self.config, "seed") else None
                )  # None means no seed (random)
                print(f"DEBUG: Sampler config - unit_sizes: {unit_sizes}, seed: {seed}")
                sampler = DemandSampler(unit_sizes=unit_sizes, seed=seed)
                processed_demands = sampler.sample_hourly_demand(filtered_demand_inputs)
                print(
                    f"DEBUG: Sampled {len(processed_demands)} demands from {len(filtered_demand_inputs)} input demands"
                )

                if not processed_demands:
                    return SimulationRunnerResult(
                        status="warning",
                        message=f"No demands sampled for hour {simulation_hour}",
                        routes=[],
                        simulation_time="0.00s",
                        routes_generated=0,
                    )
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
                    return SimulationRunnerResult(
                        status="warning",
                        message=f"No demands found for hour {simulation_hour}",
                        routes=[],
                        simulation_time="0.00s",
                        routes_generated=0,
                    )

            # Initialize event-driven simulation
            simulation = Simulation(self.network)

            # Add all processed demands to the simulation
            for demand in processed_demands:
                simulation.add_demand(demand)

            # Run the simulation (processes events in chronological order)
            routes = simulation.run()

            simulation_end_time = datetime.now()
            simulation_duration = (simulation_end_time - simulation_start_time).total_seconds()

            # Calculate percentage of time spent in get_optimal_route
            optimal_route_time = simulation.total_optimal_route_time
            optimal_route_percentage = (optimal_route_time / simulation_duration * 100) if simulation_duration > 0 else 0
            print(f"\nTime analysis for simulation (hour {simulation_hour}):")
            print(f"  Total simulation time: {simulation_duration:.3f} seconds")
            print(f"  Time in get_optimal_route: {optimal_route_time:.3f} seconds ({optimal_route_percentage:.1f}%)")

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
                        action_data = serialize_action(action)
                    elif isinstance(action, dict):
                        action_data = serialize_action_dict(action)

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

            return SimulationRunnerResult(
                status="success",
                message=f"Simulation completed successfully for hour {simulation_hour}",
                routes=routes_data,
                simulation_time=f"{simulation_duration:.2f}s",
                simulation_hour=simulation_hour,
                demands_processed=len(processed_demands),
                input_demands_count=len(filtered_demand_inputs),
                sampling_enabled=use_sampling,
                routes_generated=len(routes),
                total_units=sum(route.unit for route in routes),
                total_time_minutes=sum(route.time_taken_minutes for route in routes),
                total_fare=total_fare,
                average_fare=average_fare,
                network_routes_taken=len(self.network.routes_taken),
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            return SimulationRunnerResult(
                status="error",
                message=f"Simulation failed: {str(e)}",
                routes=[],
            )

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
        return f"SimulationRunner(network={self.network is not None}, demand_inputs={len(self.demand_inputs)})"
