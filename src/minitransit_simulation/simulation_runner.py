"""
SimulationRunner module that configures, runs, and serializes transit simulations.
Defines SimulationRunner, its configuration, and input/output data structures.
"""

import json
import traceback
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass, field

from .demand import Demand, DemandSampler, demand_input_from_csv
from .network import Network, NetworkConfig
from .serialization import SerializedAction, serialize_action, serialize_action_dict
from .services.fixedroute import FixedRouteService, FixedRouteServiceConfig
from .services.ondemand import OnDemandRouteServiceConfig
from .services.services_loader import load_services_from_dict, load_services_from_json
from .simulation import Simulation


@dataclass
class SimulationRunnerConfig(NetworkConfig, FixedRouteServiceConfig, OnDemandRouteServiceConfig):
    """Configuration class for the simulation runner, inheriting specific service configs."""
    # We use field(default_factory=...) for lists to avoid
    # the "mutable default argument" error in Python
    unit_sizes: list[int] = field(default_factory=lambda: [5])
    seed: int | None = None
    sampling: bool = True
    start_time: int | None = None
    end_time: int | None = None

    @classmethod
    def from_json(cls, file_path: str) -> "SimulationRunnerConfig":
        """
        Load configuration from a JSON config file.
        Returns default configuration if the file is not found.
        """
        try:
            print(f"DEBUG: Loading config from: {file_path}")
            with open(file_path, "r") as f:
                config = json.load(f)
            print(f"DEBUG: Config loaded successfully.")
            return cls(**config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config from {file_path}: {e}. Using defaults.")
            return cls(
                **{
                    "walk_speed": 10.0,
                    "bike_speed": 30.0,
                    "unit_sizes": [5],
                    "seed": None,
                    "sampling": True,
                }
            )


@dataclass
class SimulationRunnerInput:
    """Input parameters provided by the user/API to trigger the simulation."""
    start_hour: int = 8
    end_hour: int = 8


@dataclass
class SimulationRunnerResultRoute:
    """Represents a fully serialized journey taken by a demand unit."""
    unit: int
    time_taken_minutes: float
    total_fare: float
    actions: list[SerializedAction]


@dataclass
class SimulationRunnerResult:
    """The complete response payload returned by run_simulation."""
    status: str  # "success", "warning", or "error"
    message: str
    routes: list[SimulationRunnerResultRoute]

    # Timing and count metadata
    simulation_time: str | None = None
    routes_generated: int | None = None

    # Extended macroscopic metrics
    start_hour: int | None = None       
    end_hour: int | None = None        
    demands_processed: int | None = None
    input_demands_count: int | None = None
    sampling_enabled: bool | None = None
    total_units: int | None = None
    total_time_minutes: float | None = None
    total_fare: float | None = None
    average_fare: float | None = None
    network_routes_taken: int | None = None

    def save_to_json(self, path: str):
        """Saves the completely aggregated simulation results to a local JSON file."""
        try:
            print(f"Saving simulation results in {path}")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
            #print(f"Simulation results saved successfully to: {path}")
        except Exception as e:
            print(f"Error saving simulation results: {e}")
            traceback.print_exc()


class SimulationRunner:
    """
    Main controller class that handles the environment setup, demand injection,
    event-driven execution, and data serialization for the transit network.
    """

    def __init__(self, config: SimulationRunnerConfig):
        """Initializes the runner with the provided configuration."""
        self.network = None
        self.demand_inputs = []  
        self.city_name = None
        self.config = config

    def init_area(self, geojson_path: str, demands_path: str):
        """
        Initializes the spatial network and loads the time-dependent demand data.
        """
        self.network = Network(geojson_path, self.config)
        self.demand_inputs = demand_input_from_csv(demands_path)

    def add_services_from_dict(self, services_dict: dict):
        """Loads physical transit and bike services from a dictionary structure."""
        services = load_services_from_dict(
            services_dict,
            self.network,
            start_time=self.config.start_time or 0,
            end_time=self.config.end_time or 23,
        )
        self.network.services.extend(services)
        self._build_network()  

    def add_services_from_json(self, services_json_path: str):
        """Loads physical transit and bike services from a JSON file."""
        services = load_services_from_json(
            services_json_path,
            self.network,
            start_time=self.config.start_time or 0,
            end_time=self.config.end_time or 23,
        )
        self.network.services.extend(services)
        self._build_network()  

    def run_simulation(self, runner_input: SimulationRunnerInput) -> SimulationRunnerResult:
        """
        Executes the main simulation loop, processes the demands chronologically,
        and enforces rich data serialization for the output JSON.
        """
        # Guard Clause: Prevent execution if environment is not set up
        if not self.network or not self.demand_inputs:
            return SimulationRunnerResult(
                status="error",
                message="Simulation not initialized. Please select a city first.",
                routes=[],
            )

        try:
            # 1. Extract simulation temporal boundaries
            start_hour = getattr(runner_input, "start_hour", 8)
            end_hour = getattr(runner_input, "end_hour", start_hour)
            
            if start_hour > end_hour:
                return SimulationRunnerResult(
                    status="error", 
                    message="start_hour cannot be greater than end_hour", 
                    routes=[]
                )
            
            use_sampling = getattr(self.config, "sampling", True)
            print(f"DEBUG: Running simulation from {start_hour}:00 to {end_hour}:00. Sampling: {use_sampling}")

            # 2. Reset the network states from previous runs
            self.network.clear_routes()
            simulation_start_time = datetime.now()

            # 3. Aggregate and sample demands hour by hour
            processed_demands = []
            total_input_demands = 0
            
            unit_sizes = self.config.unit_sizes or [5]
            seed = getattr(self.config, "seed", None)
            sampler = DemandSampler(unit_sizes=unit_sizes, seed=seed) if use_sampling else None

            for current_hr in range(start_hour, end_hour + 1):
                hr_inputs = [d for d in self.demand_inputs if d.hour == current_hr]
                total_input_demands += len(hr_inputs)

                if not hr_inputs:
                    continue

                if use_sampling:
                    sampled = sampler.sample_hourly_demand(hr_inputs)
                    processed_demands.extend(sampled)
                else:
                    # --- DYNAMIC HEX WRAPPER ---
                    # Protects the underlying simulation engine from crashing when 
                    # demand_input passes pure integers instead of Hex objects.
                    class GenericHex:
                        def __init__(self, hid):
                            self.hex_id = hid
                            
                    def _ensure_hex(val):
                        return val if hasattr(val, 'hex_id') else GenericHex(val)
                    # ---------------------------

                    for demand_input in hr_inputs:
                        if demand_input.unit <= 0:
                            continue
                        demand_time = datetime(2024, 1, 1, current_hr, 0, 0)
                        processed_demands.append(Demand(
                            time=demand_time,
                            # Safely wrap the raw integer IDs into object shells
                            start_hex=_ensure_hex(demand_input.start_hex),
                            end_hex=_ensure_hex(demand_input.end_hex),
                            unit=demand_input.unit,
                        ))

            # Guard Clause: If no demands match the criteria, exit early cleanly
            if not processed_demands:
                return SimulationRunnerResult(
                    status="success",
                    message=f"Simulation completed successfully (No demands found for hours {start_hour}-{end_hour})",
                    routes=[],
                    simulation_time="0.00s",
                    start_hour=start_hour,
                    end_hour=end_hour,
                    demands_processed=0,
                    input_demands_count=total_input_demands,
                    sampling_enabled=use_sampling,
                    routes_generated=0,
                    total_units=0,
                    total_time_minutes=0,
                    total_fare=0.0,
                    average_fare=0.0,
                    network_routes_taken=0,
                )

            ## 4. Core Simulation Execution
            simulation = Simulation(self.network)
            for demand in processed_demands:
                simulation.add_demand(demand)

            # Triggers the routing and choice logic for all agents
            routes = simulation.run()

            simulation_end_time = datetime.now()
            simulation_duration = (simulation_end_time - simulation_start_time).total_seconds()

            # --- INSERT YOUR PROFILING CODE HERE ---
            # Calculate percentage of time spent in get_optimal_route
            optimal_route_time = getattr(simulation, 'total_optimal_route_time', 0.0) 
            optimal_route_percentage = (
                (optimal_route_time / simulation_duration * 100) if simulation_duration > 0 else 0
            )
            print(f"\nTime analysis for simulation (hours {start_hour}-{end_hour}):")
            print(f"  Total simulation time: {simulation_duration:.3f} seconds")
            print(f"  Time in get_optimal_route: {optimal_route_time:.3f} seconds ({optimal_route_percentage:.1f}%)")
            # ---------------------------------------

      
            

            # 5. Data Enrichment & Serialization
            routes_data = []
            for route in routes:
                route_data = {
                    "unit": route.unit,
                    "time_taken_minutes": route.time_taken_minutes,
                    "total_fare": route.total_fare,
                    "actions": [],
                }

                for action in route.actions:
                    action_data = {}
                    if hasattr(action, "start_time"):
                        action_data = serialize_action(action)
                    elif isinstance(action, dict):
                        action_data = serialize_action_dict(action)

                    # ---------------------------------------------------------
                    # CRITICAL DATA ENRICHMENT: Ensure OnDemandRide & Wait 
                    # perfectly align with the JSON schema requirements.
                    # ---------------------------------------------------------
                    act_type = type(action).__name__
                    if act_type == "OnDemandRide" or action_data.get("type") == "OnDemandRide":
                        action_data["type"] = "OnDemandRide"
                        
                        # Extract exact coordinates safely
                        s_hex = getattr(action, "start_hex", None)
                        e_hex = getattr(action, "end_hex", None)
                        start_id = getattr(s_hex, "hex_id", s_hex) if s_hex is not None else "N/A"
                        end_id = getattr(e_hex, "hex_id", e_hex) if e_hex is not None else "N/A"
                        
                        action_data["start_hex"] = start_id
                        action_data["end_hex"] = end_id
                        
                        # Extract Service Identity
                        srv = getattr(action, "service", None)
                        action_data["service_name"] = getattr(srv, "name", "Bike Share")
                        
                        # Extract Vehicle Tracking ID (Essential for fleet monitoring)
                        veh = getattr(action, "vehicle", None)
                        if veh:
                            action_data["vehicle_id"] = getattr(veh, "vehicle_id", "Unknown Bike")

                        # =========================================================
                        # ADDED: Detailed Path for Bikes (similar to Walk)
                        # Calculates the exact hex-by-hex route on the network graph
                        # =========================================================
                        action_path = getattr(action, "path", None)
                        if not action_path and start_id != "N/A" and end_id != "N/A":
                            try:
                                import networkx as nx
                                # Use the global network graph to find the shortest path
                                action_path = nx.shortest_path(
                                    self.network.graph, 
                                    source=start_id, 
                                    target=end_id
                                )
                            except Exception:
                                # Fallback to start and end if nodes are isolated
                                action_path = [start_id, end_id]
                                
                        action_data["path"] = action_path if action_path else [start_id, end_id]
                        
                        # ---------------------------------------------------------
                        # NEW: Inject Distance and Speed metrics for Bike evaluation
                        # ---------------------------------------------------------
                        # Speed is fetched directly from the global config
                        action_data["speed"] = getattr(self.config, 'bike_speed', 35.0)
                        # Distance in hex grids is exactly (number of nodes in path - 1)
                        action_data["distance"] = len(action_data["path"]) - 1 if action_data["path"] else 0
                        # =========================================================
                            
                    elif act_type == "Wait" or action_data.get("type") == "Wait":
                        # Standardize wait actions to have start/end hex matching their static location
                        loc = getattr(action, "location", None)
                        loc_id = getattr(loc, "hex_id", loc)
                        if "start_hex" not in action_data: action_data["start_hex"] = loc_id
                        if "end_hex" not in action_data: action_data["end_hex"] = loc_id
                    # ---------------------------------------------------------

                    route_data["actions"].append(action_data)

                routes_data.append(route_data)

            # 6. Global Metrics Calculation
            total_fare = sum(route.total_fare for route in routes)
            average_fare = total_fare / len(routes) if len(routes) > 0 else 0.0

            return SimulationRunnerResult(
                status="success",
                message=f"Simulation completed successfully for hours {start_hour} to {end_hour}",
                routes=routes_data,
                simulation_time=f"{simulation_duration:.2f}s",
                start_hour=start_hour,
                end_hour=end_hour,
                demands_processed=len(processed_demands),
                input_demands_count=total_input_demands,
                sampling_enabled=use_sampling,
                routes_generated=len(routes),
                total_units=sum(route.unit for route in routes),
                total_time_minutes=sum(route.time_taken_minutes for route in routes),
                total_fare=total_fare,
                average_fare=average_fare,
                network_routes_taken=len(self.network.routes_taken),
            )

        except Exception as e:
            traceback.print_exc()
            return SimulationRunnerResult(
                status="error",
                message=f"Simulation failed: {str(e)}",
                routes=[],
            )

    def get_network_info(self):
        """Returns macroscopic statistics about the currently loaded network graph."""
        if self.network is None:
            return {"error": "Network not initialized"}

        return {
            "nodes": len(self.network.graph.nodes()),
            "edges": len(self.network.graph.edges()),
            "demand_inputs": len(self.demand_inputs),
        }

    def _build_network(self):
        """
        Triggers the underlying network graph compilation.
        Must be invoked strictly after all JSON services have been appended.
        """
        if self.network is None:
            raise ValueError("Network must be initialized before building it.")

        fixed_services = [s for s in self.network.services if isinstance(s, FixedRouteService)]
        self.network.build_fixedroute_graph(fixed_services)
        self.network.build_component_distance_table()

    def __repr__(self):
        return f"SimulationRunner(network_loaded={self.network is not None}, total_demands={len(self.demand_inputs)})"