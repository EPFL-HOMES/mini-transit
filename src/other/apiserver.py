"""
APIServer class that talks with frontend and starts the simulation.
"""

import json
import os
import sys

from src.minitransit_simulation.simulation_runner import (
    SimulationRunner,
    SimulationRunnerConfig,
    SimulationRunnerInput,
    SimulationRunnerResult,
)

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class APIServer:
    """
    Main server class that handles simulation requests and manages city data.

    Attributes:
        network (Network): Network class object for the chosen city.
        demands (list): List of Demand objects for the chosen city.
    """

    def __init__(self):
        """Initialize the APIServer."""
        self.runner = SimulationRunner(self._load_config())
        self.city_name = None
        self.last_simulation_result = None  # Store last simulation result for visualization

    def _load_config(self):
        """
        Load configuration from config.json.

        Returns:
            dict: Configuration dictionary with default values if file not found.
        """
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "config.json"
        )
        return SimulationRunnerConfig.from_json(config_path)

    def init_app(self, city_name: str):
        """
        Initialize the application for a given city.

        Args:
            city_name (str): Name of the city ('Lausanne' or 'Renens').
        """
        # Validate city name
        if city_name not in ["Lausanne", "Renens"]:
            raise ValueError(f"Invalid city name: {city_name}. Must be 'Lausanne' or 'Renens'.")

        # Store city name
        self.city_name = city_name

        # Set up file paths
        geojson_path = f"data/{city_name}/{city_name}.geojson"
        demands_path = f"data/{city_name}/{city_name}_time_dependent_demands.csv"
        fixed_route_services_path = f"data/{city_name}/fixed_route_services.json"

        self.runner.init_area(
            geojson_path=geojson_path,
            demands_path=demands_path,
        )
        self.runner.add_fixed_route_services_from_json(fixed_route_services_path)

        print(
            f"Initialized {city_name}: {len(self.runner.demand_inputs)} input demands, {len(self.runner.network.graph.nodes())} hexagons, {len(self.runner.network.services)} services"
        )

    def run_simulation(self, input_json: SimulationRunnerInput) -> SimulationRunnerResult:
        """
        Run the simulation with given input parameters.

        Args:
            input_json (SimulationRunnerInput): Input parameters for the simulation.

        Returns:
            SimulationRunnerResult: Output with simulation results.
        """

        try:
            result = self.runner.run_simulation(input_json)
            # Store last simulation result for visualization
            if result.status == "success":
                self.last_simulation_result = {
                    "result": result,
                    "routes": result.routes,
                    "simulation_hour": result.simulation_hour,
                }

                print(
                    f"DEBUG: last_simulation_result stored: {self.last_simulation_result is not None}"
                )
                print(
                    f"DEBUG: last_simulation_result has routes: {len(self.last_simulation_result['routes']) if self.last_simulation_result else 0}"
                )

                # Save simulation results to file (this can fail without affecting visualization)
                try:
                    self._save_simulation_results(result)
                except Exception as save_error:
                    # Log but don't fail the simulation if saving fails
                    print(f"Warning: Failed to save simulation results: {save_error}")
            else:
                self.last_simulation_result = None

            return result
        except Exception as e:
            self.last_simulation_result = None

            return {"status": "error", "message": f"Simulation failed: {str(e)}", "routes": []}

    def _save_simulation_results(self, result: SimulationRunnerResult):
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
            print(
                f"Starting to save simulation results for {city_name} hour {result.simulation_hour}"
            )

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
            filename = f"{city_name}_hour{result.simulation_hour}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            print(f"Saving to: {filepath}")

            # Save to file
            result.save_to_json(filepath)

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
        return self.runner.get_network_info()

    def __repr__(self):
        return f"APIServer(network={self.runner.network is not None}, demand_inputs={len(self.runner.demand_inputs)})"
