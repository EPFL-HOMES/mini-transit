"""
APIServer class that talks with frontend and starts the simulation.
"""

import pandas as pd
import json
import sys
import os

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.network import Network
from src.demand import Demand
from src.hex import Hex
from src.models import FixedRouteServiceModel
from typing import List

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
        
        # Initialize network
        self.network = Network(geojson_path)
        
        # Load and parse demands from CSV
        self.demands = self._load_demands(demands_path)
        
        print(f"Initialized {city_name}: {len(self.demands)} demands, {len(self.network.graph.nodes())} hexagons")

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
            from datetime import datetime, timedelta
            
            # Extract simulation parameters
            simulation_hour = input_json.get('hour', 8)  # Default to hour 8 if not specified
            
            # Clear previous routes
            self.network.clear()
            
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
            
            routes = []
            simulation_start_time = datetime.now()
            
            # Process each demand using network optimization
            for demand in filtered_demands:
                # Get optimal route using network
                route = self.network.get_optimal_route(demand)
                
                if route is not None:
                    # Push route to network
                    self.network.push_route(route)
                    routes.append(route)
            
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
                            "start_time": action.start_time.isoformat(),
                            "end_time": action.end_time.isoformat() if action.end_time else None,
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
                    elif isinstance(action, dict):
                        # Dictionary action
                        action_data = {
                            "type": action.get('type', 'Unknown'),
                            "start_time": action['start_time'].isoformat(),
                            "end_time": action['end_time'].isoformat() if action.get('end_time') else None,
                            "duration_minutes": (action['end_time'] - action['start_time']).total_seconds() / 60.0
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
                "demands_processed": len(filtered_demands),
                "routes_generated": len(routes),
                "total_units": sum(route.unit for route in routes),
                "total_time_minutes": sum(route.time_taken_minutes for route in routes),
                "total_fare": total_fare,
                "average_fare": average_fare,
                "network_routes_taken": len(self.network.routes_taken)
            }
            
            # Save simulation results to file
            self._save_simulation_results(result, simulation_hour)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Simulation failed: {str(e)}",
                "routes": []
            }
    
    def _save_simulation_results(self, result: dict, simulation_hour: int):
        """
        Save simulation results to a JSON file.
        
        Args:
            result (dict): The simulation result dictionary.
            simulation_hour (int): The hour of the simulation.
        """
        try:
            from datetime import datetime
            import traceback
            
            print(f"Starting to save simulation results for {self.city_name} hour {simulation_hour}")
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'simulation_results')
            print(f"Results directory: {results_dir}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.city_name}_hour{simulation_hour}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            print(f"Saving to: {filepath}")
            
            # Save to file
            with open(filepath, 'w') as f:
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
            "demands": len(self.demands)
        }
    
    def __repr__(self):
        return f"APIServer(network={self.network is not None}, demands={len(self.demands)})"
