"""
Unified loader for both fixed route and on-demand services from JSON.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List

from ..primitives.hex import Hex
from .fixedroute import FixedRouteService
from .ondemand import DockingStation, OnDemandRouteServiceDocked, OnDemandVehicle


def load_services_from_json(json_path: str, network) -> List:
    """
    Load both fixed-route and on-demand services from JSON file.
    Args:
        json_path (str): Path to the JSON file containing services.
        network (Network): The network to which the services will be added.
    Returns:
        List: List of all service objects (both fixed route and on-demand).
    """
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Services file not found: {json_path}. Skipping service loading.")
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        services = []

        # Load fixed route services
        fixed_route_services_data = data.get("fixed_route_services", [])
        services.extend(_load_fixed_route_services(fixed_route_services_data, network))

        # Load on-demand services
        ondemand_services_data = data.get("ondemand_services", [])
        services.extend(_load_ondemand_services(ondemand_services_data, network))

        return services

    except Exception as e:
        print(f"Error loading services from {json_path}: {e}")
        import traceback

        traceback.print_exc()
        return []


def _load_fixed_route_services(services_data: List, network) -> List:
    """Load fixed route services from JSON data."""
    from .fixedroute import FixedRouteService

    services = []

    # Default values for service parameters
    default_stopping_time_minutes = 1
    default_travel_time_minutes = 2
    default_bidirectional = True

    # Create frequency period for the entire day
    day_start = datetime(2024, 1, 1, 0, 0, 0)
    day_end = datetime(2024, 1, 1, 23, 59, 59)

    for service_info in services_data:
        name = service_info.get("name", "Unknown Service")
        stops_hex_ids = service_info.get("stops", [])
        frequency_minutes = service_info.get("frequency", 10)
        capacity = float(service_info.get("capacity", 50))

        stopping_time_minutes = service_info.get("stopping_time", default_stopping_time_minutes)
        travel_time_minutes = service_info.get("travel_time", default_travel_time_minutes)

        stopping_time = timedelta(minutes=stopping_time_minutes)
        travel_time = timedelta(minutes=travel_time_minutes)

        stops = [Hex(hex_id) for hex_id in stops_hex_ids]

        if not stops:
            print(f"Warning: Fixed route service '{name}' has no stops. Skipping.")
            continue

        frequency_timedelta = timedelta(minutes=frequency_minutes)
        freq_period = [(day_start, day_end, frequency_timedelta)]

        fixed_route_service = FixedRouteService(
            name=name,
            stops=stops,
            capacity=capacity,
            stopping_time=stopping_time,
            travel_time=travel_time,
            network=network,
            freq_period=freq_period,
            bidirectional=default_bidirectional,
        )

        services.append(fixed_route_service)

    return services


def _load_ondemand_services(services_data: List, network) -> List:
    """Load on-demand services from JSON data."""
    services = []

    for service_info in services_data:
        name = service_info.get("name", "Unknown On-Demand Service")
        service_type = service_info.get("type", "docked")  # "docked" or "dockless"
        capacity = float(service_info.get("capacity", 1))

        # Load vehicles
        vehicles_data = service_info.get("vehicles", [])
        vehicles = []
        for vehicle_info in vehicles_data:
            vehicle_id = vehicle_info.get("vehicle_id", f"vehicle_{len(vehicles)}")
            initial_location_hex_id = vehicle_info.get("initial_location")
            vehicle_capacity = vehicle_info.get("capacity", 1)

            if initial_location_hex_id is None:
                print(f"Warning: Vehicle {vehicle_id} has no initial_location. Skipping.")
                continue

            vehicle = OnDemandVehicle(
                vehicle_id=vehicle_id,
                current_location=Hex(initial_location_hex_id),
                capacity=vehicle_capacity,
            )
            vehicles.append(vehicle)

        if not vehicles:
            print(f"Warning: On-demand service '{name}' has no vehicles. Skipping.")
            continue

        # Create service based on type
        if service_type == "docked":
            # Load docking stations
            docking_stations_data = service_info.get("docking_stations", [])
            docking_stations = []

            for dock_info in docking_stations_data:
                station_id = dock_info.get("station_id", f"dock_{len(docking_stations)}")
                location_hex_id = dock_info.get("location")
                dock_capacity = dock_info.get("capacity", 10)

                if location_hex_id is None:
                    print(f"Warning: Docking station {station_id} has no location. Skipping.")
                    continue

                docking_station = DockingStation(
                    station_id=station_id,
                    location=Hex(location_hex_id),
                    capacity=dock_capacity,
                )
                docking_stations.append(docking_station)

            # Initialize vehicles at docking stations (distribute evenly)
            for i, vehicle in enumerate(vehicles):
                if docking_stations:
                    # Place vehicle at a docking station (round-robin)
                    dock_index = i % len(docking_stations)
                    try:
                        docking_stations[dock_index].dock_vehicle(vehicle)
                        vehicle.current_location = docking_stations[dock_index].location
                    except Exception as e:
                        # If docking station is full, just set location without docking
                        print(f"Warning: Could not dock vehicle {vehicle.vehicle_id} at station {docking_stations[dock_index].station_id}: {e}")
                        vehicle.current_location = docking_stations[dock_index].location

            ondemand_service = OnDemandRouteServiceDocked(
                name=name,
                vehicles=vehicles,
                capacity=capacity,
                network=network,
                docking_stations=docking_stations,
            )

            services.append(ondemand_service)

        elif service_type == "dockless":
            from .ondemand import OnDemandRouteServiceDockless

            ondemand_service = OnDemandRouteServiceDockless(
                name=name,
                vehicles=vehicles,
                capacity=capacity,
                network=network,
            )

            services.append(ondemand_service)
        else:
            print(f"Warning: Unknown on-demand service type '{service_type}'. Skipping.")

    return services