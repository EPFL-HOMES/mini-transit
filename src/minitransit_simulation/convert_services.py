import json
from pathlib import Path
from typing import Any


# Utility functions for validating design template fields
def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_non_negative_int(value: Any) -> bool:
    return is_int(value) and value >= 0


def is_non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def validate_design_template(design_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if not isinstance(design_data, dict):
        errors.append("Top level JSON must be an object.")
        return errors

    if "fixed_route_transit" not in design_data:
        errors.append("Missing required top-level field: fixed_route_transit.")
    if "docked_bikesharing" not in design_data:
        errors.append("Missing required top-level field: docked_bikesharing.")

    fixed_route_transit = design_data.get("fixed_route_transit")
    docked_bikesharing = design_data.get("docked_bikesharing")

    if not isinstance(fixed_route_transit, list):
        errors.append("fixed_route_transit must be a list.")
        fixed_route_transit = []

    if not isinstance(docked_bikesharing, list):
        errors.append("docked_bikesharing must be a list.")
        docked_bikesharing = []

    fixed_names: set[str] = set()
    for idx, route in enumerate(fixed_route_transit, start=1):
        prefix = f"fixed_route_transit[{idx}]"
        if not isinstance(route, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        name = route.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name must be a non-empty string.")
        elif name in fixed_names:
            errors.append(f"{prefix}.name '{name}' is duplicated.")
        else:
            fixed_names.add(name)

        stops = route.get("stops")
        if not isinstance(stops, list):
            errors.append(f"{prefix}.stops must be a list.")
        else:
            if len(stops) < 2:
                errors.append(f"{prefix}.stops must contain at least two values.")
            else:
                if not all(is_int(x) for x in stops):
                    errors.append(f"{prefix}.stops must contain only integers.")
                else:
                    if len(set(stops)) < 2:
                        errors.append(f"{prefix}.stops must contain at least two different values.")
                    for i in range(len(stops) - 1):
                        if stops[i] == stops[i + 1]:
                            errors.append(
                                f"{prefix}.stops must not contain two consecutive identical values at positions {i} and {i + 1}."
                            )
                            break

        headway = route.get("headway")
        if not isinstance(headway, list):
            errors.append(f"{prefix}.headway must be a list of 13 integers.")
        else:
            if len(headway) != 13:
                errors.append(f"{prefix}.headway must contain exactly 13 values.")
            else:
                for j, value in enumerate(headway, start=1):
                    if not is_int(value):
                        errors.append(f"{prefix}.headway[{j}] must be an integer.")
                        continue
                    if value < 0:
                        errors.append(f"{prefix}.headway[{j}] must be a non-negative integer.")

        capacity = route.get("capacity")
        if not is_non_negative_int(capacity):
            errors.append(f"{prefix}.capacity must be a non-negative integer.")

        fare = route.get("fare")
        if fare is None:
            errors.append(f"{prefix}.fare must be present.")
        elif not is_non_negative_number(fare):
            errors.append(f"{prefix}.fare must be a non-negative number.")

    docked_names: set[str] = set()
    for idx, service in enumerate(docked_bikesharing, start=1):
        prefix = f"docked_bikesharing[{idx}]"
        if not isinstance(service, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        name = service.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name must be a non-empty string.")
        elif name in docked_names:
            errors.append(f"{prefix}.name '{name}' is duplicated.")
        else:
            docked_names.add(name)

        docks = service.get("docks")
        if not isinstance(docks, list):
            errors.append(f"{prefix}.docks must be a list.")
            docks = []
        else:
            if not all(is_int(x) for x in docks):
                errors.append(f"{prefix}.docks must contain only integers.")
            elif len(set(docks)) != len(docks):
                errors.append(f"{prefix}.docks must not contain duplicate values.")

        capacity = service.get("capacity")
        if not is_non_negative_int(capacity):
            errors.append(f"{prefix}.capacity must be a non-negative integer.")

        fleet_size = service.get("fleet_size")
        if not is_non_negative_int(fleet_size):
            errors.append(f"{prefix}.fleet_size must be a non-negative integer.")

        if is_non_negative_int(capacity) and is_non_negative_int(fleet_size):
            if capacity < fleet_size:
                errors.append(f"{prefix}.capacity must be greater than or equal to fleet_size.")

        initial_allocation = service.get("initial_allocation")
        valid_initial_allocation = False
        if not isinstance(initial_allocation, list):
            errors.append(f"{prefix}.initial_allocation must be a list.")
            initial_allocation = []
        else:
            if len(initial_allocation) != len(docks):
                errors.append(f"{prefix}.initial_allocation length must equal docks length.")

            if all(is_non_negative_int(x) for x in initial_allocation):
                valid_initial_allocation = True
                for alloc_index, alloc_value in enumerate(initial_allocation, start=1):
                    if alloc_value % 5 != 0:
                        errors.append(
                            f"{prefix}.initial_allocation[{alloc_index}] must be a multiple of 5."
                        )
            else:
                errors.append(f"{prefix}.initial_allocation must contain non-negative integers.")
                
        if is_int(fleet_size) and valid_initial_allocation:
            if sum(initial_allocation) != fleet_size:
                errors.append(f"{prefix}.initial_allocation sum must equal fleet_size.")

        for field_name in ("fare_base", "cutoff_min", "fare_rate"):
            value = service.get(field_name)
            if value is None:
                errors.append(f"{prefix}.{field_name} must be present.")
            elif field_name == "cutoff_min":
                if not is_non_negative_int(value):
                    errors.append(f"{prefix}.{field_name} must be a non-negative integer.")
            elif not is_non_negative_number(value):
                errors.append(f"{prefix}.{field_name} must be a non-negative number.")

    return errors


# This script converts a design template JSON file into the services and config JSON files
def convert_fixed_route_transit(design_data: dict[str, Any]) -> list[dict[str, Any]]:
    fixed_route_services: list[dict[str, Any]] = []

    for route in design_data.get("fixed_route_transit", []):
        headway_value = route.get("headway", [])
        if isinstance(headway_value, list):
            headway = [int(x) for x in headway_value]
        else:
            headway = [int(headway_value)]

        fixed_route_services.append(
            {
                "name": route.get("name", "Unknown Service"),
                "stops": route.get("stops", []),
                "headway": headway,
                "capacity": route.get("capacity", 0),
                "stopping_time": 0,
                "travel_time": 0.5,
                "base_fare": route.get("fare", 0.0),
            }
        )

    return fixed_route_services


def convert_docked_bikesharing(design_data: dict[str, Any]) -> list[dict[str, Any]]:
    ondemand_services: list[dict[str, Any]] = []

    for service in design_data.get("docked_bikesharing", []):
        docks = service.get("docks", [])
        initial_allocation = service.get("initial_allocation", [])
        dock_capacity = service.get("capacity", 0)

        vehicles = []
        for dock_index, location in enumerate(docks, start=1):
            allocated = initial_allocation[dock_index - 1] if dock_index - 1 < len(initial_allocation) else 0
            num_vehicles = allocated // 5
            for vehicle_offset in range(num_vehicles):
                vehicles.append(
                    {
                        "vehicle_id": f"bike_{len(vehicles) + 1}",
                        "initial_location": location,
                        "capacity": 5,
                    }
                )

        docking_stations = []
        for index, location in enumerate(docks, start=1):
            docking_stations.append(
                {
                    "station_id": f"dock_{index}",
                    "location": location,
                    "capacity": dock_capacity,
                }
            )

        ondemand_services.append(
            {
                "name": "Bike Share Docked",
                "type": "docked",
                "ondemand_base_fare": float(service.get("fare_base", 1.0)),
                "ondemand_time_rate_per_minute": float(service.get("fare_rate", 0.1)),
                "ondemand_base_time_cutoff_minutes": int(service.get("cutoff_min", 30)),
                "vehicles": vehicles,
                "docking_stations": docking_stations,
            }
        )

    return ondemand_services


def convert_design_to_services(input_path: Path, services_output_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        design_data = json.load(f)

    errors = validate_design_template(design_data)
    if errors:
        raise ValueError("Design template validation failed:\n" + "\n".join(errors))
    
    print("Design template validation passed. Starting conversion...")

    services_data = {
        "fixed_route_services": convert_fixed_route_transit(design_data),
        "ondemand_services": convert_docked_bikesharing(design_data),
    }

    with services_output_path.open("w", encoding="utf-8") as f:
        json.dump(services_data, f, indent=4, ensure_ascii=False)

    return {
        "services": services_data,
    }


if __name__ == "__main__":

    input_path = Path("data/Lausanne/design_template_7-10.json")
    services_output_path = Path("data/Lausanne/services_7-10.json")

    convert_design_to_services(input_path, services_output_path)
    print(f"Converted {input_path} to {services_output_path}")
