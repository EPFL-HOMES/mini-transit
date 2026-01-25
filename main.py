"""
FastAPI web application for transportation system simulation visualization.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd  # noqa: F401  # kept for parity with original code
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import src.minitransit_simulation.graph as graph  # Ensure graph module is imported
from src.other.apiserver import APIServer

app = FastAPI(title="mini-transit simulation", version="0.0.1")

# --- Static & templates (expects ./templates/index.html and optional ./static) ---
TEMPLATES_DIR = Path("templates")
STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Load city data at startup (same structure as Flask version) ---
CITIES: Dict[str, Dict[str, str]] = {
    "Lausanne": {
        "geojson": "data/Lausanne/Lausanne.geojson",
        "demands": "data/Lausanne/Lausanne_time_dependent_demands.csv",
    },
    "Renens": {
        "geojson": "data/Renens/Renens.geojson",
        "demands": "data/Renens/Renens_time_dependent_demands.csv",
    },
}

# Cache for loaded data
city_data_cache: Dict[str, Dict[str, Any]] = {}

# Initialize APIServer for simulation
api_server = APIServer()


def load_city_data(city_name: str) -> Dict[str, Any]:
    """Load and cache city GeoJSON data"""
    if city_name not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")

    if city_name not in city_data_cache:
        city_info = CITIES[city_name]

        # Read base geodata
        gdf = gpd.read_file(city_info["geojson"])

        # Build network graph
        G = graph.construct_graph(city_info["geojson"])

        # Calculate centroids for network visualization
        gdf_with_centroids = gdf.copy().reset_index(drop=True)
        gdf_with_centroids["centroid"] = gdf_with_centroids.geometry.centroid

        # Transform centroids to WGS84 for network edges
        gdf_wgs84_centroids = gdf_with_centroids.to_crs("EPSG:4326")
        gdf_wgs84_centroids["centroid"] = gdf_wgs84_centroids.geometry.centroid

        # Extract network edges with coordinates
        edges: List[Dict[str, Any]] = []
        node_count_possible = len(gdf_wgs84_centroids)
        for node1, node2 in G.edges():
            if node1 < node_count_possible and node2 < node_count_possible:
                c1 = gdf_wgs84_centroids.loc[node1, "centroid"]
                c2 = gdf_wgs84_centroids.loc[node2, "centroid"]
                edges.append({"from": [c1.x, c1.y], "to": [c2.x, c2.y]})

        # Convert to GeoJSON (WGS84) for the frontend
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        geojson_data = json.loads(gdf_wgs84.to_json())

        city_data_cache[city_name] = {
            "geojson": geojson_data,
            "gdf": gdf,  # keep original CRS & columns for demand queries
            "edges": edges,
            "graph": G,
        }

    return city_data_cache[city_name]


# --------------------- Routes ---------------------


@app.get("/", response_class=JSONResponse)
def index(request: Request):
    """Render the main page."""
    # If you have templates/templates/index.html, return it; otherwise a tiny JSON.
    if (TEMPLATES_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return JSONResponse({"message": "UI is not set up. Put index.html under ./templates/"})


@app.get("/visualization")
def visualization(request: Request):
    """Render the visualization debug page."""
    if (TEMPLATES_DIR / "visualization.html").exists():
        return templates.TemplateResponse("visualization.html", {"request": request})
    return JSONResponse({"message": "Visualization page not found"})


@app.get("/api/cities")
def get_cities():
    """Get list of available cities"""
    return list(CITIES.keys())


@app.get("/api/city/{city_name}")
def get_city_data(city_name: str):
    """Get GeoJSON data for a specific city"""
    data = load_city_data(city_name)
    return data["geojson"]


@app.get("/api/city/{city_name}/network")
def get_city_network(city_name: str):
    """Get network edges for a specific city"""
    data = load_city_data(city_name)
    return {
        "edges": data["edges"],
        "node_count": len(data["graph"].nodes()),
        "edge_count": len(data["graph"].edges()),
    }


@app.get("/api/city/{city_name}/demands")
def get_city_demands(
    city_name: str,
    hour: int = Query(0, ge=0, le=23, description="Hour of day (0-23)"),
    type: str = Query("total", pattern="^(total|in|out)$"),
):
    """
    Get demand data for a specific city, hour, and type.
    type ∈ {'total','in','out'}
    """
    data = load_city_data(city_name)
    gdf: gpd.GeoDataFrame = data["gdf"]

    # Build column name based on demand type and hour
    if type == "total":
        column = str(int(hour))
    elif type == "in":
        column = f"In_{int(hour)}"
    else:  # type == "out"
        column = f"Out_{int(hour)}"

    if column not in gdf.columns:
        raise HTTPException(status_code=400, detail=f"Column {column} not found")

    # Extract hex_id and demand values
    demands: Dict[int, float] = {}
    for _, row in gdf.iterrows():
        hex_id = int(row["hex_id"])
        demand_value = float(row[column])
        demands[hex_id] = demand_value

    demand_values = list(demands.values())
    return {
        "demands": demands,
        "min": min(demand_values) if demand_values else 0,
        "max": max(demand_values) if demand_values else 0,
    }


@app.post("/api/simulation/init/{city_name}")
def init_simulation(city_name: str):
    """Initialize simulation for a specific city"""
    if city_name not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")
    try:
        api_server.init_app(city_name)
        network_info = api_server.get_network_info()
        return {
            "status": "success",
            "message": f"Simulation initialized for {city_name}",
            "network_info": network_info,
        }
    except Exception as e:
        import traceback
        print("❌ ERROR:", e)
        traceback.print_exc()  # <-- This prints the full stack trace
        raise  # Re-raise so FastAPI logs it as a 500
        raise HTTPException(status_code=500, detail=f"Failed to initialize simulation: {e}") from e


@app.post("/api/simulation/run")
def run_simulation(input_data: Dict[str, Any] | None = None):
    """Run the simulation with given parameters"""
    try:
        from src.minitransit_simulation.simulation_runner import SimulationRunnerInput
        
        input_payload = input_data or {}
        # Convert dictionary to SimulationRunnerInput dataclass
        hour = input_payload.get("hour", 8)
        simulation_input = SimulationRunnerInput(hour=hour)
        result = api_server.run_simulation(simulation_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}") from e


@app.get("/api/simulation/status")
def get_simulation_status():
    """Get current simulation status"""
    try:
        network_info = api_server.get_network_info()
        return {
            "status": "ready" if getattr(api_server, "network", None) else "not_initialized",
            "network_info": network_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}") from e


@app.get("/api/visualization/data")
def get_visualization_data(city_name: str = Query(..., description="City name")):
    """Get visualization data including hexes, services, vehicles, and units"""
    if city_name not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")

    if api_server.runner.network is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    try:
        # Get city data
        city_data = load_city_data(city_name)
        gdf_wgs84 = city_data["gdf"].to_crs("EPSG:4326")

        # Create hex lookup (hex_id -> centroid coordinates)
        hex_coords = {}
        for idx, row in gdf_wgs84.iterrows():
            hex_id = int(row["hex_id"])
            centroid = row.geometry.centroid
            hex_coords[hex_id] = [centroid.y, centroid.x]  # [lat, lng] for Leaflet

        # Get fixed route services
        services_data = []
        for i, service in enumerate(api_server.runner.network.services):
            from src.minitransit_simulation.services.fixedroute import FixedRouteService

            if isinstance(service, FixedRouteService):
                # Get stop coordinates
                stops_coords = []
                for stop in service.stops:
                    if stop.hex_id in hex_coords:
                        stops_coords.append(hex_coords[stop.hex_id])

                # Get vehicles with their timetables
                vehicles_data = []
                for vehicle in service.vehicles:
                    vehicle_stops = []
                    # OrderedDict preserves insertion order, so items() gives stops in visit order
                    # This is important for reverse direction vehicles - they visit stops in reverse order
                    for stop_idx, (arrival, departure) in vehicle.timetable.items():
                        stop_hex = service.stops[stop_idx]
                        if stop_hex.hex_id in hex_coords:
                            vehicle_stops.append(
                                {
                                    "hex_id": stop_hex.hex_id,
                                    "coords": hex_coords[stop_hex.hex_id],
                                    "arrival_time": arrival.isoformat(),
                                    "departure_time": departure.isoformat(),
                                    "stop_index": stop_idx,  # Add for debugging
                                }
                            )
                    # DO NOT sort - OrderedDict.items() already gives stops in visit order
                    # Sorting would break reverse direction vehicles
                    vehicles_data.append(
                        {
                            "timetable": vehicle_stops,
                            "capacity": vehicle.capacity,
                            "current_load": vehicle.current_load,
                        }
                    )

                services_data.append(
                    {
                        "id": i,
                        "name": f"S{i}",
                        "service_name": service.name,
                        "stops": stops_coords,
                        "stop_hex_ids": [stop.hex_id for stop in service.stops],
                        "vehicles": vehicles_data,
                        "color": _get_service_color(i),
                    }
                )

        return {
            "hexes": hex_coords,
            "services": services_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get visualization data: {e}") from e


@app.get("/api/visualization/simulation-state")
def get_simulation_state(
    city_name: str = Query(..., description="City name"),
    current_time: str = Query(..., description="Current simulation time (ISO format)"),
):
    """Get simulation state at a specific time including units and event queue"""
    print(
        f"DEBUG: get_simulation_state called with city_name={city_name}, current_time={current_time}"
    )
    print(f"DEBUG: api_server.runner.network is None: {api_server.runner.network is None}")
    print(
        f"DEBUG: api_server.last_simulation_result is None: {api_server.last_simulation_result is None}"
    )

    if city_name not in CITIES:
        print(f"DEBUG: City {city_name} not found in CITIES")
        raise HTTPException(status_code=404, detail="City not found")

    if api_server.runner.network is None:
        print("DEBUG: Network not initialized")
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    if not api_server.last_simulation_result:
        print("DEBUG: No simulation results available")
        raise HTTPException(
            status_code=400,
            detail="No simulation results available. Please run the simulation first by clicking 'Run Simulation' button.",
        )

    try:
        from datetime import datetime

        # Handle ISO format with or without timezone
        # Remove 'Z' and milliseconds if present, then parse
        time_str = current_time.replace("Z", "").strip()
        # Remove milliseconds if present (e.g., .000)
        if "." in time_str and "T" in time_str:
            # Split on T, then on . to remove milliseconds
            parts = time_str.split("T")
            if len(parts) == 2:
                date_part = parts[0]
                time_part = parts[1].split(".")[0]  # Remove milliseconds
                time_str = f"{date_part}T{time_part}"

        try:
            # Try parsing as ISO format
            current_dt = datetime.fromisoformat(time_str)
        except (ValueError, AttributeError) as e:
            # Fallback: try parsing with common format
            try:
                current_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError as parse_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid time format: {current_time}. Error: {str(parse_error)}",
                )

        # Ensure the date is 2024-01-01 to match simulation actions
        # Actions are created with datetime(2024, 1, 1, hour, ...)
        if current_dt.date() != datetime(2024, 1, 1).date():
            print(f"DEBUG: Adjusting date from {current_dt.date()} to 2024-01-01")
            current_dt = current_dt.replace(year=2024, month=1, day=1)

        # Get city data for coordinates
        city_data = load_city_data(city_name)
        gdf_wgs84 = city_data["gdf"].to_crs("EPSG:4326")
        hex_coords = {}
        for idx, row in gdf_wgs84.iterrows():
            hex_id = int(row["hex_id"])
            centroid = row.geometry.centroid
            hex_coords[hex_id] = [centroid.y, centroid.x]

        # Get units from routes
        units_data = []
        routes = api_server.last_simulation_result["routes"]

        # Normalize current_dt once at the start
        if current_dt.tzinfo is not None:
            current_dt = current_dt.replace(tzinfo=None)

        current_dt = current_dt + timedelta(hours=1)

        print(f"DEBUG: Processing {len(routes)} routes for time {current_dt}")

        for route_idx, route in enumerate(routes):
            # Skip if route has no actions
            if not hasattr(route, "actions") or not route.actions:
                print(f"DEBUG: Route {route_idx} has no actions")
                continue

            # Check if current time is within route's time range (optional check for efficiency)
            try:
                route_start = min(
                    action.start_time for action in route.actions if hasattr(action, "start_time")
                )
                route_end_times = [
                    action.end_time
                    for action in route.actions
                    if hasattr(action, "end_time") and action.end_time
                ]
                if route_end_times:
                    route_end = max(route_end_times)
                else:
                    route_end = route_start

                # Normalize route times
                if route_start.tzinfo is not None:
                    route_start = route_start.replace(tzinfo=None)
                if route_end.tzinfo is not None:
                    route_end = route_end.replace(tzinfo=None)

                # Only skip if clearly outside range (with small buffer)
                if current_dt < route_start or current_dt > route_end:
                    print(
                        f"DEBUG: Route {route_idx} time range [{route_start}, {route_end}] doesn't include {current_dt}"
                    )
                    continue
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Error checking route {route_idx} time range: {e}")
                # Continue anyway to check individual actions

            # Find which action the unit is currently performing
            current_action = None
            for action in route.actions:
                # Skip if action doesn't have required attributes
                if not hasattr(action, "start_time"):
                    continue

                action_start = action.start_time
                # Normalize timezones - make both timezone-naive for comparison
                if action_start.tzinfo is not None:
                    action_start = action_start.replace(tzinfo=None)

                action_end = (
                    action.end_time
                    if (hasattr(action, "end_time") and action.end_time)
                    else action_start
                )
                if action_end.tzinfo is not None:
                    action_end = action_end.replace(tzinfo=None)

                # Use <= for end_time to include the exact end time
                if current_dt >= action_start and current_dt <= action_end:
                    current_action = action
                    print(
                        f"DEBUG: Route {route_idx} - found action {action.__class__.__name__} at time {current_dt} (range: {action_start} to {action_end})"
                    )
                    break

            if not current_action:
                print(f"DEBUG: Route {route_idx} - no action found for time {current_dt}")

            if current_action:
                from src.minitransit_simulation.actions.ride import Ride
                from src.minitransit_simulation.actions.wait import Wait
                from src.minitransit_simulation.actions.walk import Walk

                try:
                    unit_info = {
                        "id": route_idx,
                        "unit": getattr(route, "unit", 0),
                        "action_type": current_action.__class__.__name__,
                        "creation_time": (
                            route.actions[0].start_time.isoformat()
                            if route.actions and hasattr(route.actions[0], "start_time")
                            else None
                        ),
                    }

                    if isinstance(current_action, Wait):
                        if hasattr(current_action, "location") and hasattr(
                            current_action.location, "hex_id"
                        ):
                            hex_id = current_action.location.hex_id
                            coords = hex_coords.get(hex_id)
                            if coords:
                                unit_info["location"] = coords
                                unit_info["hex_id"] = hex_id
                            else:
                                print(
                                    f"DEBUG: Wait action - hex_id {hex_id} not found in hex_coords"
                                )
                    elif isinstance(current_action, Walk):
                        # Interpolate position during walk
                        if (
                            hasattr(current_action, "start_time")
                            and hasattr(current_action, "end_time")
                            and current_action.end_time
                        ):
                            total_duration = (
                                current_action.end_time - current_action.start_time
                            ).total_seconds()
                            if total_duration > 0:
                                progress = (
                                    current_dt - current_action.start_time
                                ).total_seconds() / total_duration
                                progress = max(0, min(1, progress))  # Clamp between 0 and 1

                                if hasattr(current_action, "start_hex") and hasattr(
                                    current_action, "end_hex"
                                ):
                                    start_hex_id = current_action.start_hex.hex_id
                                    end_hex_id = current_action.end_hex.hex_id
                                    start_coords = hex_coords.get(start_hex_id)
                                    end_coords = hex_coords.get(end_hex_id)
                                    if start_coords and end_coords:
                                        lat = (
                                            start_coords[0]
                                            + (end_coords[0] - start_coords[0]) * progress
                                        )
                                        lng = (
                                            start_coords[1]
                                            + (end_coords[1] - start_coords[1]) * progress
                                        )
                                        unit_info["location"] = [lat, lng]
                                        unit_info["start_hex_id"] = start_hex_id
                                        unit_info["end_hex_id"] = end_hex_id
                                    else:
                                        print(
                                            f"DEBUG: Walk action - missing coords: start_hex={start_hex_id} (found: {start_coords is not None}), end_hex={end_hex_id} (found: {end_coords is not None})"
                                        )
                                else:
                                    print(
                                        f"DEBUG: Walk action - missing start_hex or end_hex attributes"
                                    )
                            else:
                                print(f"DEBUG: Walk action - invalid duration: {total_duration}")
                        else:
                            print(f"DEBUG: Walk action - missing time attributes")
                    elif isinstance(current_action, Ride):
                        # Unit is on a vehicle - calculate vehicle position at current time
                        vehicle_position = None
                        if hasattr(current_action, "vehicle") and current_action.vehicle:
                            vehicle = current_action.vehicle
                            service = current_action.service

                            # Calculate vehicle position using linear interpolation
                            if hasattr(vehicle, "timetable") and vehicle.timetable:
                                # Convert timetable to list of (stop_idx, arrival, departure, coords) tuples
                                timetable_list = []
                                for stop_idx, (arrival, departure) in vehicle.timetable.items():
                                    stop_hex = service.stops[stop_idx]
                                    if stop_hex.hex_id in hex_coords:
                                        timetable_list.append(
                                            {
                                                "stop_idx": stop_idx,
                                                "arrival": (
                                                    arrival.replace(tzinfo=None)
                                                    if arrival.tzinfo
                                                    else arrival
                                                ),
                                                "departure": (
                                                    departure.replace(tzinfo=None)
                                                    if departure.tzinfo
                                                    else departure
                                                ),
                                                "coords": hex_coords[stop_hex.hex_id],
                                            }
                                        )

                                # Sort by arrival time to get visit order
                                timetable_list.sort(key=lambda x: x["arrival"])

                                # Check if vehicle is stopped at any stop
                                for i, stop in enumerate(timetable_list):
                                    if (
                                        current_dt >= stop["arrival"]
                                        and current_dt <= stop["departure"]
                                    ):
                                        vehicle_position = stop["coords"]
                                        break

                                # If not stopped, check if moving between stops
                                if vehicle_position is None:
                                    for i in range(len(timetable_list) - 1):
                                        stop1 = timetable_list[i]
                                        stop2 = timetable_list[i + 1]

                                        if (
                                            current_dt >= stop1["departure"]
                                            and current_dt <= stop2["arrival"]
                                        ):
                                            # Calculate progress
                                            total_time = (
                                                stop2["arrival"] - stop1["departure"]
                                            ).total_seconds()
                                            elapsed = (
                                                current_dt - stop1["departure"]
                                            ).total_seconds()
                                            progress = (
                                                max(0, min(1, elapsed / total_time))
                                                if total_time > 0
                                                else 0
                                            )

                                            # Interpolate position
                                            lat = (
                                                stop1["coords"][0]
                                                + (stop2["coords"][0] - stop1["coords"][0])
                                                * progress
                                            )
                                            lng = (
                                                stop1["coords"][1]
                                                + (stop2["coords"][1] - stop1["coords"][1])
                                                * progress
                                            )
                                            vehicle_position = [lat, lng]
                                            break

                        unit_info["location"] = vehicle_position
                        if hasattr(current_action, "start_hex"):
                            unit_info["start_hex_id"] = current_action.start_hex.hex_id
                        if hasattr(current_action, "end_hex"):
                            unit_info["end_hex_id"] = current_action.end_hex.hex_id
                        unit_info["service_id"] = None
                        # Find service ID
                        if hasattr(current_action, "service") and current_action.service:
                            for i, service in enumerate(api_server.runner.network.services):
                                if service == current_action.service:
                                    unit_info["service_id"] = i
                                    break

                    # Only add if we have location
                    if unit_info.get("location") is not None:
                        units_data.append(unit_info)
                        print(
                            f"DEBUG: Added unit {route_idx} at location {unit_info.get('location')} with action {unit_info.get('action_type')}"
                        )
                    else:
                        print(
                            f"DEBUG: Skipping unit {route_idx} - no location (action: {unit_info.get('action_type')})"
                        )
                except Exception as e:
                    # Skip this unit if there's an error processing it
                    import traceback

                    print(f"Error processing unit {route_idx}: {e}")
                    traceback.print_exc()
                    continue

        print(
            f"DEBUG: Returning {len(units_data)} units with locations out of {len(routes)} total routes"
        )
        if len(units_data) == 0 and len(routes) > 0:
            # Debug: show time ranges of first few routes
            print(f"DEBUG: No units found. Sample route time ranges:")
            for i, route in enumerate(routes[:3]):
                if hasattr(route, "actions") and route.actions:
                    try:
                        route_start = min(
                            action.start_time
                            for action in route.actions
                            if hasattr(action, "start_time")
                        )
                        route_end_times = [
                            action.end_time
                            for action in route.actions
                            if hasattr(action, "end_time") and action.end_time
                        ]
                        route_end = max(route_end_times) if route_end_times else route_start
                        if route_start.tzinfo is not None:
                            route_start = route_start.replace(tzinfo=None)
                        if route_end.tzinfo is not None:
                            route_end = route_end.replace(tzinfo=None)
                        print(
                            f"  Route {i}: {route_start} to {route_end}, current_dt: {current_dt}"
                        )
                    except Exception as e:
                        print(f"  Route {i}: Error getting time range: {e}")

        # Build event queue from routes (simplified representation)
        event_queue_data = []
        for route_idx, route in enumerate(routes):
            if not hasattr(route, "actions") or not route.actions:
                continue
            for action_idx, action in enumerate(route.actions):
                if not hasattr(action, "start_time"):
                    continue
                try:
                    event_queue_data.append(
                        {
                            "route_id": route_idx,
                            "action_index": action_idx,
                            "type": action.__class__.__name__,
                            "start_time": action.start_time.isoformat(),
                            "end_time": (
                                action.end_time.isoformat()
                                if (hasattr(action, "end_time") and action.end_time)
                                else None
                            ),
                            "unit": getattr(action, "unit", 0),
                        }
                    )
                except Exception as e:
                    # Skip this event if there's an error
                    continue

        # Sort by start time
        event_queue_data.sort(key=lambda x: x["start_time"])

        return {
            "current_time": current_time,
            "units": units_data,
            "event_queue": event_queue_data[:50],  # Limit to 50 for display
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback

        error_details = traceback.format_exc()
        print(f"Error in get_simulation_state: {e}")
        print(error_details)
        raise HTTPException(
            status_code=500, detail=f"Failed to get simulation state: {str(e)}"
        ) from e


@app.get("/api/visualization/unit-route")
def get_unit_route(
    city_name: str = Query(..., description="City name"),
    unit_id: int = Query(..., description="Unit ID (route index)"),
):
    """Get full route details for a specific unit"""
    if city_name not in CITIES:
        raise HTTPException(status_code=404, detail="City not found")

    if api_server.runner.network is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")

    if not api_server.last_simulation_result:
        raise HTTPException(
            status_code=400,
            detail="No simulation results available. Please run the simulation first.",
        )

    try:
        # Get city data for coordinates
        city_data = load_city_data(city_name)
        gdf_wgs84 = city_data["gdf"].to_crs("EPSG:4326")
        hex_coords = {}
        for idx, row in gdf_wgs84.iterrows():
            hex_id = int(row["hex_id"])
            centroid = row.geometry.centroid
            hex_coords[hex_id] = [centroid.y, centroid.x]  # [lat, lng]

        routes = api_server.last_simulation_result["routes"]

        if unit_id < 0 or unit_id >= len(routes):
            raise HTTPException(status_code=404, detail=f"Unit {unit_id} not found")

        route = routes[unit_id]

        # Build route steps with coordinates
        route_steps = []
        from src.minitransit_simulation.actions.ride import Ride
        from src.minitransit_simulation.actions.wait import Wait
        from src.minitransit_simulation.actions.walk import Walk

        for action_idx, action in enumerate(route.actions):
            step_info = {
                "index": action_idx,
                "type": action.__class__.__name__,
                "start_time": (
                    action.start_time.isoformat() if hasattr(action, "start_time") else None
                ),
                "end_time": (
                    action.end_time.isoformat()
                    if (hasattr(action, "end_time") and action.end_time)
                    else None
                ),
                "duration_minutes": (
                    action.duration_minutes
                    if (hasattr(action, "duration_minutes") and action.end_time)
                    else None
                ),
            }

            # Add location/coordinates based on action type
            if isinstance(action, Wait):
                if hasattr(action, "location") and hasattr(action.location, "hex_id"):
                    hex_id = action.location.hex_id
                    step_info["hex_id"] = hex_id
                    step_info["coords"] = hex_coords.get(hex_id)
            elif isinstance(action, Walk):
                if hasattr(action, "start_hex") and hasattr(action, "end_hex"):
                    start_hex_id = action.start_hex.hex_id
                    end_hex_id = action.end_hex.hex_id
                    step_info["start_hex_id"] = start_hex_id
                    step_info["end_hex_id"] = end_hex_id
                    step_info["start_coords"] = hex_coords.get(start_hex_id)
                    step_info["end_coords"] = hex_coords.get(end_hex_id)
            elif isinstance(action, Ride):
                if hasattr(action, "start_hex") and hasattr(action, "end_hex"):
                    start_hex_id = action.start_hex.hex_id
                    end_hex_id = action.end_hex.hex_id
                    step_info["start_hex_id"] = start_hex_id
                    step_info["end_hex_id"] = end_hex_id
                    step_info["start_coords"] = hex_coords.get(start_hex_id)
                    step_info["end_coords"] = hex_coords.get(end_hex_id)

                    # Add service info
                    if hasattr(action, "service") and action.service:
                        for i, service in enumerate(api_server.runner.network.services):
                            if service == action.service:
                                step_info["service_id"] = i
                                step_info["service_name"] = service.name
                                break

                    # Add vehicle timetable for visualization
                    if hasattr(action, "vehicle") and action.vehicle:
                        vehicle = action.vehicle
                        vehicle_stops = []
                        for stop_idx, (arrival, departure) in vehicle.timetable.items():
                            stop_hex = action.service.stops[stop_idx]
                            if stop_hex.hex_id in hex_coords:
                                vehicle_stops.append(
                                    {
                                        "hex_id": stop_hex.hex_id,
                                        "coords": hex_coords[stop_hex.hex_id],
                                        "arrival_time": arrival.isoformat(),
                                        "departure_time": departure.isoformat(),
                                        "stop_index": stop_idx,
                                    }
                                )
                        # Sort by arrival time to get visit order
                        vehicle_stops.sort(key=lambda x: x["arrival_time"])
                        step_info["vehicle_stops"] = vehicle_stops

            route_steps.append(step_info)

        return {
            "unit_id": unit_id,
            "unit_size": route.unit,
            "total_time_minutes": (
                route.time_taken_minutes if hasattr(route, "time_taken_minutes") else None
            ),
            "total_fare": route.total_fare if hasattr(route, "total_fare") else None,
            "creation_time": (
                route.actions[0].start_time.isoformat()
                if route.actions and hasattr(route.actions[0], "start_time")
                else None
            ),
            "steps": route_steps,
        }
    except Exception as e:
        import traceback

        print(f"Error in get_unit_route: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get unit route: {e}") from e


def _get_service_color(index: int) -> str:
    """Generate a distinct color for each service"""
    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FF8000",
        "#8000FF",
        "#FF0080",
        "#80FF00",
        "#0080FF",
        "#FF8080",
        "#80FF80",
        "#8080FF",
        "#FFFF80",
        "#FF80FF",
        "#80FFFF",
    ]
    return colors[index % len(colors)]


# --------------- Dev server ---------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
