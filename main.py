"""
FastAPI web application for transportation system simulation visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd  # noqa: F401  # kept for parity with original code
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import utils
from src.apiserver import APIServer

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
        G = utils.construct_graph(city_info["geojson"])

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
        raise HTTPException(status_code=500, detail=f"Failed to initialize simulation: {e}") from e


@app.post("/api/simulation/run")
def run_simulation(input_data: Dict[str, Any] | None = None):
    """Run the simulation with given parameters"""
    try:
        input_payload = input_data or {}
        result = api_server.run_simulation(input_payload)
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


# --------------- Dev server ---------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
