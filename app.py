"""
Flask web application for transportation system simulation visualization.
"""
from flask import Flask, render_template, jsonify, request
import geopandas as gpd
import pandas as pd
import json
from pathlib import Path
import utils
from src.apiserver import APIServer

app = Flask(__name__)

# Load city data at startup
CITIES = {
    'Lausanne': {
        'geojson': 'data/Lausanne/Lausanne.geojson',
        'demands': 'data/Lausanne/Lausanne_time_dependent_demands.csv'
    },
    'Renens': {
        'geojson': 'data/Renens/Renens.geojson', 
        'demands': 'data/Renens/Renens_time_dependent_demands.csv'
    }
}

# Cache for loaded data
city_data_cache = {}

# Initialize APIServer for simulation
api_server = APIServer()

def load_city_data(city_name):
    """Load and cache city GeoJSON data"""
    if city_name not in city_data_cache:
        city_info = CITIES[city_name]
        gdf = gpd.read_file(city_info['geojson'])
        
        # Build network graph
        G = utils.construct_graph(city_info['geojson'])
        
        # Calculate centroids for network visualization
        gdf_with_centroids = gdf.copy()
        gdf_with_centroids = gdf_with_centroids.reset_index(drop=True)
        gdf_with_centroids['centroid'] = gdf_with_centroids.geometry.centroid
        
        # Transform centroids to WGS84 for network edges
        gdf_wgs84_centroids = gdf_with_centroids.to_crs('EPSG:4326')
        gdf_wgs84_centroids['centroid'] = gdf_wgs84_centroids.geometry.centroid
        
        # Extract network edges with coordinates
        edges = []
        for edge in G.edges():
            node1, node2 = edge
            if node1 < len(gdf_wgs84_centroids) and node2 < len(gdf_wgs84_centroids):
                centroid1 = gdf_wgs84_centroids.loc[node1, 'centroid']
                centroid2 = gdf_wgs84_centroids.loc[node2, 'centroid']
                edges.append({
                    'from': [centroid1.x, centroid1.y],
                    'to': [centroid2.x, centroid2.y]
                })
        
        # Convert to GeoJSON format for web (transform to WGS84)
        gdf_wgs84 = gdf.to_crs('EPSG:4326')
        geojson_data = json.loads(gdf_wgs84.to_json())
        
        city_data_cache[city_name] = {
            'geojson': geojson_data,
            'gdf': gdf,
            'edges': edges,
            'graph': G
        }
    
    return city_data_cache[city_name]


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/cities')
def get_cities():
    """Get list of available cities"""
    return jsonify(list(CITIES.keys()))


@app.route('/api/city/<city_name>')
def get_city_data(city_name):
    """Get GeoJSON data for a specific city"""
    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    data = load_city_data(city_name)
    return jsonify(data['geojson'])


@app.route('/api/city/<city_name>/network')
def get_city_network(city_name):
    """Get network edges for a specific city"""
    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    data = load_city_data(city_name)
    return jsonify({
        'edges': data['edges'],
        'node_count': len(data['graph'].nodes()),
        'edge_count': len(data['graph'].edges())
    })


@app.route('/api/city/<city_name>/demands')
def get_city_demands(city_name):
    """Get demand data for a specific city, hour, and type"""
    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    hour = request.args.get('hour', '0')
    demand_type = request.args.get('type', 'total')  # 'total', 'in', 'out'
    
    data = load_city_data(city_name)
    gdf = data['gdf']
    
    # Build column name based on demand type and hour
    if demand_type == 'total':
        column = str(int(hour))
    elif demand_type == 'in':
        column = f'In_{int(hour)}'
    elif demand_type == 'out':
        column = f'Out_{int(hour)}'
    else:
        return jsonify({'error': 'Invalid demand type'}), 400
    
    if column not in gdf.columns:
        return jsonify({'error': f'Column {column} not found'}), 400
    
    # Extract hex_id and demand values
    demands = {}
    for idx, row in gdf.iterrows():
        hex_id = int(row['hex_id'])
        demand_value = float(row[column])
        demands[hex_id] = demand_value
    
    # Get min and max for scaling
    demand_values = list(demands.values())
    result = {
        'demands': demands,
        'min': min(demand_values) if demand_values else 0,
        'max': max(demand_values) if demand_values else 0
    }
    
    return jsonify(result)


@app.route('/api/simulation/init/<city_name>', methods=['POST'])
def init_simulation(city_name):
    """Initialize simulation for a specific city"""
    if city_name not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    try:
        api_server.init_app(city_name)
        network_info = api_server.get_network_info()
        return jsonify({
            'status': 'success',
            'message': f'Simulation initialized for {city_name}',
            'network_info': network_info
        })
    except Exception as e:
        return jsonify({'error': f'Failed to initialize simulation: {str(e)}'}), 500


@app.route('/api/simulation/run', methods=['POST'])
def run_simulation():
    """Run the simulation with given parameters"""
    try:
        # Get input JSON from request
        input_data = request.get_json() or {}
        
        # Run simulation
        result = api_server.run_simulation(input_data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500


@app.route('/api/simulation/status')
def get_simulation_status():
    """Get current simulation status"""
    try:
        network_info = api_server.get_network_info()
        return jsonify({
            'status': 'ready' if api_server.network else 'not_initialized',
            'network_info': network_info
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get status: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
