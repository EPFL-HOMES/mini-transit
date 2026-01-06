import geopandas as gpd
import networkx as nx


def find_adjacencies(gdf):
    """
    Finds pairs of adjacent polygons in a GeoDataFrame based on touching boundaries.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries.

    Returns:
        list: A list of tuples, where each tuple contains the indices of two adjacent polygons.
    """
    # Build spatial index
    gdf_sindex = gdf.sindex
    adjacency_list = []
    # Use tqdm for progress bar if you have many polygons
    iterator = gdf.iterrows()

    for index, polygon in iterator:
        geom = polygon.geometry
        # Find potential neighbors using spatial index (intersects bounds)
        possible_matches_index = list(gdf_sindex.intersection(geom.bounds))
        # Filter for precise touch condition
        # Check index != index to avoid self-adjacency
        # Check index < neighbor_index to add each pair only once
        precise_matches = [
            neighbor_index
            for neighbor_index in possible_matches_index
            if index < neighbor_index and geom.touches(gdf.geometry.loc[neighbor_index])
        ]
        for neighbor_index in precise_matches:
            adjacency_list.append(tuple(sorted((int(index), int(neighbor_index)))))

    # Remove potential duplicates if any slip through (though index < neighbor_index should prevent it)
    unique_adjacencies = sorted(list(set(adjacency_list)))
    return unique_adjacencies


def construct_graph(network):
    """
    Construct a networkx graph from a geojson file

    Args:
        network (str): network file name (inlcuding .geojson)

    Returns:
        G (networkx object): graph
    """

    # --- Load data ---
    geo_df = gpd.read_file(network)

    # --- Calculate Centroids ---
    # Make sure the index is unique and suitable for node IDs
    geo_df = geo_df.reset_index(drop=True)  # Reset index to ensure simple 0, 1, 2... IDs
    geo_df["centroid"] = geo_df.geometry.centroid

    # --- Find Adjacencies ---
    # Note: This can be slow for very large numbers of hexagons
    adjacency_list = find_adjacencies(geo_df)  # Returns pairs of indices

    # --- Build NetworkX Graph ---
    G = nx.Graph()
    # Add nodes (using the new DataFrame index)
    # Add edges based on adjacency
    G.add_edges_from(adjacency_list)

    return G
