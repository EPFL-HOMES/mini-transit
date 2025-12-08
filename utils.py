import contextily as cx

# Optional: Configure hvplot to use Bokeh backend explicitly (usually default)
# hvplot.extension('bokeh')
import geopandas as gpd
import hvplot.networkx

# Make sure to import hvplot.pandas to activate the .hvplot accessor
# This will also implicitly load hvplot.geopandas
# Make sure to import hvplot.pandas to activate the .hvplot accessor
# This will also implicitly load hvplot.geopandas
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import networkx as nx
import panel as pn  # Optional: Often used with hvPlot/HoloViews for layout/widgets
from matplotlib.colors import Normalize


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


def plot_network(G, geo_df, hour, selected_hex=None, demand_type="total", static=True):
    """
    Plots the network hourly demand in selected zones

    Args:
        G (networkx object): Network graph
        geo_df (geopandas DataFrame): Network data
        hour (int or str): Selected hour (0-23)
        selected_hex (list, optional): List of selected zone ids. Defaults to None.
        demand_type (str, optional): Demand to plot
        static (boolean): If False, create an interacitve plot using hvPlot

    Returns:
        final_plot: plot object
    """

    assert int(hour) in range(0, 24), "hour must be between 0 and 23"
    assert demand_type in [
        "total",
        "depart",
        "arrive",
    ], 'demand_type must be "total" or "depart" or "arrive"'

    if selected_hex is not None:
        geo_df = geo_df[geo_df["hex_id"].isin(selected_hex)]

    # Ensure the hour column name is a string, as expected by hvplot if df columns are strings
    hr = str(int(hour))
    if demand_type == "depart":
        hr = "Out_{}".format(hr)
    elif demand_type == "arrive":
        hr = "In_{}".format(hr)
    if hr not in geo_df.columns:
        raise ValueError(
            f"Column '{hr}' not found in GeoDataFrame. Available columns: {geo_df.columns.tolist()}"
        )

    # --- Calculate Centroids ---
    # Make sure the index is unique and suitable for node IDs
    geo_df = geo_df.reset_index(drop=True)  # Reset index to ensure simple 0, 1, 2... IDs
    geo_df["centroid"] = geo_df.geometry.centroid

    if not static:
        # Get node positions for plotting {node_id: (x, y)}
        node_positions = {idx: (point.x, point.y) for idx, point in geo_df["centroid"].items()}

        # --- hvPlot Parameters ---
        plot_width = 700  # Width of the plot in pixels
        plot_height = 700  # Height of the plot in pixels
        hex_alpha = 0.65  # Transparency
        hex_edgecolor = "black"
        # Note: facecolor is handled by 'c' and 'cmap'
        hex_linewidth = 0.5
        # Labels are replaced by hover tooltips for better interactivity
        hover_cols = ["hex_id", hr]  # Columns to show on hover
        basemap_source = "OSM"  # OpenStreetMap. Options: 'OSM', 'CartoLight', 'CartoDark', 'StamenTerrain', 'StamenToner', etc.
        plot_title = f"{network} {demand_type} demands at {hour}:00 hr"

        # Create the interactive plot using .hvplot()
        # It's often useful to explicitly state geo=True
        hex_plot = geo_df.hvplot(
            geo=True,  # Indicate it's a geospatial plot
            c=hr,  # Color polygons based on the 'hr' column value
            cmap="Reds",  # Colormap for the values
            alpha=hex_alpha,  # Polygon fill transparency
            line_color=hex_edgecolor,  # Polygon border color
            line_width=hex_linewidth,  # Polygon border width
            hover_cols=hover_cols,  # Add tooltips showing hex_id and the value
            tiles=basemap_source,  # Add the specified basemap
            frame_width=plot_width,  # Set plot width
            frame_height=plot_height,  # Set plot height
            title=plot_title,  # Set the plot title
            legend=False,  # Legend is usually not needed for choropleth
            colorbar=True,  # Show a color bar for the 'c' variable
            # xlim=(-10.0, 20.0),      # Optional: Set x-limits if needed
            # ylim=(30.0, 60.0),       # Optional: Set y-limits if needed
            crs="EPSG:3857",  # Optional: Specify CRS if needed, often inferred
            # project=True           # Optional: Reproject data to match tile CRS if necessary (usually Web Mercator)
        )

        # 2. Network Graph Plot
        # Check if graph has nodes/edges before plotting
        if G.nodes and G.edges:
            network_plot = hvplot.networkx.draw(
                G,
                pos=node_positions,
                node_size=10,  # Size of the centroid nodes
                node_color="skyblue",  # Color of the centroid nodes
                edge_width=1.0,  # Width of the connecting lines
                edge_color="lightskyblue",  # Color of the connecting lines (choose contrast)
                alpha=hex_alpha,
            ).opts(
                # Set plot options directly if needed, often inherited
                # frame_width=plot_width, frame_height=plot_height
            )
        else:
            print("Warning: Graph is empty, skipping network overlay.")
            network_plot = None  # Or an empty plot object if needed

        # --- Combine Plots ---
        if network_plot:
            final_plot = hex_plot * network_plot
        else:
            final_plot = hex_plot  # Only show hex plot if graph was empty
    else:
        plot_figsize = (15, 15)
        hex_alpha = 0.65  # Transparency
        hex_edgecolor = "black"
        hex_facecolor = "red"
        hex_linewidth = 0.5
        # --- ADJUSTED FONT SIZE ---
        # Smaller initial size, will be clear on zoom because it's vector
        label_fontsize = 3  # <<<< CHANGED FROM 5 (Try 2, 3, or 4)
        label_color = "white"
        label_weight = "bold"
        basemap_source = cx.providers.OpenStreetMap.CH

        if isinstance(hour, int) or isinstance(hour, float):
            hour = str(int(hour))
        hr = hour

        fig, ax = plt.subplots(1, 1, figsize=plot_figsize)
        ax.set_aspect("equal")

        # Plot the hexagons
        geo_df.plot(
            ax=ax,
            facecolor=hex_facecolor,
            edgecolor=hex_edgecolor,
            alpha=hex_alpha,
            linewidth=hex_linewidth,
            column=hr,
            cmap="Reds",
            legend=True,
            legend_kwds={"shrink": 0.4},
        )

        # Add labels to hexagons
        # Suppress UserWarning about non-unit complex numbers if it appears during centroid calculation
        for idx, row in geo_df.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(row["hex_id"]),
                ha="center",
                va="center",
                fontsize=label_fontsize,  # Using the smaller font size
                color=label_color,
                weight=label_weight,
                clip_on=True,  # Prevent labels from drawing outside plot area initially
            )

        # Add the basemap
        minx_plot, miny_plot, maxx_plot, maxy_plot = geo_df.total_bounds
        buffer_plot = 500
        ax.set_xlim(minx_plot - buffer_plot, maxx_plot + buffer_plot)
        ax.set_ylim(miny_plot - buffer_plot, maxy_plot + buffer_plot)

        # Use zoom='auto' first, may need adjustment depending on area size/detail
        cx.add_basemap(
            ax,
            crs=geo_df.crs.to_string(),
            source=basemap_source,
            zoom="auto",
            # If 'auto' zoom is not ideal, try setting a specific integer level, e.g., zoom=14
        )

        # Customize plot
        ax.set_title(f"Demands at {hr}:00 hr", fontsize=16)
        ax.set_axis_off()

        final_plot = ax

    return final_plot


def compute_shortest_distance(G, origin, destination):
    """
    G: networkx object as returned from plot_network
    origin: hex_id of the origin hexagon
    destination: hex_id of the detination hexagon
    """
    return 200 * nx.shortest_path_length(G, source=origin, target=destination)


def reachable_zones(G, origin, cutoff_distance):
    """
    G: networkx object as returned from plot_network
    origin: hex_id of the origin hexagon
    cutoff_distance: maximum distance [meters] from the origin hexagon
    """
    return list(nx.single_source_shortest_path_length(G, origin, cutoff=int(cutoff_distance / 200)))
