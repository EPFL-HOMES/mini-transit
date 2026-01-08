"""
Walk class representing walking action in a route.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import networkx as nx

from ..primitives.hex import Hex
from .action import Action


@dataclass
class WalkConfig:
    walk_speed: float = 10.0  # Default walking speed in hexagons per hour

class Walk(Action):
    """
    Represents a walking action between two hexagons.

    Attributes:
        start_hex (Hex): Starting hexagon.
        end_hex (Hex): Destination hexagon.
        walk_speed (float): Walking speed in hexagons per hour.
    """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        start_hex: Hex,
        end_hex: Hex,
        unit: int,
        graph: nx.Graph,
        walk_speed: float = None,
        config: WalkConfig = WalkConfig(),
    ):
        """
        Initialize a Walk action.

        Args:
            start_time (datetime): When the walk starts.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            walk_speed (float, optional): Walking speed in hexagons per hour.
                                       If None, gets from config.
        """
        super().__init__(start_time, end_time, unit=unit)
        self.config = config
        self.start_hex = start_hex
        self.end_hex = end_hex
        self.graph = graph  # Store graph for distance calculation
        self.fare = 0.0  # Walking has no fare

        if walk_speed is None:
            self.walk_speed = self.config.walk_speed
        else:
            self.walk_speed = walk_speed

        # making end_time mandatory for Walk action for now because we literally already compute it in network.py

        # yeeted for now: Calculate end time based on distance and speed if not provided
        # if not end_time:
        # self._calculate_end_time()

    def _calculate_end_time(self):
        """
        Calculate the end time based on distance and walking speed.
        Args:
            graph: The network graph to calculate distance.
        """
        # For now, assume distance is 1 hexagon (neighbors)
        # TODO handle errors where the path does not exist
        path = nx.shortest_path(self.graph, self.start_hex.hex_id, self.end_hex.hex_id)

        # Calculate total distance (number of edges in path)
        distance = len(path) - 1  # Number of edges = number of nodes - 1

        # Calculate time in hours
        time_hours = distance / self.walk_speed

        # Convert to timedelta and set end_time
        self.end_time = self.start_time + timedelta(hours=time_hours)

    @property
    def distance(self) -> float:
        """
        Get the distance of this walk.

        Returns:
            float: Distance in hexagons.
        """
        if self.graph is None:
            return 1.0  # Fallback if graph not available

        try:
            # Calculate shortest path length in hexagons
            distance = nx.shortest_path_length(
                self.graph, self.start_hex.hex_id, self.end_hex.hex_id
            )
            return float(distance)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path exists, return infinity or a large number
            return float("inf")

    def __repr__(self):
        return f"Walk(start_time={self.start_time}, start_hex={self.start_hex}, end_hex={self.end_hex}, walk_speed={self.walk_speed})"
