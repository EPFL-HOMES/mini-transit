"""
Walk class representing walking action in a route.
"""

from datetime import datetime, timedelta
import networkx as nx
from src.actions.action import Action
from src.hex import Hex
import json
import os

class Walk(Action):
    """
    Represents a walking action between two hexagons.
    
    Attributes:
        start_hex (Hex): Starting hexagon.
        end_hex (Hex): Destination hexagon.
        walk_speed (float): Walking speed in hexagons per hour.
    """
    
    def __init__(self, start_time: datetime, end_time: datetime, start_hex: Hex, end_hex: Hex, unit: float, graph: nx.Graph, walk_speed: float = None):
        """
        Initialize a Walk action.
        
        Args:
            start_time (datetime): When the walk starts.
            start_hex (Hex): Starting hexagon.
            end_hex (Hex): Destination hexagon.
            walk_speed (float, optional): Walking speed in hexagons per hour. 
                                       If None, loads from config.json.
        """
        super().__init__(start_time, end_time, units=unit)
        self.start_hex = start_hex
        self.end_hex = end_hex
        #self.graph = graph because we do not need _calculate_end_time for now
        self.fare = 0.0  # Walking has no fare

        
        if walk_speed is None:
            self.walk_speed = self._load_walk_speed_from_config()
        else:
            self.walk_speed = walk_speed
        
        # making end_time mandatory for Walk action for now because we literally already compute it in network.py

        # yeeted for now: Calculate end time based on distance and speed if not provided
        #if not end_time:
            #self._calculate_end_time()
    
    def _load_walk_speed_from_config(self) -> float:
        """
        Load walking speed from config.json.
        
        Returns:
            float: Walking speed in hexagons per hour.
        """
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('walk_speed', 10.0)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            # Default fallback
            return 10.0
    
    def _calculate_end_time(self):
        """
        Calculate the end time based on distance and walking speed.
        Args:
            graph: The network graph to calculate distance.
        """
        # For now, assume distance is 1 hexagon (neighbors)
        #TODO handle errors where the path does not exist
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
        # For now, return 1.0 (neighboring hexagons)
        # In a real implementation, you'd calculate actual distance
        return 1.0
    
    def __repr__(self):
        return f"Walk(start_time={self.start_time}, start_hex={self.start_hex}, end_hex={self.end_hex}, walk_speed={self.walk_speed})"
        
