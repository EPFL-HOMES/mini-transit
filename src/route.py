"""
Route class representing a complete route taken by a unit.
"""

from datetime import datetime, timedelta
from typing import List
from .action import Action

class Route:
    """
    Represents a complete route taken by a unit.
    
    Attributes:
        unit (float): Number of units that took this route.
        actions (List[Action]): List of actions in this route.
        time_taken (timedelta): Total time taken for this route.
        total_fare (float): Total fare for this route.
    """
    
    def __init__(self, unit: float, actions: List[Action]=[]):
        """
        Initialize a Route object.
        
        Args:
            unit (float): Number of units that took this route.
            actions: List of actions in this route (can be Action objects or dictionaries).
        """
        self.unit = unit
        self.actions = actions
        self.time_taken = self._calculate_total_time()
        self.total_fare = self._calculate_total_fare()
        # total cost means a custom metric combining fare, time, etc.
        self.total_cost = self._calculate_total_cost()

    def _calculate_total_cost(self) -> float:
        """
        Calculate total cost for this route.
        
        For now, use a simple formula: total_cost = total_fare + (time_in_minutes * 0.3)
        Returns:
            float: Total cost for this route.
        """
        total_cost = self.total_fare + (self.time_taken.total_seconds() / 60.0) * 0.3  # Example cost metric
        return total_cost
    
    def _calculate_total_time(self) -> timedelta:
        """
        Calculate total time taken for this route.
        
        Returns:
            timedelta: Total time taken.
        """
        if not self.actions:
            return timedelta(0)
        
        # Handle both Action objects and dictionary actions
        start_times = []
        end_times = []
        
        for action in self.actions:
            if hasattr(action, 'start_time'):
                # Action object
                start_times.append(action.start_time)
                if hasattr(action, 'end_time') and action.end_time is not None:
                    end_times.append(action.end_time)
            elif isinstance(action, dict) and 'start_time' in action:
                # Dictionary action
                start_times.append(action['start_time'])
                if 'end_time' in action and action['end_time'] is not None:
                    end_times.append(action['end_time'])
        
        if not start_times or not end_times:
            return timedelta(0)
        
        earliest_start = min(start_times)
        latest_end = max(end_times)
        
        return latest_end - earliest_start
    
    def _calculate_total_fare(self) -> float:
        """
        Calculate total fare for this route.
        
        For now, use a simple formula: fare = total_time_in_minutes * 0.1
        
        Returns:
            float: Total fare for this route.
        """
        if not self.actions:
            return 0.0
        
        total_fare = 0.0
        for action in self.actions:
            if hasattr(action, 'fare'):
                total_fare += action.fare
            elif isinstance(action, dict) and 'fare' in action:
                total_fare += action['fare']

        return total_fare
    
    
    @property
    def time_taken_minutes(self) -> float:
        """
        Get total time taken in minutes.
        
        Returns:
            float: Total time in minutes.
        """
        return self.time_taken.total_seconds() / 60.0
    
    @property
    def start_time(self) -> datetime:
        """
        Get the start time of this route.
        
        Returns:
            datetime: Start time of the route.
        """
        if not self.actions:
            raise ValueError("Route has no actions")
        return min(action.start_time for action in self.actions)
    
    @property
    def end_time(self) -> datetime:
        """
        Get the end time of this route.
        
        Returns:
            datetime: End time of the route.
        """
        if not self.actions:
            raise ValueError("Route has no actions")
        
        end_times = [action.end_time for action in self.actions if action.end_time is not None]
        if not end_times:
            raise ValueError("Route has no completed actions")
        
        return max(end_times)
    
    def add_action(self, action: Action):
        """
        Add an action to this route.
        
        Args:
            action (Action): Action to add.
        """
        self.actions.append(action)
        # Recalculate totals
        self.time_taken = self._calculate_total_time()
        self.total_fare = self._calculate_total_fare()
    
    def extend_actions(self, actions: List[Action]):
        """
        Extend the actions list with multiple actions.
        
        Args:
            actions (List[Action]): List of actions to add.
        """
        self.actions.extend(actions)
        # Recalculate totals
        self.time_taken = self._calculate_total_time()
        self.total_fare = self._calculate_total_fare()
    
    def __repr__(self):
        return f"Route(unit={self.unit}, actions={len(self.actions)}, time_taken={self.time_taken}, total_fare={self.total_fare})"
    
    def __eq__(self, other):
        if not isinstance(other, Route):
            return False
        return (self.unit == other.unit and 
                self.actions == other.actions and 
                self.time_taken == other.time_taken and 
                self.total_fare == other.total_fare)
