"""
Route class representing a complete route taken by a unit.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List

from src.actions.action import Action


class Route:
    """
    Represents a complete route taken by a unit.

    Attributes:
        unit (float): Number of units that took this route.
        actions (List[Action]): List of actions in this route.
        time_taken (timedelta): Total time taken for this route.
        total_fare (float): Total fare for this route.
    """

    def __init__(self, unit: float, actions: List[Action] = [], transfers: int = 0):
        """
        Initialize a Route object.

        Args:
            unit (float): Number of units that took this route.
            actions: List of actions in this route (can be Action objects or dictionaries).
        """
        self.unit = unit
        self.actions = actions
        self.num_transfers = transfers
        self.time_taken = self._calculate_total_time()
        self.total_fare = self._calculate_total_fare()
        # total cost means a custom metric combining fare, time, etc.
        self.total_cost = self._calculate_total_cost()

    def _calculate_total_cost(self) -> float:
        """
        Calculate total cost for this route.
        Utility function: total_cose = - total_fare - alpha * (total_in_vehicle_time + total_access_time + total_wait_time) - phi * num_transfers

        Returns:
            float: Total cost for this route.
        """

        def _read_utility_function_params_from_config():
            # Placeholder for reading from config
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../data/config.json"
            )
            with open(config_path, "r") as f:
                config = json.load(f)
            alpha = config.get("utility_function_alpha", 1.5)  # Default to 1.5 if not specified
            phi = config.get("utility_function_phi", 2.0)  # Default to 2.0 if not specified
            return alpha, phi

        alpha, phi = _read_utility_function_params_from_config()
        total_fare = self._calculate_total_fare()
        total_in_vehicle_time = (
            self._calculate_total_in_vehicle_time().total_seconds() / 60.0
        )  # in minutes
        total_access_time = self._calculate_total_access_time().total_seconds() / 60.0  # in minutes
        total_wait_time = self._calculate_total_wait_time().total_seconds() / 60.0  # in minutes
        total_cost = (
            -total_fare
            - alpha * (total_in_vehicle_time + total_access_time + total_wait_time)
            - phi * self.num_transfers
        )

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
            if hasattr(action, "start_time"):
                # Action object
                start_times.append(action.start_time)
                if hasattr(action, "end_time") and action.end_time is not None:
                    end_times.append(action.end_time)
            elif isinstance(action, dict) and "start_time" in action:
                # Dictionary action
                start_times.append(action["start_time"])
                if "end_time" in action and action["end_time"] is not None:
                    end_times.append(action["end_time"])

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
            if hasattr(action, "fare"):
                total_fare += action.fare
            elif isinstance(action, dict) and "fare" in action:
                total_fare += action["fare"]

        return total_fare

    def _calculate_total_access_time(self) -> timedelta:
        """
        Calculate total access time for this route. In this context "access time" is defined as the total time spent walking aka the sum of the spent time of every "Walk" type action.

        Returns:
            timedelta: Total access time.
        """
        total_access_time = timedelta(0)
        for action in self.actions:
            if hasattr(action, "start_time") and hasattr(action, "end_time"):
                # Check if action is of type Walk
                if action.__class__.__name__ == "Walk":
                    total_access_time += action.end_time - action.start_time
            elif isinstance(action, dict) and "start_time" in action and "end_time" in action:
                # Check if action is of type Walk
                if action.get("type") == "Walk":
                    total_access_time += action["end_time"] - action["start_time"]
        return total_access_time

    def _calculate_total_wait_time(self) -> timedelta:
        """
        Calculate total wait time for this route. In this context "wait time" is defined as the total time spent waiting aka the sum of the spent time of every "Wait" type action.

        Returns:
            timedelta: Total wait time.
        """
        total_wait_time = timedelta(0)
        for action in self.actions:
            if hasattr(action, "start_time") and hasattr(action, "end_time"):
                # Check if action is of type Wait
                if action.__class__.__name__ == "Wait":
                    total_wait_time += action.end_time - action.start_time
            elif isinstance(action, dict) and "start_time" in action and "end_time" in action:
                # Check if action is of type Wait
                if action.get("type") == "Wait":
                    total_wait_time += action["end_time"] - action["start_time"]
        return total_wait_time

    def _calculate_total_in_vehicle_time(self) -> timedelta:
        """
        Calculate total in-vehicle time for this route. In this context "in-vehicle time" is defined as the total time spent in vehicles aka the sum of the spent time of every "Ride" type action.

        Returns:
            timedelta: Total in-vehicle time.
        """
        total_in_vehicle_time = timedelta(0)
        for action in self.actions:
            if hasattr(action, "start_time") and hasattr(action, "end_time"):
                # Check if action is of type Ride
                if action.__class__.__name__ == "Ride":
                    total_in_vehicle_time += action.end_time - action.start_time
            elif isinstance(action, dict) and "start_time" in action and "end_time" in action:
                # Check if action is of type Ride
                if action.get("type") == "Ride":
                    total_in_vehicle_time += action["end_time"] - action["start_time"]
        return total_in_vehicle_time

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
        return (
            self.unit == other.unit
            and self.actions == other.actions
            and self.time_taken == other.time_taken
            and self.total_fare == other.total_fare
        )
