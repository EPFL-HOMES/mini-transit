"""
Action class representing a single action in a route.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class Action(ABC):
    """
    Abstract base class for actions in a route.

    Attributes:
        start_time (datetime): When the action starts.
        end_time (datetime): When the action ends.
    """

    def __init__(self, start_time: datetime, end_time: datetime = None, unit: int = None):
        """
        Initialize an Action object.

        Args:
            start_time (datetime): When the action starts.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.unit = unit

    @property
    def duration(self) -> timedelta:
        """
        Get the duration of this action.

        Returns:
            timedelta: Duration of the action.
        """
        if self.end_time is None:
            raise ValueError("Action not completed - end_time not set")
        return self.end_time - self.start_time

    @property
    def duration_minutes(self) -> float:
        """
        Get the duration in minutes.

        Returns:
            float: Duration in minutes.
        """
        return self.duration.total_seconds() / 60.0

    def __repr__(self):
        return f"{self.__class__.__name__}(start_time={self.start_time}, end_time={self.end_time})"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.start_time == other.start_time and self.end_time == other.end_time
