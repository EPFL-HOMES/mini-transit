from dataclasses import dataclass

from .actions.action import Action
from .actions.ride import Ride
from .actions.wait import Wait
from .actions.walk import Walk


@dataclass
class BaseSerializedAction:
    """Common fields present in all serialized actions."""

    type: str
    start_time: str  # Format: "HH:MM"
    end_time: str | None  # Format: "HH:MM"
    duration_minutes: float


@dataclass
class WalkSerializedAction(BaseSerializedAction):
    """Fields specific to a serialized 'Walk' action."""

    start_hex: int
    end_hex: int
    walk_speed: float
    distance: float


@dataclass
class RideSerializedAction(BaseSerializedAction):
    """Fields specific to a serialized 'Ride' action."""

    start_hex: int
    end_hex: int


@dataclass
class WaitSerializedAction(BaseSerializedAction):
    """Fields specific to a serialized 'Wait' action."""

    start_hex: int
    end_hex: int


# Type alias for convenience when typing a list of actions
SerializedAction = WalkSerializedAction | RideSerializedAction | WaitSerializedAction


def serialize_action(action: Action) -> dict:
    """
    Serialize an Action object to ensure all keys and values are JSON-serializable.

    Args:
        action (Action): The Action object to serialize.
    Returns:
        dict: A JSON-serializable version of the Action object.
    """

    # Action object
    action_data = {
        "type": action.__class__.__name__,
        "start_time": action.start_time.strftime("%H:%M"),
        "end_time": (action.end_time.strftime("%H:%M") if action.end_time else None),
        "duration_minutes": action.duration_minutes,
    }

    # Add specific fields for different action types
    if isinstance(action, Walk):
        action_data.update(
            {
                "start_hex": action.start_hex.hex_id,
                "end_hex": action.end_hex.hex_id,
                "walk_speed": action.walk_speed,
                "distance": action.distance,
            }
        )
    elif isinstance(action, Ride):
        action_data.update(
            {
                "start_hex": action.start_hex.hex_id,
                "end_hex": action.end_hex.hex_id,
            }
        )
    elif isinstance(action, Wait):
        # For Wait, location is the hex where waiting occurs
        action_data.update(
            {
                "start_hex": action.location.hex_id,
                "end_hex": action.location.hex_id,
            }
        )

    return action_data


# Formerly in APIServer.run_simulation, moved to separate file for clarity
def serialize_action_dict(action: dict) -> dict:
    """
    Serialize an action dictionary to ensure all keys and values are JSON-serializable.

    Args:
        action (dict): The action dictionary to serialize.
    Returns:
        dict: A JSON-serializable version of the action dictionary.
    """

    # Dictionary action
    start_time_obj = action["start_time"]
    end_time_obj = action.get("end_time")
    action_data = {
        "type": action.get("type", "Unknown"),
        "start_time": (
            start_time_obj.strftime("%H:%M")
            if hasattr(start_time_obj, "strftime")
            else str(start_time_obj)
        ),
        "end_time": (
            end_time_obj.strftime("%H:%M")
            if end_time_obj and hasattr(end_time_obj, "strftime")
            else (str(end_time_obj) if end_time_obj else None)
        ),
        "duration_minutes": (
            (end_time_obj - start_time_obj).total_seconds() / 60.0
            if end_time_obj and hasattr(end_time_obj, "__sub__")
            else 0.0
        ),
    }

    # Add specific fields for different action types
    action_type = action.get("type")
    if action_type == "Walk":
        action_data.update(
            {
                "start_hex": action["start_hex"].hex_id,
                "end_hex": action["end_hex"].hex_id,
                "walk_speed": action["walk_speed"],
                "distance": action["distance"],
            }
        )
    elif action_type == "Ride":
        if "start_hex" in action and "end_hex" in action:
            action_data.update(
                {
                    "start_hex": (
                        action["start_hex"].hex_id
                        if hasattr(action["start_hex"], "hex_id")
                        else action["start_hex"]
                    ),
                    "end_hex": (
                        action["end_hex"].hex_id
                        if hasattr(action["end_hex"], "hex_id")
                        else action["end_hex"]
                    ),
                }
            )
    elif action_type == "Wait":
        if "location" in action:
            location = action["location"]
            location_hex_id = location.hex_id if hasattr(location, "hex_id") else location
            action_data.update(
                {
                    "start_hex": location_hex_id,
                    "end_hex": location_hex_id,
                }
            )

    return action_data
