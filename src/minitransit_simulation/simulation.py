"""
Event-driven simulation system using priority queue.
"""

import heapq
from datetime import datetime
from typing import List, Optional

from .actions import Action
from .demand import Demand
from .primitives import Route


class Event:
    """
    Represents an event in the simulation.

    An event contains a demand and a list of remaining actions to be executed.
    Events are sorted by their end_time (when the current action ends).

    Attributes:
        demand (Demand): The demand being processed.
        actions (List[Action]): List of remaining actions to execute.
        end_time (datetime): When the current (first) action ends.
        completed_actions (List[Action]): List of actions that have been completed.
    """

    def __init__(
        self, demand: Demand, actions: List[Action], completed_actions: List[Action] = None
    ):
        """
        Initialize an Event.

        Args:
            demand: The demand being processed.
            actions: List of remaining actions to execute.
            completed_actions: List of actions that have already been completed.
        """
        self.demand = demand
        self.actions = actions.copy() if actions else []
        self.completed_actions = completed_actions.copy() if completed_actions else []

        # End time is when the current action ends (if there is one)
        if self.actions:
            self.end_time = self.actions[0].end_time
        else:
            # No more actions, event is complete
            self.end_time = datetime.max

    def is_complete(self) -> bool:
        """Check if this event is complete (no more actions)."""
        return len(self.actions) == 0

    def get_current_action(self) -> Optional[Action]:
        """Get the current action (first in the list)."""
        return self.actions[0] if self.actions else None

    def get_next_event(self) -> Optional["Event"]:
        """
        Get the next event after completing the current action.

        Returns:
            Event with the next action, or None if no more actions.
        """
        if not self.actions:
            return None

        # Move current action to completed
        current_action = self.actions.pop(0)
        new_completed = self.completed_actions + [current_action]

        # Create new event with remaining actions
        return Event(self.demand, self.actions, new_completed)

    def get_route(self) -> Route:
        """
        Get the complete route from completed actions.

        Returns:
            Route object containing all completed actions.
        """
        # When event is complete, all actions should be in completed_actions
        # If not complete, include remaining actions as well
        if self.is_complete():
            all_actions = self.completed_actions
        else:
            all_actions = self.completed_actions + self.actions
        return Route(unit=self.demand.unit, actions=all_actions)

    def __lt__(self, other):
        """
        Compare events by end_time for priority queue ordering.
        If end_time is the same, prioritize events with Ride actions.
        """
        if not isinstance(other, Event):
            return NotImplemented

        # If end times are different, use end_time
        if self.end_time != other.end_time:
            return self.end_time < other.end_time

        # If end times are the same, prioritize Ride actions
        from src.actions.ride import Ride

        self_has_ride = self.actions and isinstance(self.actions[0], Ride)
        other_has_ride = other.actions and isinstance(other.actions[0], Ride)

        # Ride actions come first when times are equal
        if self_has_ride and not other_has_ride:
            return True
        if not self_has_ride and other_has_ride:
            return False

        # If both have same type, use demand time as tiebreaker
        return self.demand.time < other.demand.time

    def __repr__(self):
        return f"Event(demand={self.demand}, actions_remaining={len(self.actions)}, end_time={self.end_time})"


class Simulation:
    """
    Event-driven simulation using a priority queue.

    Events are processed in chronological order based on when each action ends.
    """

    def __init__(self, network):
        """
        Initialize the simulation.

        Args:
            network: Network object containing the transportation network.
        """
        self.network = network
        self.event_queue = []  # Priority queue (min-heap)
        self.completed_routes = []  # Routes that have been completed

    def add_demand(self, demand: Demand):
        """
        Add a demand to the simulation by finding its optimal route and creating an event.

        Args:
            demand: The demand to add to the simulation.
        """
        # Find optimal route for this demand
        print(f"Finding optimal route for demand: {demand}")
        route = self.network.get_optimal_route(demand)

        if route is None or not route.actions:
            # No valid route found, skip this demand
            return

        # Create event with all actions from the route
        event = Event(demand, route.actions)

        # Push to priority queue
        heapq.heappush(self.event_queue, event)

    def run(self) -> List[Route]:
        """
        Run the simulation by processing all events in the queue.
        Handles capacity constraints for Ride actions.

        Returns:
            List of completed routes.
        """
        self.completed_routes = []

        while self.event_queue:
            # print(f"Simulation time: {self.network.time}")
            # Pop the event with the earliest end_time (Ride actions prioritized when times are equal)
            current_event = heapq.heappop(self.event_queue)

            if current_event.is_complete():
                # All actions completed, save the route
                route = current_event.get_route()
                self.completed_routes.append(route)
                self.network.push_route(route)
            else:
                current_action = current_event.get_current_action()

                # Check if this is a Ride action that needs capacity checking
                from src.actions.ride import Ride

                if isinstance(current_action, Ride) and current_action.vehicle is not None:
                    # The event end_time is when the ride ends, so we need to unload passengers
                    vehicle = current_action.vehicle

                    # Unload passengers when ride ends
                    vehicle.unload_passengers(current_action.unit)

                    # Complete the ride action and move to next
                    next_event = current_event.get_next_event()
                    if next_event is not None:
                        heapq.heappush(self.event_queue, next_event)
                else:
                    # Check if next action is a Ride that needs boarding
                    next_action = (
                        current_event.actions[1] if len(current_event.actions) > 1 else None
                    )

                    if isinstance(next_action, Ride) and next_action.vehicle is not None:
                        # This event is about to start a Ride - need to check capacity at boarding time
                        # Group all events that want to board the same vehicle at the same time
                        events_to_process = [current_event]
                        boarding_time = next_action.start_time
                        vehicle = next_action.vehicle

                        # Collect all events with the same vehicle and boarding time
                        # Only peek at events that have the same end_time (current time)
                        temp_events = []
                        while (
                            self.event_queue
                            and self.event_queue[0].end_time == current_event.end_time
                        ):
                            peek_event = heapq.heappop(self.event_queue)
                            peek_next_action = (
                                peek_event.actions[1] if len(peek_event.actions) > 1 else None
                            )

                            if (
                                isinstance(peek_next_action, Ride)
                                and peek_next_action.vehicle == vehicle
                                and peek_next_action.start_time == boarding_time
                            ):
                                events_to_process.append(peek_event)
                            else:
                                temp_events.append(peek_event)

                        # Put back events that don't match
                        for event in temp_events:
                            heapq.heappush(self.event_queue, event)

                        # Process events in the group, checking capacity
                        for event in events_to_process:
                            ride_action = event.actions[1]  # Next action is the Ride

                            # Check if vehicle has capacity
                            if vehicle.current_load + ride_action.unit <= vehicle.capacity:
                                # Board the vehicle
                                vehicle.load_passengers(ride_action.unit)

                                # Complete current action and move to Ride
                                next_event = event.get_next_event()
                                if next_event is not None:
                                    heapq.heappush(self.event_queue, next_event)
                            else:
                                # Capacity exceeded - find next vehicle
                                updated_event = self._find_next_vehicle_for_event(event)
                                if updated_event is not None:
                                    heapq.heappush(self.event_queue, updated_event)
                                # If no next vehicle found, skip this demand (could log a warning)
                    else:
                        # Not a Ride action or no vehicle - process normally
                        next_event = current_event.get_next_event()

                        if next_event is not None:
                            heapq.heappush(self.event_queue, next_event)

        return self.completed_routes

    def _find_next_vehicle_for_event(self, event) -> Optional["Event"]:
        """
        Find the next available vehicle for a Ride action when capacity is exceeded.
        Updates the event with new Wait and Ride actions, and updates all subsequent actions.

        Args:
            event: Event where the next action is a Ride that couldn't board due to capacity.

        Returns:
            Updated event with next vehicle, or None if no vehicle found.
        """
        from datetime import timedelta

        from src.actions.ride import Ride
        from src.actions.wait import Wait
        from src.actions.walk import Walk

        # The next action (index 1) is the Ride we need to update
        if len(event.actions) < 2:
            return event

        ride_action = event.actions[1]
        if not isinstance(ride_action, Ride) or ride_action.vehicle is None:
            return event

        service = ride_action.service
        start_hex = ride_action.start_hex
        end_hex = ride_action.end_hex
        unit = ride_action.unit
        current_time = event.actions[0].end_time if event.actions else ride_action.start_time

        # Find next vehicle after the current one
        start_index = service.stop_hex_lookup[start_hex]
        end_index = service.stop_hex_lookup[end_hex]

        # Try to find next departure after current vehicle's departure
        try:
            # Get departure time of current vehicle
            current_vehicle = ride_action.vehicle
            _, current_departure = current_vehicle.timetable[start_index]

            # Find next vehicle after current departure
            next_vehicle = service.get_next_departure(
                current_departure + timedelta(seconds=1),  # Just after current departure
                start_index,
                end_index,
            )

            # Get new times from next vehicle
            _, next_departure = next_vehicle.timetable[start_index]
            next_arrival, _ = next_vehicle.timetable[end_index]

            # Create new Wait and Ride actions
            new_wait = Wait(current_time, next_departure, start_hex, unit)
            new_ride = Ride(
                next_departure,
                next_arrival,
                start_hex,
                end_hex,
                unit,
                service=service,
                vehicle=next_vehicle,
            )

            # Update all subsequent actions that depend on the ride end time
            updated_actions = [
                event.actions[0],
                new_wait,
                new_ride,
            ]  # Keep first action, add new wait and ride

            # Calculate time offset: how much later the new ride arrives compared to the old one
            time_offset = next_arrival - ride_action.end_time

            # Update subsequent actions (skip the old ride action at index 1)
            # Start from index 2 (after the old ride action)
            for action in event.actions[2:]:
                if isinstance(action, Walk):
                    # Update Walk action times based on new ride end time
                    new_start = action.start_time + time_offset
                    new_end = action.end_time + time_offset
                    updated_walk = Walk(
                        new_start,
                        new_end,
                        action.start_hex,
                        action.end_hex,
                        action.unit,
                        action.graph,
                        action.walk_speed,
                    )
                    updated_actions.append(updated_walk)
                elif isinstance(action, Ride):
                    # If there's another ride, update its start and end times
                    new_ride_start = action.start_time + time_offset
                    new_ride_end = action.end_time + time_offset
                    updated_ride = Ride(
                        new_ride_start,
                        new_ride_end,
                        action.start_hex,
                        action.end_hex,
                        action.unit,
                        action.service,
                        action.vehicle,
                    )
                    updated_actions.append(updated_ride)
                elif isinstance(action, Wait):
                    # Update Wait action times
                    new_wait_start = action.start_time + time_offset
                    new_wait_end = action.end_time + time_offset
                    updated_wait = Wait(new_wait_start, new_wait_end, action.location, action.unit)
                    updated_actions.append(updated_wait)
                else:
                    # Keep other actions as is (shouldn't happen, but just in case)
                    updated_actions.append(action)

            return Event(event.demand, updated_actions, event.completed_actions)

        except (ValueError, RuntimeError):
            # No next vehicle found
            return None

    def clear(self):
        """Clear the event queue and completed routes."""
        self.event_queue = []
        self.completed_routes = []
