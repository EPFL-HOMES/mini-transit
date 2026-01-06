from abc import ABC


class Service(ABC):
    def __init__(self, name):
        self.name = name

    def get_fare(self, start_hex, end_hex) -> float:
        # Dummy implementation for fare calculation
        # In a real scenario, this would involve complex logic
        raise NotImplementedError

    def get_travel_time(self, start_hex, end_hex) -> float:
        # Dummy implementation for travel time calculation
        # In a real scenario, this would involve complex logic
        raise NotImplementedError

    def get_route(self, unit, start_time, start_hex, end_hex):
        # Dummy implementation for route retrieval
        # In a real scenario, this would involve complex logic
        raise NotImplementedError
