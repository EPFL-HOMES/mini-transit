"""
Module for Service class representing a transit service.
"""
import random

class Service:
    def __init__(self):
       pass

    def get_fare(start_hex: int, end_hex: int) -> float:
        # placeholder implementation
        return random.uniform(1.0, 10.0)

    def get_travel_time(start_hex: int, end_hex: int) -> float:
        # placeholder implementation
        return random.uniform(5.0, 60.0)