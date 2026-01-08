"""
Demand class representing a specific travel demand or request.
"""

from datetime import datetime

import pandas as pd

from ..models import DemandInput
from ..primitives.hex import Hex


class Demand:
    """
    Represents a specific travel demand or request.

    Attributes:
        time (datetime): The time at which the demand occurs.
        start_hex (Hex): The starting hexagonal cell for the demand.
        end_hex (Hex): The destination hexagonal cell for the demand.
        unit (int): A numerical value associated with the demand (e.g., number of passengers, weight).
    """

    def __init__(self, time: datetime, start_hex: Hex, end_hex: Hex, unit: int):
        """
        Initialize a Demand object.

        Args:
            time (datetime): The time at which the demand occurs.
            start_hex (Hex): The starting hexagonal cell for the demand.
            end_hex (Hex): The destination hexagonal cell for the demand.
            unit (int): A numerical value associated with the demand.
        """
        self.time = time
        self.start_hex = start_hex
        self.end_hex = end_hex
        self.unit = unit

    def __repr__(self):
        return f"Demand(time={self.time}, start_hex={self.start_hex}, end_hex={self.end_hex}, unit={self.unit})"

    def __eq__(self, other):
        if not isinstance(other, Demand):
            return False
        return (
            self.time == other.time
            and self.start_hex == other.start_hex
            and self.end_hex == other.end_hex
        )


def demand_input_from_csv(csv_path: str):
    """
    Load input demands from CSV file as DemandInput objects.

    Args:
        csv_path (str): Path to the CSV file containing time-dependent demands.

    Returns:
        list: List of DemandInput objects.
    """
    demand_inputs = []

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        # Parse each row to create DemandInput objects
        for _, row in df.iterrows():
            # CSV has columns: departure_hour, start_hex_id, end_hex_id, demands
            hour = int(row.get("departure_hour", 0))
            start_hex_id = int(row.get("start_hex_id", 0))
            end_hex_id = int(row.get("end_hex_id", 0))
            unit = int(row.get("demands", 0))

            # Create Hex objects
            start_hex = Hex(start_hex_id)
            end_hex = Hex(end_hex_id)

            # Create DemandInput object
            # The unit field represents λ (lambda) for Poisson sampling
            demand_input = DemandInput(
                hour=hour,
                start_hex=start_hex,
                end_hex=end_hex,
                unit=unit,
            )
            demand_inputs.append(demand_input)

    except Exception as e:
        print(f"Error loading demands from {csv_path}: {e}")
        # Return empty list if there's an error
        return []

    return demand_inputs
