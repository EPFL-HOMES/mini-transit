import math
import random
from datetime import datetime, timedelta
from typing import List, Optional

from src.demand import Demand
from src.models import DemandInput, DemandModel


class DemandSampler:
    def __init__(
        self,
        unit_sizes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        demand_data: optional backing data (not strictly needed for sampling here)
        unit_sizes: list of possible integer group sizes for a single demand
        seed: optional RNG seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.unit_sizes: List[int] = unit_sizes if unit_sizes is not None else [5]

    # ------------------------ Poisson process utils ------------------------ #
    def _poisson(self, lam: float) -> int:
        """
        Sample from a Poisson(lam) using Knuth's algorithm.
        lam is the expected number of events in the interval (e.g. per hour).
        """
        if lam <= 0:
            return 0

        L = math.exp(-lam)
        k = 0
        p = 1.0

        # Knuth algorithm
        while p > L:
            k += 1
            p *= self.rng.random()

        return k - 1

    def _sample_unit_size(self) -> int:
        """
        Sample the integer 'unit' for a single demand (e.g. group size).
        """
        if not self.unit_sizes:
            return 1  # sensible default

        if len(self.unit_sizes) == 1:
            return int(self.unit_sizes[0])

        return int(self.rng.choice(self.unit_sizes))

    def _sample_time_within_hour(self, base_time):
        """
        Sample a time uniformly within the hour starting at base_time.
        Returns a datetime with integer minutes only (seconds set to 0).

        If base_time is a datetime, return a datetime in [base_time, base_time + 1h).
        If it's numeric (e.g., hour index), convert to datetime and return datetime in [hour, hour+1).
        """
        # Sample a random minute (0-59) within the hour
        minute = self.rng.randint(0, 59)

        if isinstance(base_time, datetime):
            return base_time.replace(minute=minute, second=0, microsecond=0)
        else:
            # assume base_time is something like 0, 1, 2, ... (hour index)
            # Convert to datetime using a base date
            hour = int(base_time)
            return datetime(2024, 1, 1, hour, minute, 0)

    # ------------------------ Public API ------------------------ #
    def sample_hourly_demand(self, demand_inputs: List[DemandInput]) -> List[DemandModel]:
        """
        For each DemandInput, interpret demand_input.unit as λ, the expected number
        of trip requests in that hour, for the OD pair (start_hex, end_hex).

        For each such row, we:
          1. Sample N ~ Poisson(λ).
          2. For each of the N demands, sample a time uniformly within that hour.
          3. Draw an INTEGER unit size from self.unit_sizes for that demand.

        Returns a flat list of DemandModel (actually Demand) objects.
        """
        demands: List[DemandModel] = []

        for demand_input in demand_inputs:
            lam = float(demand_input.unit)  # from your CSV "demands" column
            n_trips = self._poisson(lam)

            for _ in range(n_trips):
                arrival_time = self._sample_time_within_hour(demand_input.hour)
                unit_size: int = self._sample_unit_size()

                demand_model = Demand(
                    time=arrival_time,
                    start_hex=demand_input.start_hex,
                    end_hex=demand_input.end_hex,
                    unit=unit_size,  # now guaranteed int
                )
                demands.append(demand_model)

        return demands
