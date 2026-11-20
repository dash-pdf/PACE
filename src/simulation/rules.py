"""PACE: FIA Sporting Rules & Classification Engine Module.

Enforces official FIA Formula 1 sporting regulations:
1. Article 54.1: A car is classified if and only if it completes >= 90% of winner's laps:
   L_i >= floor(0.90 * L_winner).
2. Lapped cars completing >= 90% are classified finishers behind lead-lap finishers.
3. Unclassified retirements (DNF) are ranked behind all classified finishers by laps completed.
4. Strictly guarantees a valid, tie-free 1..N permutation finish order.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
import pandas as pd
from src.simulation.state import CarState, CarStatus, RaceState, TrackCondition


class FIAClassificationEngine:
    """Classifies final race outcomes under FIA Formula 1 Sporting Regulations."""

    @staticmethod
    def classify_race(race: RaceState) -> pd.DataFrame:
        """Produce the official, mathematically valid finishing classification.
        
        Args:
            race: Completed RaceState.
            
        Returns:
            df: DataFrame sorted 1 to N with columns:
                ['finish_position', 'driver_id', 'constructor_id', 'grid_position',
                 'laps_completed', 'elapsed_time', 'time_penalties', 'is_classified',
                 'status', 'gap_to_leader']
        """
        all_cars = list(race.cars.values())
        if not all_cars:
            return pd.DataFrame()

        # 1. Identify race winner
        # Winner is the active/finished car with highest laps completed and lowest elapsed time
        candidates = [c for c in all_cars if c.status in (CarStatus.ACTIVE, CarStatus.FINISHED)]
        if not candidates:
            # Fallback if everyone retired
            candidates = all_cars

        winner = max(
            candidates,
            key=lambda c: (c.laps_completed, -(c.elapsed_time + c.penalties_accrued)),
        )
        winner_laps = winner.laps_completed
        winner_total_time = winner.elapsed_time + winner.penalties_accrued

        # 90% threshold for classification (FIA Sporting Code Article 54.1)
        min_classified_laps = math.floor(0.90 * winner_laps) if winner_laps > 0 else 0

        # Enforce FIA Article 30.5: Mandatory use of at least 2 distinct dry compounds in dry race
        # Exception: Waived if wet track condition (Damp/Wet) or if any intermediate/wet compound used
        is_wet_race = race.track_condition in (TrackCondition.DAMP, TrackCondition.WET)

        for car in all_cars:
            if car.status in (CarStatus.ACTIVE, CarStatus.FINISHED):
                has_wet_compound = any(c in ("intermediate", "wet") for c in car.compounds_used)
                if not is_wet_race and not has_wet_compound:
                    dry_compounds = {c for c in car.compounds_used if c in ("soft", "medium", "hard")}
                    # Only enforce if car actually completed at least 20 laps (avoid penalising early retirements)
                    if len(dry_compounds) < 2 and car.laps_completed >= min_classified_laps:
                        car.status = CarStatus.DSQ

        # Partition cars into buckets
        classified_cars: List[CarState] = []
        dnf_cars: List[CarState] = []
        dns_cars: List[CarState] = []
        dsq_cars: List[CarState] = []

        for car in all_cars:
            if car.status == CarStatus.DSQ:
                dsq_cars.append(car)
            elif car.status == CarStatus.DNS:
                dns_cars.append(car)
            elif car.laps_completed >= min_classified_laps and not car.status.is_dnf:
                car.status = CarStatus.FINISHED
                classified_cars.append(car)
            else:
                dnf_cars.append(car)

        # 2. Sort Classified Finishers:
        # First by laps completed (descending: on-the-lead-lap cars beat -1 lap cars)
        # Then by total race time (elapsed + penalties) ascending
        classified_cars.sort(
            key=lambda c: (-c.laps_completed, c.elapsed_time + c.penalties_accrued)
        )

        # 3. Sort DNF cars:
        # Behind all classified finishers, ordered by laps completed descending, then elapsed time ascending
        dnf_cars.sort(
            key=lambda c: (-c.laps_completed, c.elapsed_time + c.penalties_accrued)
        )

        # Combine all groups in strict FIA priority order
        final_order = classified_cars + dnf_cars + dns_cars + dsq_cars

        rows = []
        for rank_pos, car in enumerate(final_order, 1):
            is_class = car in classified_cars
            total_time = car.elapsed_time + car.penalties_accrued

            # Gap to leader string formatting
            if car == winner:
                gap_str = "WINNER"
            elif is_class:
                lap_diff = winner_laps - car.laps_completed
                if lap_diff == 0:
                    delta_sec = total_time - winner_total_time
                    gap_str = f"+{delta_sec:.3f}s"
                else:
                    gap_str = f"+{lap_diff} Lap{'s' if lap_diff > 1 else ''}"
            elif car.status.is_dnf:
                gap_str = f"DNF (Lap {car.laps_completed})"
            elif car.status == CarStatus.DNS:
                gap_str = "DNS"
            elif car.status == CarStatus.DSQ:
                gap_str = "DSQ"
            else:
                gap_str = f"NC (Lap {car.laps_completed})"

            rows.append({
                "finish_position": rank_pos,
                "driver_id": car.driver_id,
                "constructor_id": car.constructor_id,
                "grid_position": car.grid_position,
                "laps_completed": car.laps_completed,
                "pit_stops": car.pit_stops_count,
                "compounds": "/".join(sorted(car.compounds_used)),
                "elapsed_time": round(total_time, 3),
                "time_penalties": car.penalties_accrued,
                "is_classified": is_class,
                "status": car.status.value,
                "gap_to_leader": gap_str,
            })

        df = pd.DataFrame(rows)
        return df
