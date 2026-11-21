"""PACE: Dynamic Weather, Track Moisture & Tyre Crossover Module (Stage 5).

Models:
1. Dynamic track moisture progression M(l) in [0.0, 1.0].
2. Track condition transitions: DRY (M < 0.20) <-> DAMP (0.20 <= M < 0.65) <-> WET (M >= 0.65).
3. Weather radar forecast generation with epistemic forecast horizon uncertainty.
4. Optimal tyre crossover decision logic (Slick <-> Intermediate <-> Wet).
5. Wet weather incident risk and aquaplaning multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.simulation.state import CarState, RaceState, TrackCondition
from src.simulation.tyres import CompoundType
from src.simulation.events import EventDispatcher, EventType, SimulationEvent


@dataclass
class WeatherEvolutionConfig:
    """Configuration for evolving meteorological conditions across a Grand Prix."""

    rain_start_lap: Optional[int] = None
    rain_peak_lap: Optional[int] = None
    rain_end_lap: Optional[int] = None
    peak_moisture: float = 0.80
    forecast_uncertainty_laps: int = 2


class WeatherEngine:
    """Manages track surface wetness, forecast generation, and strategic tyre crossover points."""

    def __init__(self, config: Optional[WeatherEvolutionConfig] = None):
        self.config = config or WeatherEvolutionConfig()

    def update_weather_state(
        self,
        race: RaceState,
        dispatcher: Optional[EventDispatcher] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> TrackCondition:
        """Advance weather state for the current lap and update race condition."""
        cfg = self.config
        lap = race.current_lap

        # If no rain configured, remain dry
        if cfg.rain_start_lap is None:
            race.track_moisture = 0.0
            race.track_condition = TrackCondition.DRY
            return TrackCondition.DRY

        # Compute deterministic or stochastic moisture curve
        if lap < cfg.rain_start_lap:
            moisture = 0.0
        elif lap <= (cfg.rain_peak_lap or (cfg.rain_start_lap + 8)):
            # Moisture rising to peak
            peak_lap = cfg.rain_peak_lap or (cfg.rain_start_lap + 8)
            frac = float(lap - cfg.rain_start_lap) / max(1.0, float(peak_lap - cfg.rain_start_lap))
            moisture = cfg.peak_moisture * frac
        elif cfg.rain_end_lap and lap <= cfg.rain_end_lap:
            # Sustained rain or slow initial drying
            moisture = cfg.peak_moisture * 0.90
        elif cfg.rain_end_lap and lap > cfg.rain_end_lap:
            # Track drying phase (racing line drys over ~10-14 laps)
            dry_laps = lap - cfg.rain_end_lap
            moisture = max(0.0, cfg.peak_moisture * (1.0 - dry_laps / 12.0))
        else:
            moisture = cfg.peak_moisture

        # Add minor random noise to moisture
        if rng is not None:
            moisture = float(np.clip(moisture + rng.normal(0.0, 0.02), 0.0, 1.0))

        old_condition = race.track_condition

        # Classify surface condition based on physical moisture thresholds
        if moisture < 0.20:
            new_condition = TrackCondition.DRY
        elif moisture < 0.65:
            new_condition = TrackCondition.DAMP
        else:
            new_condition = TrackCondition.WET

        race.track_moisture = moisture
        race.track_condition = new_condition

        if new_condition != old_condition and dispatcher is not None:
            dispatcher.emit(
                SimulationEvent(
                    event_type=EventType.WEATHER_CHANGE,
                    lap=lap,
                    details={
                        "old_condition": old_condition.value,
                        "new_condition": new_condition.value,
                        "moisture_level": round(moisture, 3),
                    },
                )
            )

        return new_condition

    def evaluate_crossover_tyre(
        self,
        current_compound: str,
        track_moisture: float,
        laps_remaining: int,
    ) -> Optional[CompoundType]:
        """Determine if current conditions dictate a switch to wet or dry tyres.

        Crossover Points:
        - M >= 0.65: Full Wet (or Intermediate if drying rapidly).
        - 0.20 <= M < 0.65: Intermediate (optimal damp window).
        - M < 0.20: Slicks (Medium / Hard / Soft).
        """
        c_type = CompoundType(current_compound.lower())

        if track_moisture >= 0.65:
            # Heavy standing water: require Full Wet (or Intermediate if already on inters and laps < 6)
            if c_type.is_dry:
                return CompoundType.WET if track_moisture >= 0.75 else CompoundType.INTERMEDIATE
            elif c_type == CompoundType.INTERMEDIATE and track_moisture >= 0.80:
                return CompoundType.WET

        elif 0.22 <= track_moisture < 0.65:
            # Damp track: Intermediate is optimal
            if c_type.is_dry:
                return CompoundType.INTERMEDIATE
            elif c_type == CompoundType.WET:
                return CompoundType.INTERMEDIATE

        elif track_moisture < 0.18:
            # Dry racing line: Slicks required
            if c_type.is_wet:
                return CompoundType.MEDIUM

        return None
