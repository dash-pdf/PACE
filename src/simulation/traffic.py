"""PACE: Traffic, DRS & Stochastic Overtaking Module (Stage 4).

Models:
1. Micro-Sector progress & instantaneous on-track gap tracking across 3 sectors.
2. Circuit-specific DRS zones, detection points, and aerodynamic drag reduction.
3. Turbulent dirty-air wake penalties (cornering downforce loss + accelerated tyre wear).
4. Stochastic overtaking conditioned on net pace delta, tyre age, circuit difficulty, and driver defensive ability.
5. Blue-flag yielding logic for lapped cars under FIA Sporting Regulations (Appendix H).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.simulation.state import CarState, CarStatus, RaceMode, RaceState, TrackCondition
from src.simulation.events import EventDispatcher, EventType, SimulationEvent


@dataclass(frozen=True)
class CircuitDRSConfig:
    """Circuit-specific aerodynamic and overtaking characteristics."""

    circuit_id: str
    num_drs_zones: int
    drs_delta_per_zone_s: float  # Base lap time reduction per active DRS straight
    overtaking_difficulty: float  # 1.0 (Low, e.g. Monza) to 4.8 (Extreme, e.g. Monaco)
    overtake_threshold_s: float  # Net pace delta needed for ~50% pass probability
    sector_fractions: Tuple[float, float, float] = (0.28, 0.42, 0.30)
    drs_sectors: Tuple[int, ...] = (2, 3)  # Sectors with DRS activation straights
    primary_overtake_sector: int = 3  # Main overtaking zone sector


# Empirical circuit configurations calibrated from historical F1 Grand Prix data
CIRCUIT_DRS_PROFILES: Dict[str, CircuitDRSConfig] = {
    "albert_park": CircuitDRSConfig(
        circuit_id="albert_park",
        num_drs_zones=4,
        drs_delta_per_zone_s=0.12,
        overtaking_difficulty=2.5,
        overtake_threshold_s=0.80,
        sector_fractions=(0.29, 0.41, 0.30),
        drs_sectors=(1, 2, 3),
        primary_overtake_sector=1,
    ),
    "bahrain": CircuitDRSConfig(
        circuit_id="bahrain",
        num_drs_zones=3,
        drs_delta_per_zone_s=0.16,
        overtaking_difficulty=1.5,
        overtake_threshold_s=0.55,
        sector_fractions=(0.31, 0.40, 0.29),
        drs_sectors=(1, 2, 3),
        primary_overtake_sector=1,
    ),
    "baku": CircuitDRSConfig(
        circuit_id="baku",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.24,
        overtaking_difficulty=1.4,
        overtake_threshold_s=0.48,
        sector_fractions=(0.30, 0.38, 0.32),
        drs_sectors=(1, 3),
        primary_overtake_sector=3,
    ),
    "catalunya": CircuitDRSConfig(
        circuit_id="catalunya",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.18,
        overtaking_difficulty=3.0,
        overtake_threshold_s=1.10,
        sector_fractions=(0.28, 0.43, 0.29),
        drs_sectors=(1, 3),
        primary_overtake_sector=1,
    ),
    "hungaroring": CircuitDRSConfig(
        circuit_id="hungaroring",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.15,
        overtaking_difficulty=4.0,
        overtake_threshold_s=1.60,
        sector_fractions=(0.29, 0.42, 0.29),
        drs_sectors=(1, 3),
        primary_overtake_sector=1,
    ),
    "imola": CircuitDRSConfig(
        circuit_id="imola",
        num_drs_zones=1,
        drs_delta_per_zone_s=0.22,
        overtaking_difficulty=3.8,
        overtake_threshold_s=1.50,
        sector_fractions=(0.32, 0.38, 0.30),
        drs_sectors=(3,),
        primary_overtake_sector=3,
    ),
    "interlagos": CircuitDRSConfig(
        circuit_id="interlagos",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.18,
        overtaking_difficulty=1.8,
        overtake_threshold_s=0.62,
        sector_fractions=(0.28, 0.44, 0.28),
        drs_sectors=(1, 3),
        primary_overtake_sector=1,
    ),
    "jeddah": CircuitDRSConfig(
        circuit_id="jeddah",
        num_drs_zones=3,
        drs_delta_per_zone_s=0.15,
        overtaking_difficulty=2.2,
        overtake_threshold_s=0.75,
        sector_fractions=(0.32, 0.36, 0.32),
        drs_sectors=(1, 2, 3),
        primary_overtake_sector=1,
    ),
    "marina_bay": CircuitDRSConfig(
        circuit_id="marina_bay",
        num_drs_zones=4,
        drs_delta_per_zone_s=0.11,
        overtaking_difficulty=4.2,
        overtake_threshold_s=1.70,
        sector_fractions=(0.27, 0.43, 0.30),
        drs_sectors=(1, 2, 3),
        primary_overtake_sector=2,
    ),
    "miami": CircuitDRSConfig(
        circuit_id="miami",
        num_drs_zones=3,
        drs_delta_per_zone_s=0.15,
        overtaking_difficulty=2.4,
        overtake_threshold_s=0.80,
        sector_fractions=(0.30, 0.40, 0.30),
        drs_sectors=(1, 2, 3),
        primary_overtake_sector=3,
    ),
    "monaco": CircuitDRSConfig(
        circuit_id="monaco",
        num_drs_zones=1,
        drs_delta_per_zone_s=0.12,
        overtaking_difficulty=4.8,
        overtake_threshold_s=2.40,
        sector_fractions=(0.27, 0.46, 0.27),
        drs_sectors=(3,),
        primary_overtake_sector=1,
    ),
    "monza": CircuitDRSConfig(
        circuit_id="monza",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.22,
        overtaking_difficulty=1.2,
        overtake_threshold_s=0.40,
        sector_fractions=(0.33, 0.34, 0.33),
        drs_sectors=(1, 3),
        primary_overtake_sector=1,
    ),
    "red_bull_ring": CircuitDRSConfig(
        circuit_id="red_bull_ring",
        num_drs_zones=3,
        drs_delta_per_zone_s=0.16,
        overtaking_difficulty=1.4,
        overtake_threshold_s=0.50,
        sector_fractions=(0.31, 0.39, 0.30),
        drs_sectors=(1, 2, 3),
        primary_overtake_sector=1,
    ),
    "silverstone": CircuitDRSConfig(
        circuit_id="silverstone",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.19,
        overtaking_difficulty=2.2,
        overtake_threshold_s=0.75,
        sector_fractions=(0.28, 0.44, 0.28),
        drs_sectors=(2, 3),
        primary_overtake_sector=2,
    ),
    "spa": CircuitDRSConfig(
        circuit_id="spa",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.25,
        overtaking_difficulty=1.3,
        overtake_threshold_s=0.45,
        sector_fractions=(0.30, 0.42, 0.28),
        drs_sectors=(1, 3),
        primary_overtake_sector=1,
    ),
    "suzuka": CircuitDRSConfig(
        circuit_id="suzuka",
        num_drs_zones=1,
        drs_delta_per_zone_s=0.20,
        overtaking_difficulty=3.8,
        overtake_threshold_s=1.50,
        sector_fractions=(0.32, 0.41, 0.27),
        drs_sectors=(3,),
        primary_overtake_sector=1,
    ),
    "vegas": CircuitDRSConfig(
        circuit_id="vegas",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.25,
        overtaking_difficulty=1.5,
        overtake_threshold_s=0.50,
        sector_fractions=(0.28, 0.42, 0.30),
        drs_sectors=(2, 3),
        primary_overtake_sector=2,
    ),
    "villeneuve": CircuitDRSConfig(
        circuit_id="villeneuve",
        num_drs_zones=3,
        drs_delta_per_zone_s=0.17,
        overtaking_difficulty=1.6,
        overtake_threshold_s=0.55,
        sector_fractions=(0.28, 0.42, 0.30),
        drs_sectors=(2, 3),
        primary_overtake_sector=3,
    ),
    "zandvoort": CircuitDRSConfig(
        circuit_id="zandvoort",
        num_drs_zones=2,
        drs_delta_per_zone_s=0.15,
        overtaking_difficulty=4.0,
        overtake_threshold_s=1.60,
        sector_fractions=(0.30, 0.40, 0.30),
        drs_sectors=(1, 3),
        primary_overtake_sector=1,
    ),
}

DEFAULT_DRS_CONFIG = CircuitDRSConfig(
    circuit_id="default",
    num_drs_zones=2,
    drs_delta_per_zone_s=0.18,
    overtaking_difficulty=2.5,
    overtake_threshold_s=0.85,
    sector_fractions=(0.28, 0.42, 0.30),
    drs_sectors=(2, 3),
    primary_overtake_sector=3,
)


@dataclass
class OvertakeAttemptResult:
    """Outcome of an on-track overtaking attempt."""

    attempted: bool
    success: bool
    attacker_id: str
    defender_id: str
    sector: int
    prob_success: float
    net_pace_delta: float
    tyre_age_delta: int
    drs_active: bool
    reason: str = ""


class TrafficOvertakingEngine:
    """Orchestrates sector-level gap tracking, DRS, wake aerodynamics, overtaking, and blue flags."""

    def __init__(self, profiles: Optional[Dict[str, CircuitDRSConfig]] = None):
        self.profiles = profiles or CIRCUIT_DRS_PROFILES

    def get_circuit_config(self, circuit_id: str) -> CircuitDRSConfig:
        return self.profiles.get(circuit_id, DEFAULT_DRS_CONFIG)

    def is_drs_eligible(
        self,
        gap_at_detection: float,
        race_mode: RaceMode,
        lap: int,
        track_condition: TrackCondition,
    ) -> bool:
        """Evaluate DRS eligibility under FIA Sporting Regulations.

        Conditions for DRS activation:
        1. Gap to car ahead at detection point <= 1.000s.
        2. Race mode is GREEN (strictly disabled under Yellow / SC / VSC / Red Flag).
        3. Race lap >= 2 (enabled after race start or safety car restarts).
        4. Track condition is DRY (disabled by Race Director in damp/wet conditions).
        """
        if gap_at_detection > 1.000:
            return False
        if race_mode != RaceMode.GREEN:
            return False
        if lap < 2:
            return False
        if track_condition != TrackCondition.DRY:
            return False
        return True

    def compute_dirty_air_effect(
        self,
        gap_ahead: float,
        in_drs_sector: bool,
        has_drs: bool,
        config: CircuitDRSConfig,
    ) -> Tuple[float, float]:
        """Compute sector aerodynamic wake time delta and accelerated tyre wear multiplier.

        Args:
            gap_ahead: Gap in seconds to car directly ahead.
            in_drs_sector: Whether current sector contains a DRS straight.
            has_drs: Whether following car has DRS active.
            config: Circuit DRS profile.

        Returns:
            (sector_aero_delta_s, tyre_wear_multiplier):
                - sector_aero_delta_s: Additional latency from dirty air downforce loss,
                  minus straight-line DRS drag reduction if active.
                - tyre_wear_multiplier: Wear rate multiplier from lateral tyre slip (1.0 to 1.35x).
        """
        if gap_ahead > 1.80:
            # Clean air: no wake disturbance, no tyre wear penalty
            return 0.0, 1.0

        # Turbulent wake intensity decreases linearly from 1.0 at bumper to 0.0 at 1.8s
        wake_intensity = max(0.0, 1.0 - (gap_ahead / 1.80))

        # Cornering downforce loss in dirty air (adds 0.06s - 0.18s per sector)
        aero_cornering_loss = 0.15 * wake_intensity

        # DRS drag reduction if active in this sector
        drs_gain = config.drs_delta_per_zone_s if (in_drs_sector and has_drs) else 0.0

        net_sector_aero_delta = aero_cornering_loss - drs_gain

        # Lateral micro-slip in turbulent wake accelerates surface tyre heating & wear
        wear_multiplier = 1.0 + 0.30 * wake_intensity

        return net_sector_aero_delta, wear_multiplier

    def evaluate_defensive_rating(
        self,
        defender_utility: float,
        mean_utility: float,
        driver_cv: float = 0.003,
    ) -> float:
        """Compute a driver's defensive resistance delta based on rating and consistency.

        Top drivers (Verstappen, Alonso, Hamilton) defend more effectively (+0.15 to +0.25s threshold).
        Rookies or inconsistent drivers have lower defensive resistance (-0.10 to 0.0s).
        """
        rel_utility = defender_utility - mean_utility
        # Normalized tanh scale [-0.20s, +0.25s]
        base_def = float(0.22 * np.tanh(rel_utility / 1.2))
        # Consistency penalty (higher CV = more errors under pressure)
        cv_penalty = max(0.0, (driver_cv - 0.0025) * 20.0)
        return float(np.clip(base_def - cv_penalty, -0.15, 0.30))

    def attempt_overtake(
        self,
        attacker: CarState,
        defender: CarState,
        attacker_pace: float,
        defender_pace: float,
        gap_ahead: float,
        has_drs: bool,
        defender_has_drs: bool = False,
        sector: int = 1,
        config: Optional[CircuitDRSConfig] = None,
        defender_defensive_rating: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> OvertakeAttemptResult:
        """Evaluate a stochastic overtaking attempt in a sector with an overtaking zone.

        Conditions:
        - Must be within striking distance (gap <= 0.75s).
        - Attacker gains DRS straight-line boost if defender does not also have DRS.
          If defender also has DRS ('DRS train'), straight-line delta is neutralized.
        - Probability governed by net pace delta, tyre age difference, DRS advantage,
          circuit difficulty threshold, and defender defensive rating.
        """
        cfg = config or DEFAULT_DRS_CONFIG

        # Striking distance threshold (0.75s)
        if gap_ahead > 0.75:
            return OvertakeAttemptResult(
                attempted=False,
                success=False,
                attacker_id=attacker.driver_id,
                defender_id=defender.driver_id,
                sector=sector,
                prob_success=0.0,
                net_pace_delta=0.0,
                tyre_age_delta=defender.tyre_age - attacker.tyre_age,
                drs_active=has_drs,
                reason="Gap too large (>0.75s) for overtake move",
            )

        # 1. Base Pace Differential (positive when attacker is naturally faster)
        base_pace_delta = defender_pace - attacker_pace

        # 2. Tyre Grip Differential (fresh tyres vs worn tyres)
        tyre_age_diff = defender.tyre_age - attacker.tyre_age
        tyre_grip_delta = float(np.clip(tyre_age_diff * 0.035, -0.60, 1.20))

        # 3. DRS Advantage on straight:
        # If defender also has DRS (DRS train), straight-line delta drops to negligible ~0.05s
        if has_drs and not defender_has_drs:
            drs_advantage = 0.35
        elif has_drs and defender_has_drs:
            drs_advantage = 0.05  # DRS train neutralizing effect
        else:
            drs_advantage = 0.0

        # Total net pace advantage
        net_pace_delta = base_pace_delta + tyre_grip_delta + drs_advantage

        # 4. Check whether attacker actually launches a move
        # Requires positive or near-zero delta (not being dropped by faster car)
        if net_pace_delta < -0.20 and not has_drs and tyre_age_diff < 5:
            return OvertakeAttemptResult(
                attempted=False,
                success=False,
                attacker_id=attacker.driver_id,
                defender_id=defender.driver_id,
                sector=sector,
                prob_success=0.0,
                net_pace_delta=net_pace_delta,
                tyre_age_delta=tyre_age_diff,
                drs_active=has_drs,
                reason="Attacker lacks momentum/delta to launch pass",
            )

        # 5. Logistic Probability of Success
        scale = 0.35
        effective_threshold = cfg.overtake_threshold_s + defender_defensive_rating
        z = (net_pace_delta - effective_threshold) / scale

        # Sigmoid link
        prob_pass = float(1.0 / (1.0 + np.exp(-z)))
        prob_pass = float(np.clip(prob_pass, 0.02, 0.95))

        if deterministic:
            success = prob_pass >= 0.50
        else:
            rand_val = rng.uniform(0.0, 1.0) if rng is not None else 0.50
            success = rand_val < prob_pass

        reason = "Overtake succeeded" if success else "Defended / move rebuffed"

        return OvertakeAttemptResult(
            attempted=True,
            success=success,
            attacker_id=attacker.driver_id,
            defender_id=defender.driver_id,
            sector=sector,
            prob_success=prob_pass,
            net_pace_delta=net_pace_delta,
            tyre_age_delta=tyre_age_diff,
            drs_active=has_drs,
            reason=reason,
        )

    def evaluate_blue_flag(
        self,
        lead_car: CarState,
        lapped_car: CarState,
        gap: float,
    ) -> Tuple[float, float, bool]:
        """Apply FIA Appendix H Blue Flag yielding regulations.

        When a car on the lead lap approaches within 1.2s of a car >= 1 lap down,
        marshals wave blue flags. The lapped car must yield within 3 sectors.

        Returns:
            (lead_car_penalty_s, lapped_car_penalty_s, is_lapped):
                - lead_car_penalty_s: Minor traffic negotiation latency (+0.12s to +0.22s).
                - lapped_car_penalty_s: Yielding off the racing line (+0.70s to +1.10s).
                - is_lapped: True if blue flag condition met.
        """
        laps_diff = lead_car.laps_completed - lapped_car.laps_completed
        if laps_diff >= 1 and gap <= 1.20:
            return 0.16, 0.85, True
        return 0.0, 0.0, False
