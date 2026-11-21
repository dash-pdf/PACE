"""PACE: Collisions, Incidents, Debris & Damage Module (Stage 5).

Models:
1. Context-dependent incident generation (Lap 1 Turn 1 scramble, close overtake contact,
   dense midfield DRS trains, wet surface aquaplaning).
2. Multi-car collision dependencies (primary and secondary car damage coupling).
3. Aerodynamic and mechanical damage penalties affecting subsequent lap performance.
4. Carbon-fibre debris scattering across sectors and secondary puncture risk.
5. Neutralisation triggers (VSC, Safety Car, Red Flag requests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.simulation.state import CarState, CarStatus, RaceMode, RaceState, TrackCondition
from src.simulation.events import EventDispatcher, EventType, SimulationEvent


class IncidentSeverity(str, Enum):
    MINOR_AERO_DAMAGE = "minor_aero_damage"  # Front wing / endplate damage (+0.35s to +0.80s/lap)
    PUNCTURE = "puncture"                    # Instant deflation (limps to pit lane +35s)
    SUSPENSION_TERMINAL = "suspension_terminal"  # Terminal single-car DNF
    MULTI_CAR_COLLISION = "multi_car_collision"  # Both cars suffer terminal DNF


class NeutralisationTrigger(str, Enum):
    NONE = "none"
    LOCAL_YELLOW = "local_yellow"
    VSC = "vsc"
    SAFETY_CAR = "safety_car"
    RED_FLAG = "red_flag"


@dataclass
class IncidentResult:
    """Outcome of an incident occurrence."""

    occurred: bool
    severity: Optional[IncidentSeverity] = None
    primary_driver_id: str = ""
    secondary_driver_id: Optional[str] = None
    sector: int = 1
    debris_generated: bool = False
    neutralisation_recommended: NeutralisationTrigger = NeutralisationTrigger.NONE
    details: Dict[str, object] = field(default_factory=dict)


class IncidentEngine:
    """Evaluates context-dependent collision hazards, damage accumulation, and debris."""

    def __init__(
        self,
        base_lap_incident_rate: float = 0.0008,
        turn1_incident_prob: float = 0.055,
        overtake_contact_prob: float = 0.025,
    ):
        self.base_rate = base_lap_incident_rate
        self.turn1_prob = turn1_incident_prob
        self.overtake_prob = overtake_contact_prob

    def evaluate_turn1_scramble(
        self,
        race: RaceState,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> Optional[IncidentResult]:
        """Evaluate Lap 1 Turn 1 multi-car incident hazard under dense pack conditions."""
        if race.current_lap != 1 or deterministic:
            return None

        # Draw probability for Turn 1 chaos
        draw = rng.uniform(0.0, 1.0) if rng is not None else 1.0
        # Wet weather triples Turn 1 incident probability
        wet_mult = 2.5 if race.track_condition in (TrackCondition.DAMP, TrackCondition.WET) else 1.0
        effective_prob = self.turn1_prob * wet_mult

        if draw >= effective_prob:
            return None

        # Pick two cars in the mid-pack (P6 to P18) where congestion is maximum
        active_cars = [c for c in race.get_ordered_field() if c.status.is_active]
        if len(active_cars) < 4:
            return None

        midpack_cars = [c for c in active_cars if 4 <= c.track_position <= min(18, len(active_cars))]
        if len(midpack_cars) < 2:
            midpack_cars = active_cars

        idx1, idx2 = rng.choice(len(midpack_cars), size=2, replace=False) if rng is not None else (0, 1)
        c1 = midpack_cars[idx1]
        c2 = midpack_cars[idx2]

        # 30% chance of double DNF, 70% chance of front wing damage
        is_terminal = rng.uniform(0.0, 1.0) < 0.30 if rng is not None else False
        if is_terminal:
            severity = IncidentSeverity.MULTI_CAR_COLLISION
            c1.retire(lap=1, reason=CarStatus.DNF_INCIDENT)
            c2.retire(lap=1, reason=CarStatus.DNF_INCIDENT)
            neut = NeutralisationTrigger.SAFETY_CAR
        else:
            severity = IncidentSeverity.MINOR_AERO_DAMAGE
            c1.damage_factor = min(2.0, c1.damage_factor + 0.55)
            c2.damage_factor = min(2.0, c2.damage_factor + 0.40)
            neut = NeutralisationTrigger.VSC

        c1.incident_involvement_count += 1
        c2.incident_involvement_count += 1
        race.active_debris_sectors.add(1)

        return IncidentResult(
            occurred=True,
            severity=severity,
            primary_driver_id=c1.driver_id,
            secondary_driver_id=c2.driver_id,
            sector=1,
            debris_generated=True,
            neutralisation_recommended=neut,
            details={
                "incident_type": "turn1_standing_start_collision",
                "lap": 1,
                "c1": c1.driver_id,
                "c2": c2.driver_id,
                "is_terminal": is_terminal,
            },
        )

    def evaluate_overtake_contact(
        self,
        attacker: CarState,
        defender: CarState,
        gap_ahead: float,
        sector: int,
        race: RaceState,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> Optional[IncidentResult]:
        """Evaluate wheel-to-wheel contact during close-quarters overtaking attempts."""
        if deterministic or gap_ahead > 0.45:
            return None

        # Contact probability scaled by dampness and driver aggression
        wet_mult = 1.8 if race.track_condition in (TrackCondition.DAMP, TrackCondition.WET) else 1.0
        prob = self.overtake_prob * wet_mult

        draw = rng.uniform(0.0, 1.0) if rng is not None else 1.0
        if draw >= prob:
            return None

        # Collision occurred!
        draw_sev = rng.uniform(0.0, 1.0) if rng is not None else 0.5
        if draw_sev < 0.20:
            # Multi-car terminal crash
            severity = IncidentSeverity.MULTI_CAR_COLLISION
            attacker.retire(lap=race.current_lap, reason=CarStatus.DNF_INCIDENT)
            defender.retire(lap=race.current_lap, reason=CarStatus.DNF_INCIDENT)
            neut = NeutralisationTrigger.SAFETY_CAR
        elif draw_sev < 0.45:
            # Single car suspension damage (terminal for attacker or defender)
            severity = IncidentSeverity.SUSPENSION_TERMINAL
            victim = attacker if (rng is not None and rng.uniform(0.0, 1.0) < 0.5) else defender
            survivor = defender if victim == attacker else attacker
            victim.retire(lap=race.current_lap, reason=CarStatus.DNF_INCIDENT)
            survivor.damage_factor = min(2.0, survivor.damage_factor + 0.60)
            neut = NeutralisationTrigger.SAFETY_CAR
        elif draw_sev < 0.70:
            # Puncture from carbon wing contact
            severity = IncidentSeverity.PUNCTURE
            attacker.damage_factor = min(2.0, attacker.damage_factor + 0.45)
            # Defender suffers puncture, must limp to pits
            defender.elapsed_time += 32.0
            neut = NeutralisationTrigger.VSC
        else:
            # Front wing / floor endplate clipping
            severity = IncidentSeverity.MINOR_AERO_DAMAGE
            attacker.damage_factor = min(2.0, attacker.damage_factor + 0.45)
            defender.damage_factor = min(2.0, defender.damage_factor + 0.25)
            neut = NeutralisationTrigger.LOCAL_YELLOW

        attacker.incident_involvement_count += 1
        defender.incident_involvement_count += 1
        race.active_debris_sectors.add(sector)

        return IncidentResult(
            occurred=True,
            severity=severity,
            primary_driver_id=attacker.driver_id,
            secondary_driver_id=defender.driver_id,
            sector=sector,
            debris_generated=True,
            neutralisation_recommended=neut,
            details={
                "incident_type": "wheel_to_wheel_overtake_contact",
                "lap": race.current_lap,
                "sector": sector,
                "severity": severity.value,
            },
        )

    def evaluate_debris_hazard(
        self,
        car: CarState,
        sector: int,
        race: RaceState,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> bool:
        """Evaluate puncture risk when a car traverses a sector with active carbon debris."""
        if deterministic or sector not in race.active_debris_sectors:
            return False

        # Only applies during active green flag (under SC/VSC marshals clear or cars drive slow)
        if race.race_mode != RaceMode.GREEN:
            return False

        # 6% hazard of tyre puncture or floor damage from running over sharp carbon shards
        draw = rng.uniform(0.0, 1.0) if rng is not None else 1.0
        if draw < 0.06:
            car.damage_factor = min(2.0, car.damage_factor + 0.35)
            return True
        return False
