"""Canonical Entity Registry for PACE.

Provides a unified, authoritative registry and deterministic resolution system
for Formula 1 drivers, constructors, circuits, and Grand Prix events across
all historical, modern, and future data sources (Ergast/Jolpica, FastF1, OpenF1,
TUM Racetrack Database, and Kaggle archives).
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Generic, TypeVar
import unicodedata

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CanonicalRegistryError(Exception):
    """Base exception for canonical entity registry errors."""
    pass


class UnresolvedEntityError(CanonicalRegistryError):
    """Raised when an entity identifier cannot be resolved to a canonical record."""

    def __init__(self, entity_type: str, query: Any, message: Optional[str] = None):
        self.entity_type = entity_type
        self.query = query
        msg = message or f"Unable to resolve {entity_type} identifier: '{query}'"
        super().__init__(msg)


class AmbiguousEntityError(CanonicalRegistryError):
    """Raised when a query matches multiple canonical entities ambiguously."""

    def __init__(self, entity_type: str, query: Any, matches: List[str], message: Optional[str] = None):
        self.entity_type = entity_type
        self.query = query
        self.matches = matches
        msg = message or f"Ambiguous {entity_type} query '{query}' matched multiple entities: {matches}"
        super().__init__(msg)


class EntityCollisionError(CanonicalRegistryError):
    """Raised when registering an alias or identity that collides with an existing entity."""
    pass


# ---------------------------------------------------------------------------
# Normalization Helper
# ---------------------------------------------------------------------------

def normalize_key(val: Any) -> str:
    """Normalize input string to a clean, canonical lookup key.
    
    Performs NFKD unicode decomposition (stripping accents such as é -> e, ü -> u),
    lowercasing, punctuation stripping, and whitespace-to-underscore normalization.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # Strip accents / diacritics
    s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s_clean = s_norm.lower()
    # Replace whitespace and hyphens with underscores
    s_clean = re.sub(r"[\s\-]+", "_", s_clean)
    # Remove any characters that are not alphanumeric or underscores
    s_clean = re.sub(r"[^\w]", "", s_clean)
    # Collapse multiple underscores
    s_clean = re.sub(r"_+", "_", s_clean).strip("_")
    return s_clean


# ---------------------------------------------------------------------------
# Entity Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalDriver:
    """Authoritative canonical driver definition."""
    canonical_id: str                      # e.g., 'max_verstappen', 'antonelli'
    code: str                              # 3-letter abbreviation, e.g., 'VER', 'ANT'
    permanent_number: int                  # Car number, e.g., 1 (or 33), 12, -1 if unassigned
    full_name: str                         # Display name, e.g., 'Max Verstappen'
    given_name: str                        # First name
    family_name: str                       # Surname
    nationality: str                       # e.g., 'Dutch', 'Italian'
    ergast_id: str                         # Ergast/Jolpica identifier
    fastf1_code: str                       # FastF1 abbreviation
    openf1_number: Optional[int] = None    # OpenF1 permanent/session number
    aliases: List[str] = field(default_factory=list)
    active_2026: bool = True               # Flag for 2026 championship participation


@dataclass(frozen=True)
class CanonicalConstructor:
    """Authoritative canonical constructor definition."""
    canonical_id: str                      # e.g., 'red_bull', 'sauber', 'rb'
    name: str                              # Display name, e.g., 'Red Bull Racing'
    ergast_id: str                         # Ergast/Jolpica identifier
    fastf1_name: str                       # FastF1 team name string
    nationality: str                       # e.g., 'Austrian', 'Swiss'
    aliases: List[str] = field(default_factory=list) # Historical rebrandings, sponsors
    active_2026: bool = True               # Participating in 2026


@dataclass(frozen=True)
class CanonicalCircuit:
    """Authoritative canonical circuit definition."""
    canonical_id: str                      # e.g., 'vegas', 'albert_park', 'marina_bay'
    name: str                              # Full name, e.g., 'Las Vegas Strip Street Circuit'
    country: str                           # Host country, e.g., 'USA'
    locality: str                          # City/locality, e.g., 'Las Vegas'
    ergast_id: str                         # Ergast/Jolpica identifier
    fastf1_id: str                         # FastF1 identifier
    lap_length_km: float                   # Length in kilometers
    typical_laps: int                      # Race distance in laps
    default_transit_loss_s: float          # Calibrated pit lane transit duration (seconds)
    tum_id: Optional[str] = None           # TUM Racetrack Database identifier
    openf1_key: Optional[str] = None       # OpenF1 circuit key or short name
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalEvent:
    """Authoritative Grand Prix event definition linking season, round and circuit."""
    canonical_id: str                      # Format: '{season}_{round:02d}_{circuit_id}'
    season: int                            # Championship season (e.g., 2026)
    round: int                             # Calendar round (e.g., 21)
    event_name: str                        # Grand Prix title (e.g., 'Las Vegas Grand Prix')
    circuit_id: str                        # Canonical circuit identifier (e.g., 'vegas')
    country: str                           # Host country
    session_ids: Dict[str, str] = field(default_factory=dict) # 'fp1', 'qualifying', 'race'


@dataclass(frozen=True)
class ResolutionResult(Generic[T]):
    """Container representing the outcome of an entity resolution lookup."""
    canonical_id: Optional[str]
    entity: Optional[T]
    is_resolved: bool
    query: Any
    strategy: str                          # 'exact_id', 'code', 'number', 'alias', 'normalized', 'unresolved'
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Canonical Entity Registry
# ---------------------------------------------------------------------------

class CanonicalEntityRegistry:
    """Central registry and identity resolution engine for PACE."""

    def __init__(self):
        # Driver indices
        self._drivers: Dict[str, CanonicalDriver] = {}
        self._driver_by_code: Dict[str, str] = {}
        self._driver_by_alias: Dict[str, str] = {}
        self._driver_ambiguous_aliases: Dict[str, Set[str]] = {}
        # (permanent_number, season) -> canonical_id; fallback number -> canonical_id
        self._driver_by_number_season: Dict[Tuple[int, int], str] = {}
        self._driver_by_number_default: Dict[int, str] = {}

        # Constructor indices
        self._constructors: Dict[str, CanonicalConstructor] = {}
        self._constructor_by_alias: Dict[str, str] = {}
        self._constructor_ambiguous_aliases: Dict[str, Set[str]] = {}

        # Circuit indices
        self._circuits: Dict[str, CanonicalCircuit] = {}
        self._circuit_by_alias: Dict[str, str] = {}
        self._circuit_ambiguous_aliases: Dict[str, Set[str]] = {}
        self._circuit_by_tum: Dict[str, str] = {}
        self._circuit_by_openf1: Dict[str, str] = {}

        # Event indices
        self._events: Dict[str, CanonicalEvent] = {}
        self._event_by_season_round: Dict[Tuple[int, int], str] = {}

        # Pre-populate built-in canonical catalog
        self._initialize_canonical_catalog()

    # -----------------------------------------------------------------------
    # Registration Methods
    # -----------------------------------------------------------------------

    def _index_driver_alias(self, norm_a: str, c_id: str) -> None:
        """Index a driver alias with collision and ambiguity tracking."""
        if not norm_a:
            return
        if norm_a in self._driver_ambiguous_aliases:
            self._driver_ambiguous_aliases[norm_a].add(c_id)
        elif norm_a in self._driver_by_alias:
            existing = self._driver_by_alias[norm_a]
            if existing != c_id:
                # Mark as ambiguous rather than silently overwriting
                self._driver_ambiguous_aliases[norm_a] = {existing, c_id}
                del self._driver_by_alias[norm_a]
        else:
            self._driver_by_alias[norm_a] = c_id

    def _index_constructor_alias(self, norm_a: str, c_id: str) -> None:
        """Index a constructor alias with collision and ambiguity tracking."""
        if not norm_a:
            return
        if norm_a in self._constructor_ambiguous_aliases:
            self._constructor_ambiguous_aliases[norm_a].add(c_id)
        elif norm_a in self._constructor_by_alias:
            existing = self._constructor_by_alias[norm_a]
            if existing != c_id:
                self._constructor_ambiguous_aliases[norm_a] = {existing, c_id}
                del self._constructor_by_alias[norm_a]
        else:
            self._constructor_by_alias[norm_a] = c_id

    def _index_circuit_alias(self, norm_a: str, c_id: str) -> None:
        """Index a circuit alias with collision and ambiguity tracking."""
        if not norm_a:
            return
        if norm_a in self._circuit_ambiguous_aliases:
            self._circuit_ambiguous_aliases[norm_a].add(c_id)
        elif norm_a in self._circuit_by_alias:
            existing = self._circuit_by_alias[norm_a]
            if existing != c_id:
                self._circuit_ambiguous_aliases[norm_a] = {existing, c_id}
                del self._circuit_by_alias[norm_a]
        else:
            self._circuit_by_alias[norm_a] = c_id

    def register_driver(self, driver: CanonicalDriver, season_numbers: Optional[Dict[int, int]] = None) -> None:
        """Register a canonical driver and construct bidirectional lookup indices."""
        c_id = driver.canonical_id
        if c_id in self._drivers:
            logger.debug(f"Overwriting driver registration for canonical ID '{c_id}'")

        self._drivers[c_id] = driver

        # Index by canonical_id normalized
        self._index_driver_alias(normalize_key(c_id), c_id)

        # Index by 3-letter code
        if driver.code:
            code_clean = driver.code.strip().upper()
            self._driver_by_code[code_clean] = c_id
            self._index_driver_alias(normalize_key(code_clean), c_id)

        # Index by Ergast ID
        if driver.ergast_id:
            self._index_driver_alias(normalize_key(driver.ergast_id), c_id)

        # Index by full name, given name + family name
        if driver.full_name:
            self._index_driver_alias(normalize_key(driver.full_name), c_id)
        if driver.family_name:
            self._index_driver_alias(normalize_key(driver.family_name), c_id)

        # Index by permanent number
        if driver.permanent_number > 0:
            self._driver_by_number_default[driver.permanent_number] = c_id
            # Also record for 2026 if active
            if driver.active_2026:
                self._driver_by_number_season[(driver.permanent_number, 2026)] = c_id

        # Index custom seasonal numbers if provided (e.g., number 1 vs 33 for Verstappen)
        if season_numbers:
            for s_num, yr in season_numbers.items():
                self._driver_by_number_season[(s_num, yr)] = c_id

        # Index all declared aliases
        for alias in driver.aliases:
            self._index_driver_alias(normalize_key(alias), c_id)

    def register_constructor(self, constructor: CanonicalConstructor) -> None:
        """Register a canonical constructor and construct alias indices."""
        c_id = constructor.canonical_id
        self._constructors[c_id] = constructor

        self._index_constructor_alias(normalize_key(c_id), c_id)
        if constructor.name:
            self._index_constructor_alias(normalize_key(constructor.name), c_id)
        if constructor.ergast_id:
            self._index_constructor_alias(normalize_key(constructor.ergast_id), c_id)
        if constructor.fastf1_name:
            self._index_constructor_alias(normalize_key(constructor.fastf1_name), c_id)

        for alias in constructor.aliases:
            self._index_constructor_alias(normalize_key(alias), c_id)

    def register_circuit(self, circuit: CanonicalCircuit) -> None:
        """Register a canonical circuit and construct alias indices."""
        c_id = circuit.canonical_id
        self._circuits[c_id] = circuit

        self._index_circuit_alias(normalize_key(c_id), c_id)
        if circuit.name:
            self._index_circuit_alias(normalize_key(circuit.name), c_id)
        if circuit.ergast_id:
            self._index_circuit_alias(normalize_key(circuit.ergast_id), c_id)
        if circuit.fastf1_id:
            self._index_circuit_alias(normalize_key(circuit.fastf1_id), c_id)
        if circuit.locality:
            self._index_circuit_alias(normalize_key(circuit.locality), c_id)

        if circuit.tum_id:
            self._circuit_by_tum[circuit.tum_id] = c_id
            self._index_circuit_alias(normalize_key(circuit.tum_id), c_id)

        if circuit.openf1_key:
            self._circuit_by_openf1[str(circuit.openf1_key)] = c_id
            self._index_circuit_alias(normalize_key(circuit.openf1_key), c_id)

        for alias in circuit.aliases:
            self._index_circuit_alias(normalize_key(alias), c_id)

    def register_event(self, event: CanonicalEvent) -> None:
        """Register a canonical Grand Prix event."""
        self._events[event.canonical_id] = event
        self._event_by_season_round[(event.season, event.round)] = event.canonical_id

    # -----------------------------------------------------------------------
    # Resolution Methods
    # -----------------------------------------------------------------------

    def resolve_driver(
        self,
        query: Union[str, int],
        season: Optional[int] = None,
        strict: bool = False,
    ) -> ResolutionResult[CanonicalDriver]:
        """Deterministically resolve any driver identifier, code, number, or alias."""
        if query is None:
            if strict:
                raise UnresolvedEntityError("driver", query, "Driver query cannot be None")
            return ResolutionResult(None, None, False, query, "unresolved", "Driver query is None")

        # 1. Integer permanent number resolution
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            num = int(query)
            c_id = None
            if season is not None:
                c_id = self._driver_by_number_season.get((num, season))
            if not c_id:
                c_id = self._driver_by_number_default.get(num)

            if c_id and c_id in self._drivers:
                return ResolutionResult(c_id, self._drivers[c_id], True, query, "number", None)

        q_str = str(query).strip()

        # 2. Exact match against canonical_id
        if q_str in self._drivers:
            return ResolutionResult(q_str, self._drivers[q_str], True, query, "exact_id", None)

        # 3. 3-letter driver code match
        if len(q_str) == 3 and q_str.upper() in self._driver_by_code:
            c_id = self._driver_by_code[q_str.upper()]
            return ResolutionResult(c_id, self._drivers[c_id], True, query, "code", None)

        # 4. Normalized alias / name match
        norm_q = normalize_key(q_str)
        if norm_q in self._driver_ambiguous_aliases:
            matches = sorted(list(self._driver_ambiguous_aliases[norm_q]))
            err_msg = f"Ambiguous driver query '{query}' matched multiple drivers: {matches}"
            if strict:
                raise AmbiguousEntityError("driver", query, matches, err_msg)
            return ResolutionResult(None, None, False, query, "ambiguous", err_msg)

        if norm_q in self._driver_by_alias:
            c_id = self._driver_by_alias[norm_q]
            return ResolutionResult(c_id, self._drivers[c_id], True, query, "alias", None)

        # Unresolved
        err_msg = f"Unknown driver identifier '{query}'"
        if strict:
            raise UnresolvedEntityError("driver", query, err_msg)
        return ResolutionResult(None, None, False, query, "unresolved", err_msg)

    def resolve_constructor(
        self,
        query: str,
        strict: bool = False,
    ) -> ResolutionResult[CanonicalConstructor]:
        """Deterministically resolve any constructor identifier, team name, or historical alias."""
        if not query or not str(query).strip():
            if strict:
                raise UnresolvedEntityError("constructor", query, "Constructor query cannot be empty")
            return ResolutionResult(None, None, False, query, "unresolved", "Constructor query is empty")

        q_str = str(query).strip()

        # 1. Exact match against canonical_id
        if q_str in self._constructors:
            return ResolutionResult(q_str, self._constructors[q_str], True, query, "exact_id", None)

        # 2. Normalized alias / name match
        norm_q = normalize_key(q_str)
        if norm_q in self._constructor_ambiguous_aliases:
            matches = sorted(list(self._constructor_ambiguous_aliases[norm_q]))
            err_msg = f"Ambiguous constructor query '{query}' matched multiple constructors: {matches}"
            if strict:
                raise AmbiguousEntityError("constructor", query, matches, err_msg)
            return ResolutionResult(None, None, False, query, "ambiguous", err_msg)

        if norm_q in self._constructor_by_alias:
            c_id = self._constructor_by_alias[norm_q]
            return ResolutionResult(c_id, self._constructors[c_id], True, query, "alias", None)

        err_msg = f"Unknown constructor identifier '{query}'"
        if strict:
            raise UnresolvedEntityError("constructor", query, err_msg)
        return ResolutionResult(None, None, False, query, "unresolved", err_msg)

    def resolve_circuit(
        self,
        query: str,
        strict: bool = False,
    ) -> ResolutionResult[CanonicalCircuit]:
        """Deterministically resolve any circuit identifier, grand prix name, city, or alias."""
        if not query or not str(query).strip():
            if strict:
                raise UnresolvedEntityError("circuit", query, "Circuit query cannot be empty")
            return ResolutionResult(None, None, False, query, "unresolved", "Circuit query is empty")

        q_str = str(query).strip()

        # 1. Exact match against canonical_id
        if q_str in self._circuits:
            return ResolutionResult(q_str, self._circuits[q_str], True, query, "exact_id", None)

        # 2. TUM database lookup
        if q_str in self._circuit_by_tum:
            c_id = self._circuit_by_tum[q_str]
            return ResolutionResult(c_id, self._circuits[c_id], True, query, "tum_id", None)

        # 3. OpenF1 circuit key lookup
        if q_str in self._circuit_by_openf1:
            c_id = self._circuit_by_openf1[q_str]
            return ResolutionResult(c_id, self._circuits[c_id], True, query, "openf1_key", None)

        # 4. Normalized alias / locality / grand prix name match
        norm_q = normalize_key(q_str)
        if norm_q in self._circuit_ambiguous_aliases:
            matches = sorted(list(self._circuit_ambiguous_aliases[norm_q]))
            err_msg = f"Ambiguous circuit query '{query}' matched multiple circuits: {matches}"
            if strict:
                raise AmbiguousEntityError("circuit", query, matches, err_msg)
            return ResolutionResult(None, None, False, query, "ambiguous", err_msg)

        if norm_q in self._circuit_by_alias:
            c_id = self._circuit_by_alias[norm_q]
            return ResolutionResult(c_id, self._circuits[c_id], True, query, "alias", None)

        err_msg = f"Unknown circuit identifier '{query}'"
        if strict:
            raise UnresolvedEntityError("circuit", query, err_msg)
        return ResolutionResult(None, None, False, query, "unresolved", err_msg)

    def resolve_event(
        self,
        season: int,
        round_num: Optional[int] = None,
        event_name: Optional[str] = None,
        circuit_id: Optional[str] = None,
        strict: bool = False,
    ) -> ResolutionResult[CanonicalEvent]:
        """Resolve a Grand Prix event by (season, round) or (season, circuit/name)."""
        # 1. Direct season and round lookup
        if round_num is not None:
            c_id = self._event_by_season_round.get((season, round_num))
            if c_id and c_id in self._events:
                return ResolutionResult(c_id, self._events[c_id], True, (season, round_num), "season_round", None)

        # 2. Lookup by circuit
        if circuit_id is not None:
            circ_res = self.resolve_circuit(circuit_id, strict=False)
            if circ_res.is_resolved:
                target_circuit = circ_res.canonical_id
                for ev in self._events.values():
                    if ev.season == season and ev.circuit_id == target_circuit:
                        return ResolutionResult(ev.canonical_id, ev, True, (season, circuit_id), "season_circuit", None)

        # 3. Lookup by event title match
        if event_name:
            norm_ev = normalize_key(event_name)
            for ev in self._events.values():
                if ev.season == season and norm_ev in normalize_key(ev.event_name):
                    return ResolutionResult(ev.canonical_id, ev, True, (season, event_name), "event_name", None)

        err_msg = f"Unknown event for season {season} (round={round_num}, circuit={circuit_id}, name={event_name})"
        if strict:
            raise UnresolvedEntityError("event", (season, round_num, circuit_id), err_msg)
        return ResolutionResult(None, None, False, (season, round_num), "unresolved", err_msg)

    # -----------------------------------------------------------------------
    # Direct Getters
    # -----------------------------------------------------------------------

    def get_driver(self, canonical_id: str) -> Optional[CanonicalDriver]:
        """Retrieve driver directly by canonical ID."""
        return self._drivers.get(canonical_id)

    def get_constructor(self, canonical_id: str) -> Optional[CanonicalConstructor]:
        """Retrieve constructor directly by canonical ID."""
        return self._constructors.get(canonical_id)

    def get_circuit(self, canonical_id: str) -> Optional[CanonicalCircuit]:
        """Retrieve circuit directly by canonical ID."""
        return self._circuits.get(canonical_id)

    def get_event(self, canonical_id: str) -> Optional[CanonicalEvent]:
        """Retrieve event directly by canonical ID."""
        return self._events.get(canonical_id)

    def list_drivers(self, active_only: bool = False) -> List[CanonicalDriver]:
        """List registered drivers."""
        if active_only:
            return [d for d in self._drivers.values() if d.active_2026]
        return list(self._drivers.values())

    def list_constructors(self, active_only: bool = False) -> List[CanonicalConstructor]:
        """List registered constructors."""
        if active_only:
            return [c for c in self._constructors.values() if c.active_2026]
        return list(self._constructors.values())

    def list_circuits(self) -> List[CanonicalCircuit]:
        """List registered circuits."""
        return list(self._circuits.values())

    def list_events(self, season: Optional[int] = None) -> List[CanonicalEvent]:
        """List registered Grand Prix events."""
        if season is not None:
            return [e for e in self._events.values() if e.season == season]
        return list(self._events.values())

    # -----------------------------------------------------------------------
    # Built-in Catalog Initializer
    # -----------------------------------------------------------------------

    def _initialize_canonical_catalog(self) -> None:
        """Populate the authoritative catalog of drivers, constructors, circuits and events."""
        # ===================================================================
        # 1. CIRCUITS (including explicit resolution for 'vegas')
        # ===================================================================
        circuits_data = [
            CanonicalCircuit(
                canonical_id="vegas",
                name="Las Vegas Strip Street Circuit",
                country="USA",
                locality="Las Vegas",
                ergast_id="vegas",
                fastf1_id="vegas",
                lap_length_km=6.201,
                typical_laps=50,
                default_transit_loss_s=20.5,
                tum_id="LasVegas",
                openf1_key="vegas",
                aliases=[
                    "las_vegas", "las vegas", "las-vegas", "vegas strip",
                    "las vegas strip", "las vegas strip street circuit", "lasvegas"
                ],
            ),
            CanonicalCircuit(
                canonical_id="marina_bay",
                name="Marina Bay Street Circuit",
                country="Singapore",
                locality="Singapore",
                ergast_id="marina_bay",
                fastf1_id="marina_bay",
                lap_length_km=4.940,
                typical_laps=62,
                default_transit_loss_s=28.0,
                tum_id="MarinaBay",
                openf1_key="marina_bay",
                aliases=[
                    "singapore", "singapore street circuit", "marina bay",
                    "marina bay street circuit", "singapore grand prix"
                ],
            ),
            CanonicalCircuit(
                canonical_id="albert_park",
                name="Albert Park Grand Prix Circuit",
                country="Australia",
                locality="Melbourne",
                ergast_id="albert_park",
                fastf1_id="albert_park",
                lap_length_km=5.278,
                typical_laps=58,
                default_transit_loss_s=20.5,
                tum_id="Melbourne",
                openf1_key="albert_park",
                aliases=["melbourne", "albert park", "australian gp", "australia"],
            ),
            CanonicalCircuit(
                canonical_id="americas",
                name="Circuit of the Americas",
                country="USA",
                locality="Austin",
                ergast_id="americas",
                fastf1_id="americas",
                lap_length_km=5.513,
                typical_laps=56,
                default_transit_loss_s=21.8,
                tum_id="Austin",
                openf1_key="americas",
                aliases=["cota", "austin", "circuit of the americas", "united states", "us gp"],
            ),
            CanonicalCircuit(
                canonical_id="bahrain",
                name="Bahrain International Circuit",
                country="Bahrain",
                locality="Sakhir",
                ergast_id="bahrain",
                fastf1_id="bahrain",
                lap_length_km=5.412,
                typical_laps=57,
                default_transit_loss_s=24.5,
                tum_id="Bahrain",
                openf1_key="bahrain",
                aliases=["sakhir", "bahrain international circuit", "bahrain gp"],
            ),
            CanonicalCircuit(
                canonical_id="baku",
                name="Baku City Circuit",
                country="Azerbaijan",
                locality="Baku",
                ergast_id="baku",
                fastf1_id="baku",
                lap_length_km=6.003,
                typical_laps=51,
                default_transit_loss_s=21.0,
                tum_id="Baku",
                openf1_key="baku",
                aliases=["baku city circuit", "azerbaijan", "azerbaijan gp"],
            ),
            CanonicalCircuit(
                canonical_id="catalunya",
                name="Circuit de Barcelona-Catalunya",
                country="Spain",
                locality="Montmeló",
                ergast_id="catalunya",
                fastf1_id="catalunya",
                lap_length_km=4.657,
                typical_laps=66,
                default_transit_loss_s=22.5,
                tum_id="Barcelona",
                openf1_key="catalunya",
                aliases=["barcelona", "montmelo", "circuit de catalunya", "spanish gp", "spain"],
            ),
            CanonicalCircuit(
                canonical_id="hungaroring",
                name="Hungaroring",
                country="Hungary",
                locality="Mogyoród",
                ergast_id="hungaroring",
                fastf1_id="hungaroring",
                lap_length_km=4.381,
                typical_laps=70,
                default_transit_loss_s=21.5,
                tum_id="Budapest",
                openf1_key="hungaroring",
                aliases=["budapest", "hungarian gp", "hungary"],
            ),
            CanonicalCircuit(
                canonical_id="imola",
                name="Autodromo Enzo e Dino Ferrari",
                country="Italy",
                locality="Imola",
                ergast_id="imola",
                fastf1_id="imola",
                lap_length_km=4.909,
                typical_laps=63,
                default_transit_loss_s=24.0,
                tum_id="Imola",
                openf1_key="imola",
                aliases=["autodromo enzo e dino ferrari", "emilia romagna", "san marino", "imola circuit"],
            ),
            CanonicalCircuit(
                canonical_id="interlagos",
                name="Autódromo José Carlos Pace",
                country="Brazil",
                locality="São Paulo",
                ergast_id="interlagos",
                fastf1_id="interlagos",
                lap_length_km=4.309,
                typical_laps=71,
                default_transit_loss_s=21.0,
                tum_id="Interlagos",
                openf1_key="interlagos",
                aliases=["sao paulo", "são paulo", "brazil", "brazilian gp", "autodromo jose carlos pace"],
            ),
            CanonicalCircuit(
                canonical_id="jeddah",
                name="Jeddah Corniche Circuit",
                country="Saudi Arabia",
                locality="Jeddah",
                ergast_id="jeddah",
                fastf1_id="jeddah",
                lap_length_km=6.174,
                typical_laps=50,
                default_transit_loss_s=20.0,
                tum_id="Jeddah",
                openf1_key="jeddah",
                aliases=["jeddah corniche circuit", "saudi arabia", "saudi gp"],
            ),
            CanonicalCircuit(
                canonical_id="losail",
                name="Lusail International Circuit",
                country="Qatar",
                locality="Lusail",
                ergast_id="losail",
                fastf1_id="losail",
                lap_length_km=5.419,
                typical_laps=57,
                default_transit_loss_s=23.0,
                tum_id="Lusail",
                openf1_key="losail",
                aliases=["lusail", "qatar", "lusail international circuit", "qatar gp"],
            ),
            CanonicalCircuit(
                canonical_id="madring",
                name="Madring",
                country="Spain",
                locality="Madrid",
                ergast_id="madring",
                fastf1_id="madring",
                lap_length_km=5.474,
                typical_laps=56,
                default_transit_loss_s=21.5,
                tum_id=None,
                openf1_key="madring",
                aliases=["madrid", "madring circuit", "madrid gp"],
            ),
            CanonicalCircuit(
                canonical_id="miami",
                name="Miami International Autodrome",
                country="USA",
                locality="Miami",
                ergast_id="miami",
                fastf1_id="miami",
                lap_length_km=5.412,
                typical_laps=57,
                default_transit_loss_s=20.2,
                tum_id="Miami",
                openf1_key="miami",
                aliases=["miami international autodrome", "hard rock stadium", "miami gp"],
            ),
            CanonicalCircuit(
                canonical_id="monaco",
                name="Circuit de Monaco",
                country="Monaco",
                locality="Monte-Carlo",
                ergast_id="monaco",
                fastf1_id="monaco",
                lap_length_km=3.337,
                typical_laps=78,
                default_transit_loss_s=22.0,
                tum_id="Monaco",
                openf1_key="monaco",
                aliases=["monte carlo", "circuit de monaco", "monaco gp"],
            ),
            CanonicalCircuit(
                canonical_id="monza",
                name="Autodromo Nazionale di Monza",
                country="Italy",
                locality="Monza",
                ergast_id="monza",
                fastf1_id="monza",
                lap_length_km=5.793,
                typical_laps=53,
                default_transit_loss_s=24.0,
                tum_id="Monza",
                openf1_key="monza",
                aliases=["monza circuit", "temple of speed", "italian gp", "italy"],
            ),
            CanonicalCircuit(
                canonical_id="red_bull_ring",
                name="Red Bull Ring",
                country="Austria",
                locality="Spielberg",
                ergast_id="red_bull_ring",
                fastf1_id="red_bull_ring",
                lap_length_km=4.318,
                typical_laps=71,
                default_transit_loss_s=20.0,
                tum_id="Spielberg",
                openf1_key="red_bull_ring",
                aliases=["spielberg", "austria", "austrian gp", "a1 ring", "osterreichring"],
            ),
            CanonicalCircuit(
                canonical_id="rodriguez",
                name="Autódromo Hermanos Rodríguez",
                country="Mexico",
                locality="Mexico City",
                ergast_id="rodriguez",
                fastf1_id="rodriguez",
                lap_length_km=4.304,
                typical_laps=71,
                default_transit_loss_s=22.5,
                tum_id="MexicoCity",
                openf1_key="rodriguez",
                aliases=["mexico", "mexico city", "hermanos rodriguez", "autodromo hermanos rodriguez", "mexican gp"],
            ),
            CanonicalCircuit(
                canonical_id="sepang",
                name="Sepang International Circuit",
                country="Malaysia",
                locality="Kuala Lumpur",
                ergast_id="sepang",
                fastf1_id="sepang",
                lap_length_km=5.543,
                typical_laps=56,
                default_transit_loss_s=22.0,
                tum_id="Sepang",
                openf1_key="sepang",
                aliases=["malaysia", "sepang circuit", "malaysian gp"],
            ),
            CanonicalCircuit(
                canonical_id="shanghai",
                name="Shanghai International Circuit",
                country="China",
                locality="Jiading",
                ergast_id="shanghai",
                fastf1_id="shanghai",
                lap_length_km=5.451,
                typical_laps=56,
                default_transit_loss_s=22.8,
                tum_id="Shanghai",
                openf1_key="shanghai",
                aliases=["china", "chinese gp", "shanghai international circuit"],
            ),
            CanonicalCircuit(
                canonical_id="silverstone",
                name="Silverstone Circuit",
                country="UK",
                locality="Silverstone",
                ergast_id="silverstone",
                fastf1_id="silverstone",
                lap_length_km=5.891,
                typical_laps=52,
                default_transit_loss_s=20.8,
                tum_id="Silverstone",
                openf1_key="silverstone",
                aliases=["british gp", "silverstone circuit", "great britain", "uk"],
            ),
            CanonicalCircuit(
                canonical_id="spa",
                name="Circuit de Spa-Francorchamps",
                country="Belgium",
                locality="Spa",
                ergast_id="spa",
                fastf1_id="spa",
                lap_length_km=7.004,
                typical_laps=44,
                default_transit_loss_s=22.0,
                tum_id="Spa",
                openf1_key="spa",
                aliases=["spa francorchamps", "belgium", "belgian gp", "circuit de spa-francorchamps"],
            ),
            CanonicalCircuit(
                canonical_id="suzuka",
                name="Suzuka Circuit",
                country="Japan",
                locality="Suzuka",
                ergast_id="suzuka",
                fastf1_id="suzuka",
                lap_length_km=5.807,
                typical_laps=53,
                default_transit_loss_s=22.2,
                tum_id="Suzuka",
                openf1_key="suzuka",
                aliases=["japan", "japanese gp", "suzuka circuit", "suzuka international racing course"],
            ),
            CanonicalCircuit(
                canonical_id="villeneuve",
                name="Circuit Gilles Villeneuve",
                country="Canada",
                locality="Montreal",
                ergast_id="villeneuve",
                fastf1_id="villeneuve",
                lap_length_km=4.361,
                typical_laps=70,
                default_transit_loss_s=18.5,
                tum_id="Montreal",
                openf1_key="villeneuve",
                aliases=["montreal", "canada", "canadian gp", "circuit gilles villeneuve", "gilles villeneuve"],
            ),
            CanonicalCircuit(
                canonical_id="yas_marina",
                name="Yas Marina Circuit",
                country="UAE",
                locality="Abu Dhabi",
                ergast_id="yas_marina",
                fastf1_id="yas_marina",
                lap_length_km=5.281,
                typical_laps=58,
                default_transit_loss_s=22.0,
                tum_id="YasMarina",
                openf1_key="yas_marina",
                aliases=["abu dhabi", "yas marina", "abu dhabi gp", "uae"],
            ),
            CanonicalCircuit(
                canonical_id="zandvoort",
                name="Circuit Park Zandvoort",
                country="Netherlands",
                locality="Zandvoort",
                ergast_id="zandvoort",
                fastf1_id="zandvoort",
                lap_length_km=4.259,
                typical_laps=72,
                default_transit_loss_s=19.5,
                tum_id="Zandvoort",
                openf1_key="zandvoort",
                aliases=["dutch gp", "netherlands", "circuit zandvoort", "zandvoort circuit"],
            ),
            # Historical circuits
            CanonicalCircuit(
                canonical_id="magny_cours",
                name="Circuit de Nevers Magny-Cours",
                country="France",
                locality="Magny-Cours",
                ergast_id="magny_cours",
                fastf1_id="magny_cours",
                lap_length_km=4.411,
                typical_laps=70,
                default_transit_loss_s=21.0,
                aliases=["magny cours", "french gp"],
            ),
            CanonicalCircuit(
                canonical_id="nurburgring",
                name="Nürburgring",
                country="Germany",
                locality="Nürburg",
                ergast_id="nurburgring",
                fastf1_id="nurburgring",
                lap_length_km=5.148,
                typical_laps=60,
                default_transit_loss_s=22.0,
                tum_id="Nuerburgring",
                aliases=["nurburgring", "european gp", "eifel gp"],
            ),
            CanonicalCircuit(
                canonical_id="hockenheimring",
                name="Hockenheimring",
                country="Germany",
                locality="Hockenheim",
                ergast_id="hockenheimring",
                fastf1_id="hockenheimring",
                lap_length_km=4.574,
                typical_laps=67,
                default_transit_loss_s=21.5,
                tum_id="Hockenheim",
                aliases=["hockenheim", "german gp"],
            ),
            CanonicalCircuit(
                canonical_id="portimao",
                name="Autódromo Internacional do Algarve",
                country="Portugal",
                locality="Portimão",
                ergast_id="portimao",
                fastf1_id="portimao",
                lap_length_km=4.653,
                typical_laps=66,
                default_transit_loss_s=21.0,
                aliases=["portimao", "algarve", "portuguese gp"],
            ),
            CanonicalCircuit(
                canonical_id="mugello",
                name="Autodromo Internazionale del Mugello",
                country="Italy",
                locality="Scarperia e San Piero",
                ergast_id="mugello",
                fastf1_id="mugello",
                lap_length_km=5.245,
                typical_laps=59,
                default_transit_loss_s=22.0,
                aliases=["mugello", "tuscan gp"],
            ),
            CanonicalCircuit(
                canonical_id="sochi",
                name="Sochi Autodrom",
                country="Russia",
                locality="Sochi",
                ergast_id="sochi",
                fastf1_id="sochi",
                lap_length_km=5.848,
                typical_laps=53,
                default_transit_loss_s=23.0,
                tum_id="Sochi",
                aliases=["sochi", "russian gp"],
            ),
        ]
        for c in circuits_data:
            self.register_circuit(c)

        # ===================================================================
        # 2. CONSTRUCTORS
        # ===================================================================
        constructors_data = [
            CanonicalConstructor(
                canonical_id="red_bull",
                name="Red Bull Racing",
                ergast_id="red_bull",
                fastf1_name="Red Bull",
                nationality="Austrian",
                aliases=["red bull", "red bull racing", "rbr", "red bull f1"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="ferrari",
                name="Scuderia Ferrari",
                ergast_id="ferrari",
                fastf1_name="Ferrari",
                nationality="Italian",
                aliases=["scuderia ferrari", "ferrari f1", "scuderia"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="mclaren",
                name="McLaren Formula 1 Team",
                ergast_id="mclaren",
                fastf1_name="McLaren",
                nationality="British",
                aliases=["mclaren", "mclaren f1", "mclaren racing"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="mercedes",
                name="Mercedes-AMG PETRONAS F1 Team",
                ergast_id="mercedes",
                fastf1_name="Mercedes",
                nationality="German",
                aliases=["mercedes", "mercedes amg", "mercedes-amg", "brawn", "brawn gp"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="aston_martin",
                name="Aston Martin Aramco F1 Team",
                ergast_id="aston_martin",
                fastf1_name="Aston Martin",
                nationality="British",
                aliases=[
                    "aston martin", "racing point", "force india", "sahara force india",
                    "jordan", "spyker", "midland"
                ],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="alpine",
                name="BWT Alpine F1 Team",
                ergast_id="alpine",
                fastf1_name="Alpine",
                nationality="French",
                aliases=["alpine", "alpine f1", "renault", "renault f1", "lotus f1", "lotus"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="williams",
                name="Williams Racing",
                ergast_id="williams",
                fastf1_name="Williams",
                nationality="British",
                aliases=["williams", "williams racing", "williams f1"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="haas",
                name="MoneyGram Haas F1 Team",
                ergast_id="haas",
                fastf1_name="Haas",
                nationality="American",
                aliases=["haas", "haas f1", "haas f1 team"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="sauber",
                name="Stake F1 Team Kick Sauber",
                ergast_id="sauber",
                fastf1_name="Sauber",
                nationality="Swiss",
                aliases=[
                    "sauber", "kick sauber", "stake f1", "stake",
                    "alfa", "alfa romeo", "alfa romeo racing", "bmw sauber"
                ],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="rb",
                name="Visa Cash App RB F1 Team",
                ergast_id="rb",
                fastf1_name="RB",
                nationality="Italian",
                aliases=[
                    "rb", "racing bulls", "vcarb", "alphatauri", "alpha tauri",
                    "toro rosso", "scuderia toro rosso", "minardi"
                ],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="cadillac",
                name="Cadillac Formula 1 Team",
                ergast_id="cadillac",
                fastf1_name="Cadillac",
                nationality="American",
                aliases=["cadillac", "cadillac f1", "andretti cadillac"],
                active_2026=True,
            ),
            CanonicalConstructor(
                canonical_id="audi",
                name="Audi Revolut F1 Team",
                ergast_id="audi",
                fastf1_name="Audi",
                nationality="German",
                aliases=["audi", "audi f1"],
                active_2026=True,
            ),
        ]
        for cn in constructors_data:
            self.register_constructor(cn)

        # ===================================================================
        # 3. DRIVERS
        # ===================================================================
        drivers_data = [
            CanonicalDriver(
                canonical_id="max_verstappen",
                code="VER",
                permanent_number=1,
                full_name="Max Verstappen",
                given_name="Max",
                family_name="Verstappen",
                nationality="Dutch",
                ergast_id="max_verstappen",
                fastf1_code="VER",
                openf1_number=1,
                aliases=["verstappen", "max verstappen", "m. verstappen"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="norris",
                code="NOR",
                permanent_number=4,
                full_name="Lando Norris",
                given_name="Lando",
                family_name="Norris",
                nationality="British",
                ergast_id="norris",
                fastf1_code="NOR",
                openf1_number=4,
                aliases=["norris", "lando norris", "l. norris"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="leclerc",
                code="LEC",
                permanent_number=16,
                full_name="Charles Leclerc",
                given_name="Charles",
                family_name="Leclerc",
                nationality="Monegasque",
                ergast_id="leclerc",
                fastf1_code="LEC",
                openf1_number=16,
                aliases=["leclerc", "charles leclerc", "c. leclerc"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="hamilton",
                code="HAM",
                permanent_number=44,
                full_name="Lewis Hamilton",
                given_name="Lewis",
                family_name="Hamilton",
                nationality="British",
                ergast_id="hamilton",
                fastf1_code="HAM",
                openf1_number=44,
                aliases=["hamilton", "lewis hamilton", "l. hamilton"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="antonelli",
                code="ANT",
                permanent_number=12,
                full_name="Andrea Kimi Antonelli",
                given_name="Andrea Kimi",
                family_name="Antonelli",
                nationality="Italian",
                ergast_id="antonelli",
                fastf1_code="ANT",
                openf1_number=12,
                aliases=["antonelli", "kimi antonelli", "andrea kimi antonelli"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="piastri",
                code="PIA",
                permanent_number=81,
                full_name="Oscar Piastri",
                given_name="Oscar",
                family_name="Piastri",
                nationality="Australian",
                ergast_id="piastri",
                fastf1_code="PIA",
                openf1_number=81,
                aliases=["piastri", "oscar piastri", "o. piastri"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="russell",
                code="RUS",
                permanent_number=63,
                full_name="George Russell",
                given_name="George",
                family_name="Russell",
                nationality="British",
                ergast_id="russell",
                fastf1_code="RUS",
                openf1_number=63,
                aliases=["russell", "george russell", "g. russell"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="sainz",
                code="SAI",
                permanent_number=55,
                full_name="Carlos Sainz",
                given_name="Carlos",
                family_name="Sainz",
                nationality="Spanish",
                ergast_id="sainz",
                fastf1_code="SAI",
                openf1_number=55,
                aliases=["sainz", "carlos sainz", "carlos sainz jr", "sainz jr"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="alonso",
                code="ALO",
                permanent_number=14,
                full_name="Fernando Alonso",
                given_name="Fernando",
                family_name="Alonso",
                nationality="Spanish",
                ergast_id="alonso",
                fastf1_code="ALO",
                openf1_number=14,
                aliases=["alonso", "fernando alonso", "f. alonso"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="gasly",
                code="GAS",
                permanent_number=10,
                full_name="Pierre Gasly",
                given_name="Pierre",
                family_name="Gasly",
                nationality="French",
                ergast_id="gasly",
                fastf1_code="GAS",
                openf1_number=10,
                aliases=["gasly", "pierre gasly", "p. gasly"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="tsunoda",
                code="TSU",
                permanent_number=22,
                full_name="Yuki Tsunoda",
                given_name="Yuki",
                family_name="Tsunoda",
                nationality="Japanese",
                ergast_id="tsunoda",
                fastf1_code="TSU",
                openf1_number=22,
                aliases=["tsunoda", "yuki tsunoda", "y. tsunoda"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="albon",
                code="ALB",
                permanent_number=23,
                full_name="Alexander Albon",
                given_name="Alexander",
                family_name="Albon",
                nationality="Thai",
                ergast_id="albon",
                fastf1_code="ALB",
                openf1_number=23,
                aliases=["albon", "alex albon", "alexander albon"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="ocon",
                code="OCO",
                permanent_number=31,
                full_name="Esteban Ocon",
                given_name="Esteban",
                family_name="Ocon",
                nationality="French",
                ergast_id="ocon",
                fastf1_code="OCO",
                openf1_number=31,
                aliases=["ocon", "esteban ocon", "e. ocon"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="stroll",
                code="STR",
                permanent_number=18,
                full_name="Lance Stroll",
                given_name="Lance",
                family_name="Stroll",
                nationality="Canadian",
                ergast_id="stroll",
                fastf1_code="STR",
                openf1_number=18,
                aliases=["stroll", "lance stroll", "l. stroll"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="hulkenberg",
                code="HUL",
                permanent_number=27,
                full_name="Nico Hülkenberg",
                given_name="Nico",
                family_name="Hülkenberg",
                nationality="German",
                ergast_id="hulkenberg",
                fastf1_code="HUL",
                openf1_number=27,
                aliases=["hulkenberg", "nico hulkenberg", "nico huelkenberg", "huelkenberg"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="bearman",
                code="BEA",
                permanent_number=87,
                full_name="Oliver Bearman",
                given_name="Oliver",
                family_name="Bearman",
                nationality="British",
                ergast_id="bearman",
                fastf1_code="BEA",
                openf1_number=87,
                aliases=["bearman", "oliver bearman", "ollie bearman"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="bortoleto",
                code="BOR",
                permanent_number=5,
                full_name="Gabriel Bortoleto",
                given_name="Gabriel",
                family_name="Bortoleto",
                nationality="Brazilian",
                ergast_id="bortoleto",
                fastf1_code="BOR",
                openf1_number=5,
                aliases=["bortoleto", "gabriel bortoleto"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="lawson",
                code="LAW",
                permanent_number=30,
                full_name="Liam Lawson",
                given_name="Liam",
                family_name="Lawson",
                nationality="New Zealander",
                ergast_id="lawson",
                fastf1_code="LAW",
                openf1_number=30,
                aliases=["lawson", "liam lawson", "l. lawson"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="colapinto",
                code="COL",
                permanent_number=43,
                full_name="Franco Colapinto",
                given_name="Franco",
                family_name="Colapinto",
                nationality="Argentine",
                ergast_id="colapinto",
                fastf1_code="COL",
                openf1_number=43,
                aliases=["colapinto", "franco colapinto"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="hadjar",
                code="HAD",
                permanent_number=6,
                full_name="Isack Hadjar",
                given_name="Isack",
                family_name="Hadjar",
                nationality="French",
                ergast_id="hadjar",
                fastf1_code="HAD",
                openf1_number=6,
                aliases=["hadjar", "isack hadjar"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="arvid_lindblad",
                code="LIN",
                permanent_number=41,
                full_name="Arvid Lindblad",
                given_name="Arvid",
                family_name="Lindblad",
                nationality="British",
                ergast_id="arvid_lindblad",
                fastf1_code="LIN",
                openf1_number=41,
                aliases=["lindblad", "arvid lindblad"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="bottas",
                code="BOT",
                permanent_number=77,
                full_name="Valtteri Bottas",
                given_name="Valtteri",
                family_name="Bottas",
                nationality="Finnish",
                ergast_id="bottas",
                fastf1_code="BOT",
                openf1_number=77,
                aliases=["bottas", "valtteri bottas", "v. bottas"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="perez",
                code="PER",
                permanent_number=11,
                full_name="Sergio Pérez",
                given_name="Sergio",
                family_name="Pérez",
                nationality="Mexican",
                ergast_id="perez",
                fastf1_code="PER",
                openf1_number=11,
                aliases=["perez", "sergio perez", "checo perez", "checo"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="ryo_hirakawa",
                code="HIR",
                permanent_number=50,
                full_name="Ryo Hirakawa",
                given_name="Ryo",
                family_name="Hirakawa",
                nationality="Japanese",
                ergast_id="ryo_hirakawa",
                fastf1_code="HIR",
                openf1_number=50,
                aliases=["hirakawa", "ryo hirakawa"],
                active_2026=True,
            ),
            CanonicalDriver(
                canonical_id="rafael_camara",
                code="CAM",
                permanent_number=51,
                full_name="Rafael Câmara",
                given_name="Rafael",
                family_name="Câmara",
                nationality="Brazilian",
                ergast_id="rafael_camara",
                fastf1_code="CAM",
                openf1_number=51,
                aliases=["camara", "rafael camara", "rafael camara"],
                active_2026=True,
            ),
            # Key Modern & Historical Drivers
            CanonicalDriver(
                canonical_id="zhou",
                code="ZHO",
                permanent_number=24,
                full_name="Guanyu Zhou",
                given_name="Guanyu",
                family_name="Zhou",
                nationality="Chinese",
                ergast_id="zhou",
                fastf1_code="ZHO",
                openf1_number=24,
                aliases=["zhou", "guanyu zhou", "zhou guanyu"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="kevin_magnussen",
                code="MAG",
                permanent_number=20,
                full_name="Kevin Magnussen",
                given_name="Kevin",
                family_name="Magnussen",
                nationality="Danish",
                ergast_id="kevin_magnussen",
                fastf1_code="MAG",
                openf1_number=20,
                aliases=["magnussen", "kevin magnussen", "k. magnussen"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="ricciardo",
                code="RIC",
                permanent_number=3,
                full_name="Daniel Ricciardo",
                given_name="Daniel",
                family_name="Ricciardo",
                nationality="Australian",
                ergast_id="ricciardo",
                fastf1_code="RIC",
                openf1_number=3,
                aliases=["ricciardo", "daniel ricciardo", "d. ricciardo"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="sargeant",
                code="SAR",
                permanent_number=2,
                full_name="Logan Sargeant",
                given_name="Logan",
                family_name="Sargeant",
                nationality="American",
                ergast_id="sargeant",
                fastf1_code="SAR",
                openf1_number=2,
                aliases=["sargeant", "logan sargeant"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="de_vries",
                code="DEV",
                permanent_number=21,
                full_name="Nyck de Vries",
                given_name="Nyck",
                family_name="de Vries",
                nationality="Dutch",
                ergast_id="de_vries",
                fastf1_code="DEV",
                openf1_number=21,
                aliases=["de vries", "nyck de vries"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="doohan",
                code="DOO",
                permanent_number=7,
                full_name="Jack Doohan",
                given_name="Jack",
                family_name="Doohan",
                nationality="Australian",
                ergast_id="doohan",
                fastf1_code="DOO",
                openf1_number=7,
                aliases=["doohan", "jack doohan"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="vettel",
                code="VET",
                permanent_number=5,
                full_name="Sebastian Vettel",
                given_name="Sebastian",
                family_name="Vettel",
                nationality="German",
                ergast_id="vettel",
                fastf1_code="VET",
                aliases=["vettel", "sebastian vettel", "s. vettel"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="raikkonen",
                code="RAI",
                permanent_number=7,
                full_name="Kimi Räikkönen",
                given_name="Kimi",
                family_name="Räikkönen",
                nationality="Finnish",
                ergast_id="raikkonen",
                fastf1_code="RAI",
                aliases=["raikkonen", "kimi raikkonen", "kimi raikkonen", "the iceman"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="button",
                code="BUT",
                permanent_number=22,
                full_name="Jenson Button",
                given_name="Jenson",
                family_name="Button",
                nationality="British",
                ergast_id="button",
                fastf1_code="BUT",
                aliases=["button", "jenson button"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="rosberg",
                code="ROS",
                permanent_number=6,
                full_name="Nico Rosberg",
                given_name="Nico",
                family_name="Rosberg",
                nationality="German",
                ergast_id="rosberg",
                fastf1_code="ROS",
                aliases=["rosberg", "nico rosberg"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="massa",
                code="MAS",
                permanent_number=19,
                full_name="Felipe Massa",
                given_name="Felipe",
                family_name="Massa",
                nationality="Brazilian",
                ergast_id="massa",
                fastf1_code="MAS",
                aliases=["massa", "felipe massa"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="latifi",
                code="LAT",
                permanent_number=6,
                full_name="Nicholas Latifi",
                given_name="Nicholas",
                family_name="Latifi",
                nationality="Canadian",
                ergast_id="latifi",
                fastf1_code="LAT",
                aliases=["latifi", "nicholas latifi"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="mick_schumacher",
                code="MSC",
                permanent_number=47,
                full_name="Mick Schumacher",
                given_name="Mick",
                family_name="Schumacher",
                nationality="German",
                ergast_id="mick_schumacher",
                fastf1_code="MSC",
                aliases=["mick schumacher", "schumacher"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="giovinazzi",
                code="GIO",
                permanent_number=99,
                full_name="Antonio Giovinazzi",
                given_name="Antonio",
                family_name="Giovinazzi",
                nationality="Italian",
                ergast_id="giovinazzi",
                fastf1_code="GIO",
                aliases=["giovinazzi", "antonio giovinazzi"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="kvyat",
                code="KVY",
                permanent_number=26,
                full_name="Daniil Kvyat",
                given_name="Daniil",
                family_name="Kvyat",
                nationality="Russian",
                ergast_id="kvyat",
                fastf1_code="KVY",
                aliases=["kvyat", "daniil kvyat"],
                active_2026=False,
            ),
            CanonicalDriver(
                canonical_id="grosjean",
                code="GRO",
                permanent_number=8,
                full_name="Romain Grosjean",
                given_name="Romain",
                family_name="Grosjean",
                nationality="French",
                ergast_id="grosjean",
                fastf1_code="GRO",
                aliases=["grosjean", "romain grosjean"],
                active_2026=False,
            ),
        ]
        for d in drivers_data:
            # Custom seasonal mapping overrides (e.g. Verstappen 33 pre-2022 vs 1 in 2022-2026)
            s_nums = {}
            if d.canonical_id == "max_verstappen":
                s_nums = {33: 2021, 1: 2026}
            elif d.canonical_id == "vettel":
                s_nums = {5: 2021}
            self.register_driver(d, season_numbers=s_nums)

        # ===================================================================
        # 4. 2026 CALENDAR EVENTS (including Las Vegas Round 21)
        # ===================================================================
        events_2026_data = [
            (1, "Australian Grand Prix", "albert_park", "Australia"),
            (2, "Chinese Grand Prix", "shanghai", "China"),
            (3, "Japanese Grand Prix", "suzuka", "Japan"),
            (4, "Miami Grand Prix", "miami", "USA"),
            (5, "Canadian Grand Prix", "villeneuve", "Canada"),
            (6, "Monaco Grand Prix", "monaco", "Monaco"),
            (7, "Barcelona Grand Prix", "catalunya", "Spain"),
            (8, "Austrian Grand Prix", "red_bull_ring", "Austria"),
            (9, "British Grand Prix", "silverstone", "UK"),
            (10, "Belgian Grand Prix", "spa", "Belgium"),
            (11, "Hungarian Grand Prix", "hungaroring", "Hungary"),
            (12, "Dutch Grand Prix", "zandvoort", "Netherlands"),
            (13, "Italian Grand Prix", "monza", "Italy"),
            (14, "Spanish Grand Prix", "madring", "Spain"),
            (15, "Azerbaijan Grand Prix", "baku", "Azerbaijan"),
            (16, "Bahrain Grand Prix in Malaysia", "sepang", "Malaysia"),
            (17, "Singapore Grand Prix", "marina_bay", "Singapore"),
            (18, "United States Grand Prix", "americas", "USA"),
            (19, "Mexico City Grand Prix", "rodriguez", "Mexico"),
            (20, "Brazilian Grand Prix", "interlagos", "Brazil"),
            (21, "Las Vegas Grand Prix", "vegas", "USA"),
            (22, "Qatar Grand Prix", "losail", "Qatar"),
            (23, "Abu Dhabi Grand Prix", "yas_marina", "UAE"),
        ]
        for rnd, name, c_id, country in events_2026_data:
            canon_event_id = f"2026_{rnd:02d}_{c_id}"
            ev = CanonicalEvent(
                canonical_id=canon_event_id,
                season=2026,
                round=rnd,
                event_name=name,
                circuit_id=c_id,
                country=country,
                session_ids={
                    "fp1": f"{canon_event_id}_fp1",
                    "fp2": f"{canon_event_id}_fp2",
                    "fp3": f"{canon_event_id}_fp3",
                    "qualifying": f"{canon_event_id}_qualifying",
                    "race": f"{canon_event_id}_race",
                },
            )
            self.register_event(ev)


# ---------------------------------------------------------------------------
# Module-level Singleton & Public API
# ---------------------------------------------------------------------------

_REGISTRY_INSTANCE: Optional[CanonicalEntityRegistry] = None


def get_canonical_registry() -> CanonicalEntityRegistry:
    """Obtain or initialize the process-wide canonical entity registry singleton."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = CanonicalEntityRegistry()
    return _REGISTRY_INSTANCE


def resolve_driver(
    query: Union[str, int],
    season: Optional[int] = None,
    strict: bool = False,
) -> ResolutionResult[CanonicalDriver]:
    """Resolve a driver query using the canonical registry."""
    return get_canonical_registry().resolve_driver(query, season=season, strict=strict)


def resolve_constructor(
    query: str,
    strict: bool = False,
) -> ResolutionResult[CanonicalConstructor]:
    """Resolve a constructor query using the canonical registry."""
    return get_canonical_registry().resolve_constructor(query, strict=strict)


def resolve_circuit(
    query: str,
    strict: bool = False,
) -> ResolutionResult[CanonicalCircuit]:
    """Resolve a circuit query using the canonical registry."""
    return get_canonical_registry().resolve_circuit(query, strict=strict)


def resolve_event(
    season: int,
    round_num: Optional[int] = None,
    event_name: Optional[str] = None,
    circuit_id: Optional[str] = None,
    strict: bool = False,
) -> ResolutionResult[CanonicalEvent]:
    """Resolve a Grand Prix event using the canonical registry."""
    return get_canonical_registry().resolve_event(
        season=season,
        round_num=round_num,
        event_name=event_name,
        circuit_id=circuit_id,
        strict=strict,
    )


def get_canonical_driver_id(query: Union[str, int], season: Optional[int] = None) -> str:
    """Return strictly resolved canonical driver ID or raise UnresolvedEntityError."""
    res = resolve_driver(query, season=season, strict=True)
    return str(res.canonical_id)


def get_canonical_constructor_id(query: str) -> str:
    """Return strictly resolved canonical constructor ID or raise UnresolvedEntityError."""
    res = resolve_constructor(query, strict=True)
    return str(res.canonical_id)


def get_canonical_circuit_id(query: str) -> str:
    """Return strictly resolved canonical circuit ID or raise UnresolvedEntityError."""
    res = resolve_circuit(query, strict=True)
    return str(res.canonical_id)
