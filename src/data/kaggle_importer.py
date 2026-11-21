"""PACE: Kaggle Formula 1 Historical Dataset Importer & Reconciliation Engine.

Integrates the Kaggle Formula 1 World Championship historical dataset (Rohan Rao)
spanning 1950 through 2024 into PACE's existing historical data infrastructure.

Capabilities:
1. Kaggle API credential discovery and automated acquisition with local fallback.
2. Robust parsing of Kaggle relational CSV exports (with MySQL \\N null handling).
3. Canonical Entity Registry expansion for historical drivers, constructors, circuits, and events.
4. Schema conversion conforming strictly to PACE PyArrow specifications.
5. Referential-integrity validation and corrupted-record quarantine.
6. Source-precedence reconciliation merging Kaggle (1950-2022) with FastF1 (2023-2026).
7. Complete multi-era driver-circuit appearance and teammate matchup compilation.
8. Provable zero-leakage guarantee for historical predictive features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.canonical_registry import (
    CanonicalDriver,
    CanonicalConstructor,
    CanonicalCircuit,
    CanonicalEvent,
    CanonicalEntityRegistry,
    get_canonical_registry,
    resolve_driver,
    resolve_constructor,
    resolve_circuit,
    resolve_event,
    normalize_key,
)
from src.data.driver_circuit import get_regulation_era
from src.data.fastf1_collector import parse_time_to_ms
from src.data.schemas import (
    DRIVERS_SCHEMA,
    CONSTRUCTORS_SCHEMA,
    CIRCUITS_SCHEMA,
    EVENTS_SCHEMA,
    RACE_RESULTS_SCHEMA,
    QUALIFYING_RESULTS_SCHEMA,
    SPRINT_RESULTS_SCHEMA,
    LAPS_SCHEMA,
    PIT_STOPS_SCHEMA,
    DRIVER_CIRCUIT_APPEARANCES_SCHEMA,
    DRIVER_CIRCUIT_AGGREGATED_SCHEMA,
    DRIVER_STANDINGS_SCHEMA,
    CONSTRUCTOR_STANDINGS_SCHEMA,
)
from src.data.validation import (
    validate_driver_circuit_appearances,
    validate_driver_circuit_aggregated,
)
from src.data.storage import (
    save_parquet,
    load_parquet,
    dataset_exists,
    get_historical_path,
    get_driver_circuit_path,
    DEFAULT_BASE_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models & Reports
# ---------------------------------------------------------------------------

@dataclass
class KaggleTableInventory:
    """Summary of a single detected Kaggle source table."""
    table_name: str
    file_path: Path
    row_count: int
    column_count: int
    columns: List[str]
    file_size_bytes: int


@dataclass
class KaggleInventory:
    """Overall inventory of available Kaggle CSV files."""
    source_dir: Path
    is_valid: bool
    tables: Dict[str, KaggleTableInventory] = field(default_factory=dict)
    missing_required_tables: List[str] = field(default_factory=list)
    acquisition_mode: str = "local"           # 'kaggle_api', 'local_fallback', 'synthetic'
    dataset_version: str = "unknown"
    scanned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReconciliationConflict:
    """Details of a discrepancy detected between overlapping historical sources."""
    season: int
    round: int
    driver_id: str
    field_name: str
    fastf1_value: Any
    kaggle_value: Any
    resolved_value: Any
    resolution_rule: str


@dataclass
class ReconciliationReport:
    """Audit report produced when reconciling Kaggle records with FastF1 records."""
    total_fastf1_records: int = 0
    total_kaggle_records: int = 0
    overlap_count: int = 0
    identical_count: int = 0
    conflicts_detected: List[ReconciliationConflict] = field(default_factory=list)
    new_historical_records_added: int = 0
    reconciled_master_count: int = 0
    seasons_covered: List[int] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Quality assurance metrics evaluating schema conformity and integrity."""
    total_records_processed: int = 0
    valid_records_count: int = 0
    quarantined_records_count: int = 0
    quarantine_reasons: Dict[str, int] = field(default_factory=dict)
    referential_integrity_errors: int = 0
    duplicate_pk_errors: int = 0
    null_or_unresolved_canonical_entities: int = 0
    seasons_range: Tuple[int, int] = (0, 0)
    distinct_drivers_count: int = 0
    distinct_circuits_count: int = 0
    distinct_constructors_count: int = 0


@dataclass
class KaggleImportResult:
    """Comprehensive return bundle summarizing the entire Kaggle integration process."""
    success: bool
    inventory: KaggleInventory
    reconciliation: ReconciliationReport
    quality: DataQualityReport
    datasets_written: Dict[str, Path]
    summary_message: str


# ---------------------------------------------------------------------------
# Acquisition Manager
# ---------------------------------------------------------------------------

class KaggleAcquisitionManager:
    """Handles credentials checking and dataset retrieval from Kaggle."""

    KAGGLE_DATASET_ID = "rohanrao/formula-1-world-championship-1950-2020"

    REQUIRED_TABLES = [
        "races.csv",
        "drivers.csv",
        "constructors.csv",
        "circuits.csv",
        "results.csv",
        "status.csv",
    ]

    OPTIONAL_TABLES = [
        "qualifying.csv",
        "pit_stops.csv",
        "lap_times.csv",
        "driver_standings.csv",
        "constructor_standings.csv",
        "sprint_results.csv",
    ]

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.default_raw_dir = self.base_dir / "data" / "raw" / "kaggle"

    def check_kaggle_credentials(self) -> Tuple[bool, str]:
        """Verify whether Kaggle API credentials exist in environment or ~/.kaggle."""
        # 1. Environment variables
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True, "Found KAGGLE_USERNAME and KAGGLE_KEY in environment"

        # 2. Config file
        cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        cfg_file = cfg_dir / "kaggle.json"
        if cfg_file.exists() and cfg_file.stat().st_size > 0:
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("username") and data.get("key"):
                        return True, f"Found valid Kaggle credentials at {cfg_file}"
            except Exception as e:
                return False, f"Error reading {cfg_file}: {e}"

        return False, "No Kaggle API credentials found in environment or ~/.kaggle/kaggle.json"

    def acquire_dataset(self, target_dir: Optional[Path] = None, force_download: bool = False) -> Path:
        """Acquire Kaggle dataset via API or locate existing local directory."""
        dest = target_dir or self.default_raw_dir
        dest.mkdir(parents=True, exist_ok=True)

        # Check if already present and populated
        if not force_download and self.has_required_files(dest):
            logger.info(f"Kaggle dataset already present in {dest}")
            return dest

        has_creds, reason = self.check_kaggle_credentials()
        if not has_creds:
            logger.info(
                f"Kaggle API credentials not available ({reason}). "
                f"Using local directory: {dest}"
            )
            return dest

        try:
            import kaggle  # type: ignore
            logger.info(f"Downloading {self.KAGGLE_DATASET_ID} via Kaggle API to {dest}...")
            kaggle.api.dataset_download_files(self.KAGGLE_DATASET_ID, path=str(dest), unzip=True)
            logger.info("Kaggle dataset download and unzipping complete.")
            return dest
        except ImportError:
            logger.warning("Package 'kaggle' not installed. Falling back to local directory.")
            return dest
        except Exception as e:
            logger.warning(f"Failed to download Kaggle dataset via API: {e}. Falling back to local directory.")
            return dest

    def has_required_files(self, source_dir: Path) -> bool:
        """Check if required tables exist in target directory."""
        if not source_dir.exists():
            return False
        for req in self.REQUIRED_TABLES:
            if not (source_dir / req).exists():
                return False
        return True

    def scan_inventory(self, source_dir: Path) -> KaggleInventory:
        """Scan target directory and construct detailed metadata inventory."""
        tables: Dict[str, KaggleTableInventory] = {}
        missing: List[str] = []

        for req in self.REQUIRED_TABLES:
            p = source_dir / req
            if p.exists() and p.stat().st_size > 0:
                try:
                    df_head = pd.read_csv(p, nrows=2)
                    # Approximate row count
                    with open(p, "rb") as f:
                        line_count = sum(1 for _ in f) - 1
                    tables[req] = KaggleTableInventory(
                        table_name=req.replace(".csv", ""),
                        file_path=p,
                        row_count=max(0, line_count),
                        column_count=len(df_head.columns),
                        columns=list(df_head.columns),
                        file_size_bytes=p.stat().st_size,
                    )
                except Exception as e:
                    logger.warning(f"Error scanning {p}: {e}")
                    missing.append(req)
            else:
                missing.append(req)

        # Scan optional tables
        for opt in self.OPTIONAL_TABLES:
            p = source_dir / opt
            if p.exists() and p.stat().st_size > 0:
                try:
                    df_head = pd.read_csv(p, nrows=2)
                    with open(p, "rb") as f:
                        line_count = sum(1 for _ in f) - 1
                    tables[opt] = KaggleTableInventory(
                        table_name=opt.replace(".csv", ""),
                        file_path=p,
                        row_count=max(0, line_count),
                        column_count=len(df_head.columns),
                        columns=list(df_head.columns),
                        file_size_bytes=p.stat().st_size,
                    )
                except Exception:
                    pass

        is_valid = len(missing) == 0
        return KaggleInventory(
            source_dir=source_dir,
            is_valid=is_valid,
            tables=tables,
            missing_required_tables=missing,
            acquisition_mode="local" if source_dir.exists() else "unresolved",
            dataset_version="rohanrao_1950_2024",
        )


# ---------------------------------------------------------------------------
# Kaggle Historical Importer
# ---------------------------------------------------------------------------

class KaggleImporter:
    """Ingests, cleans, canonicalizes, and reconciles the Kaggle Formula 1 archive."""

    def __init__(
        self,
        source_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        registry: Optional[CanonicalEntityRegistry] = None,
    ):
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.manager = KaggleAcquisitionManager(base_dir=self.base_dir)
        self.source_dir = source_dir or self.manager.default_raw_dir
        self.registry = registry or get_canonical_registry()

        # Ingestion cache mappings
        self._circuit_id_map: Dict[int, str] = {}       # kaggle circuitId -> canonical_id
        self._constructor_id_map: Dict[int, str] = {}   # kaggle constructorId -> canonical_id
        self._driver_id_map: Dict[int, str] = {}        # kaggle driverId -> canonical_id
        self._status_id_map: Dict[int, str] = {}        # kaggle statusId -> status_text
        self._race_id_map: Dict[int, Dict[str, Any]] = {} # kaggle raceId -> race metadata

        self.quarantined_records: List[Dict[str, Any]] = []

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Read Kaggle CSV with automatic MySQL \\N null conversion."""
        p = self.source_dir / filename
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p, na_values=["\\N", r"\N", ""], keep_default_na=True)

    # -----------------------------------------------------------------------
    # Step 1: Ingest & Register Canonical Reference Entities
    # -----------------------------------------------------------------------

    def import_circuits(self) -> pd.DataFrame:
        """Parse circuits.csv, expand registry for historical circuits, return normalized DataFrame."""
        df_raw = self.load_csv("circuits.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=CIRCUITS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            k_id = int(row["circuitId"])
            c_ref = str(row.get("circuitRef", "")).strip()
            name = str(row.get("name", "")).strip()
            loc = str(row.get("location", "")).strip()
            country = str(row.get("country", "")).strip()
            lat = float(row.get("lat", 0.0)) if pd.notna(row.get("lat")) else 0.0
            lng = float(row.get("lng", 0.0)) if pd.notna(row.get("lng")) else 0.0
            url = str(row.get("url", "")).strip()

            # Resolve through Canonical Registry
            res = self.registry.resolve_circuit(c_ref, strict=False)
            if res.is_resolved:
                canon_id = str(res.canonical_id)
            else:
                # Register new historical circuit into registry
                canon_id = c_ref
                new_circuit = CanonicalCircuit(
                    canonical_id=canon_id,
                    name=name,
                    country=country,
                    locality=loc,
                    ergast_id=c_ref,
                    fastf1_id=c_ref,
                    lap_length_km=5.0,
                    typical_laps=60,
                    default_transit_loss_s=22.0,
                    aliases=[name.lower(), loc.lower()],
                )
                self.registry.register_circuit(new_circuit)

            self._circuit_id_map[k_id] = canon_id

            records.append({
                "circuit_id": canon_id,
                "circuit_name": name,
                "locality": loc,
                "country": country,
                "latitude": lat,
                "longitude": lng,
                "circuit_url": url,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=CIRCUITS_SCHEMA.names)
        return df.drop_duplicates(subset=["circuit_id"], keep="first")

    def import_constructors(self) -> pd.DataFrame:
        """Parse constructors.csv, register historical constructors, return normalized DataFrame."""
        df_raw = self.load_csv("constructors.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=CONSTRUCTORS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            k_id = int(row["constructorId"])
            c_ref = str(row.get("constructorRef", "")).strip()
            name = str(row.get("name", "")).strip()
            nat = str(row.get("nationality", "")).strip()
            url = str(row.get("url", "")).strip()

            # Resolve or register
            res = self.registry.resolve_constructor(c_ref, strict=False)
            if res.is_resolved:
                canon_id = str(res.canonical_id)
            else:
                canon_id = c_ref
                new_const = CanonicalConstructor(
                    canonical_id=canon_id,
                    name=name,
                    ergast_id=c_ref,
                    fastf1_name=name,
                    nationality=nat,
                    aliases=[name.lower()],
                    active_2026=False,
                )
                self.registry.register_constructor(new_const)

            self._constructor_id_map[k_id] = canon_id

            records.append({
                "constructor_id": canon_id,
                "constructor_name": name,
                "nationality": nat,
                "constructor_url": url,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=CONSTRUCTORS_SCHEMA.names)
        return df.drop_duplicates(subset=["constructor_id"], keep="first")

    def import_drivers(self) -> pd.DataFrame:
        """Parse drivers.csv, register historical drivers, return normalized DataFrame."""
        df_raw = self.load_csv("drivers.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=DRIVERS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            k_id = int(row["driverId"])
            d_ref = str(row.get("driverRef", "")).strip()
            code = str(row.get("code", "")).strip() if pd.notna(row.get("code")) else ""
            num_val = row.get("number")
            num = int(num_val) if pd.notna(num_val) else -1
            forename = str(row.get("forename", "")).strip()
            surname = str(row.get("surname", "")).strip()
            full_name = f"{forename} {surname}".strip()
            dob = str(row.get("dob", "")).strip() if pd.notna(row.get("dob")) else ""
            nat = str(row.get("nationality", "")).strip()
            url = str(row.get("url", "")).strip()

            # Resolve or register
            res = self.registry.resolve_driver(d_ref, strict=False)
            if res.is_resolved:
                canon_id = str(res.canonical_id)
            else:
                canon_id = d_ref
                new_driver = CanonicalDriver(
                    canonical_id=canon_id,
                    code=code if len(code) == 3 else "",
                    permanent_number=num,
                    full_name=full_name,
                    given_name=forename,
                    family_name=surname,
                    nationality=nat,
                    ergast_id=d_ref,
                    fastf1_code=code,
                    openf1_number=num if num > 0 else None,
                    aliases=[full_name.lower(), surname.lower()],
                    active_2026=False,
                )
                self.registry.register_driver(new_driver)

            self._driver_id_map[k_id] = canon_id

            records.append({
                "driver_id": canon_id,
                "driver_code": code,
                "driver_number": num,
                "given_name": forename,
                "family_name": surname,
                "full_name": full_name,
                "date_of_birth": dob,
                "nationality": nat,
                "driver_url": url,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=DRIVERS_SCHEMA.names)
        return df.drop_duplicates(subset=["driver_id"], keep="first")

    def import_status(self) -> Dict[int, str]:
        """Parse status.csv into a lookup dictionary."""
        df_status = self.load_csv("status.csv")
        if df_status.empty:
            return {}
        for _, row in df_status.iterrows():
            self._status_id_map[int(row["statusId"])] = str(row.get("status", "")).strip()
        return self._status_id_map

    # -----------------------------------------------------------------------
    # Step 2: Ingest Races and Events
    # -----------------------------------------------------------------------

    def import_races(self) -> pd.DataFrame:
        """Parse races.csv and return normalized EVENTS_SCHEMA DataFrame."""
        df_raw = self.load_csv("races.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=EVENTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            r_id = int(row["raceId"])
            season = int(row["year"])
            rnd = int(row["round"])
            c_raw_id = int(row["circuitId"])
            name = str(row.get("name", "")).strip()
            date = str(row.get("date", "")).strip() if pd.notna(row.get("date")) else ""
            url = str(row.get("url", "")).strip()

            c_id = self._circuit_id_map.get(c_raw_id, "unknown")

            self._race_id_map[r_id] = {
                "season": season,
                "round": rnd,
                "event_name": name,
                "circuit_id": c_id,
                "event_date": date,
            }

            records.append({
                "season": season,
                "round": rnd,
                "event_name": name,
                "circuit_id": c_id,
                "country": "",
                "event_date": date,
                "event_url": url,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=EVENTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round"], keep="first")

    # -----------------------------------------------------------------------
    # Step 3: Ingest Results & Qualifying
    # -----------------------------------------------------------------------

    def import_results(self) -> pd.DataFrame:
        """Parse results.csv into normalized RACE_RESULTS_SCHEMA DataFrame."""
        df_raw = self.load_csv("results.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=RACE_RESULTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            r_id = int(row["raceId"])
            d_raw_id = int(row["driverId"])
            c_raw_id = int(row["constructorId"])

            # Referential integrity check
            if r_id not in self._race_id_map or d_raw_id not in self._driver_id_map:
                self.quarantined_records.append({
                    "table": "results",
                    "reason": "Missing referential integrity (raceId or driverId not found)",
                    "raw_data": row.to_dict(),
                })
                continue

            race_info = self._race_id_map[r_id]
            season = race_info["season"]
            rnd = race_info["round"]
            driver_id = self._driver_id_map[d_raw_id]
            constructor_id = self._constructor_id_map.get(c_raw_id, "unknown")

            num_val = row.get("number")
            num = int(num_val) if pd.notna(num_val) else -1
            grid = int(row["grid"]) if pd.notna(row.get("grid")) else 0
            pos_val = row.get("position")
            pos = int(pos_val) if pd.notna(pos_val) else None
            pos_text = str(row.get("positionText", "")).strip()
            pts = float(row.get("points", 0.0)) if pd.notna(row.get("points")) else 0.0
            laps = int(row.get("laps", 0)) if pd.notna(row.get("laps")) else 0

            status_id = int(row.get("statusId", 0)) if pd.notna(row.get("statusId")) else 0
            status_text = self._status_id_map.get(status_id, "Finished")

            # Classification logic
            s_lower = status_text.lower()
            p_upper = pos_text.upper()
            is_dsq = p_upper == "D" or "disqualif" in s_lower or "dsq" in s_lower
            is_dns = p_upper == "W" or "did not start" in s_lower or "dns" in s_lower or "withdrew" in s_lower
            is_classified = (
                not is_dsq and not is_dns and (
                    s_lower.startswith("finished")
                    or "+ lap" in s_lower
                    or "+ laps" in s_lower
                    or s_lower.startswith("+")
                    or "lapped" in s_lower
                )
            )
            is_dnf = not is_classified and not is_dns and not is_dsq

            tot_ms_val = row.get("milliseconds")
            tot_ms = float(tot_ms_val) if pd.notna(tot_ms_val) else None
            time_str = str(row.get("time", "")).strip() if pd.notna(row.get("time")) else ""
            fl_rank_val = row.get("rank")
            fl_rank = int(fl_rank_val) if pd.notna(fl_rank_val) else None
            fl_time = str(row.get("fastestLapTime", "")).strip() if pd.notna(row.get("fastestLapTime")) else ""
            fl_spd_val = row.get("fastestLapSpeed")
            fl_spd = float(fl_spd_val) if pd.notna(fl_spd_val) else None

            # Driver code from registry
            d_entity = self.registry.get_driver(driver_id)
            d_code = d_entity.code if d_entity else ""

            records.append({
                "season": season,
                "round": rnd,
                "driver_id": driver_id,
                "constructor_id": constructor_id,
                "driver_number": num,
                "driver_code": d_code,
                "grid_position": grid,
                "finish_position": pos,
                "position_text": pos_text,
                "points": pts,
                "laps_completed": laps,
                "status": status_text,
                "is_classified": is_classified,
                "is_dnf": is_dnf,
                "is_dns": is_dns,
                "is_dsq": is_dsq,
                "total_time_ms": tot_ms,
                "total_time_str": time_str,
                "fastest_lap_rank": fl_rank,
                "fastest_lap_time": fl_time,
                "fastest_lap_speed": fl_spd,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=RACE_RESULTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def import_qualifying(self) -> pd.DataFrame:
        """Parse qualifying.csv into normalized QUALIFYING_RESULTS_SCHEMA DataFrame."""
        df_raw = self.load_csv("qualifying.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=QUALIFYING_RESULTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            r_id = int(row["raceId"])
            d_raw_id = int(row["driverId"])
            c_raw_id = int(row["constructorId"])

            if r_id not in self._race_id_map or d_raw_id not in self._driver_id_map:
                continue

            race_info = self._race_id_map[r_id]
            driver_id = self._driver_id_map[d_raw_id]
            constructor_id = self._constructor_id_map.get(c_raw_id, "unknown")

            pos = int(row["position"]) if pd.notna(row.get("position")) else 0
            num = int(row["number"]) if pd.notna(row.get("number")) else -1
            q1_str = str(row.get("q1", "")).strip() if pd.notna(row.get("q1")) else ""
            q2_str = str(row.get("q2", "")).strip() if pd.notna(row.get("q2")) else ""
            q3_str = str(row.get("q3", "")).strip() if pd.notna(row.get("q3")) else ""

            q1_ms = parse_time_to_ms(q1_str) if q1_str else None
            q2_ms = parse_time_to_ms(q2_str) if q2_str else None
            q3_ms = parse_time_to_ms(q3_str) if q3_str else None

            d_entity = self.registry.get_driver(driver_id)
            d_code = d_entity.code if d_entity else ""

            records.append({
                "season": race_info["season"],
                "round": race_info["round"],
                "driver_id": driver_id,
                "constructor_id": constructor_id,
                "driver_number": num,
                "driver_code": d_code,
                "qualifying_position": pos,
                "q1_time_str": q1_str,
                "q1_time_ms": q1_ms,
                "q2_time_str": q2_str,
                "q2_time_ms": q2_ms,
                "q3_time_str": q3_str,
                "q3_time_ms": q3_ms,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=QUALIFYING_RESULTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def import_driver_standings(self) -> pd.DataFrame:
        """Parse driver_standings.csv into DRIVER_STANDINGS_SCHEMA DataFrame."""
        df_raw = self.load_csv("driver_standings.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=DRIVER_STANDINGS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            r_id = int(row["raceId"])
            d_raw_id = int(row["driverId"])
            if r_id not in self._race_id_map or d_raw_id not in self._driver_id_map:
                continue

            race_info = self._race_id_map[r_id]
            driver_id = self._driver_id_map[d_raw_id]

            records.append({
                "season": race_info["season"],
                "round": race_info["round"],
                "driver_id": driver_id,
                "points": float(row.get("points", 0.0)) if pd.notna(row.get("points")) else 0.0,
                "position": int(row.get("position", 0)) if pd.notna(row.get("position")) else 0,
                "position_text": str(row.get("positionText", "")),
                "wins": int(row.get("wins", 0)) if pd.notna(row.get("wins")) else 0,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=DRIVER_STANDINGS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def import_constructor_standings(self) -> pd.DataFrame:
        """Parse constructor_standings.csv into CONSTRUCTOR_STANDINGS_SCHEMA DataFrame."""
        df_raw = self.load_csv("constructor_standings.csv")
        if df_raw.empty:
            return pd.DataFrame(columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for _, row in df_raw.iterrows():
            r_id = int(row["raceId"])
            c_raw_id = int(row["constructorId"])
            if r_id not in self._race_id_map or c_raw_id not in self._constructor_id_map:
                continue

            race_info = self._race_id_map[r_id]
            constructor_id = self._constructor_id_map[c_raw_id]

            records.append({
                "season": race_info["season"],
                "round": race_info["round"],
                "constructor_id": constructor_id,
                "points": float(row.get("points", 0.0)) if pd.notna(row.get("points")) else 0.0,
                "position": int(row.get("position", 0)) if pd.notna(row.get("position")) else 0,
                "position_text": str(row.get("positionText", "")),
                "wins": int(row.get("wins", 0)) if pd.notna(row.get("wins")) else 0,
                "source_api": "kaggle.rohanrao",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "constructor_id"], keep="first")


# ---------------------------------------------------------------------------
# Source Reconciliation Layer
# ---------------------------------------------------------------------------

class HistoricalDataReconciler:
    """Merges Kaggle historical records with FastF1 modern records using documented precedence."""

    MODERN_ERA_START_SEASON = 2023  # FastF1 is primary for 2023+

    def reconcile_race_results(
        self,
        df_fastf1: pd.DataFrame,
        df_kaggle: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, ReconciliationReport]:
        """Reconcile race results across FastF1 and Kaggle."""
        report = ReconciliationReport(
            total_fastf1_records=len(df_fastf1),
            total_kaggle_records=len(df_kaggle),
        )

        if df_fastf1.empty and df_kaggle.empty:
            return pd.DataFrame(columns=RACE_RESULTS_SCHEMA.names), report

        if df_fastf1.empty:
            report.new_historical_records_added = len(df_kaggle)
            report.reconciled_master_count = len(df_kaggle)
            report.seasons_covered = sorted(df_kaggle["season"].unique().tolist())
            return df_kaggle, report

        if df_kaggle.empty:
            report.reconciled_master_count = len(df_fastf1)
            report.seasons_covered = sorted(df_fastf1["season"].unique().tolist())
            return df_fastf1, report

        # 1. Non-overlapping older records (< 2023): pure Kaggle additions
        older_kaggle = df_kaggle[df_kaggle["season"] < self.MODERN_ERA_START_SEASON].copy()
        report.new_historical_records_added = len(older_kaggle)

        # 2. Modern records (>= 2023): FastF1 is authoritative, compare overlaps
        modern_fastf1 = df_fastf1[df_fastf1["season"] >= self.MODERN_ERA_START_SEASON].copy()
        modern_kaggle = df_kaggle[df_kaggle["season"] >= self.MODERN_ERA_START_SEASON].copy()

        # Build lookup for FastF1 records
        f1_map = {}
        for _, row in modern_fastf1.iterrows():
            k = (int(row["season"]), int(row["round"]), str(row["driver_id"]))
            f1_map[k] = row

        reconciled_modern_rows = list(modern_fastf1.to_dict(orient="records"))

        for _, k_row in modern_kaggle.iterrows():
            k = (int(k_row["season"]), int(k_row["round"]), str(k_row["driver_id"]))
            if k in f1_map:
                report.overlap_count += 1
                f_row = f1_map[k]

                # Compare fields: finish_position, grid_position, points, is_classified, is_dnf
                fields_to_compare = ["finish_position", "grid_position", "points", "is_classified", "is_dnf"]
                has_diff = False
                for f_name in fields_to_compare:
                    f_val = f_row.get(f_name)
                    k_val = k_row.get(f_name)
                    if pd.notna(f_val) and pd.notna(k_val) and f_val != k_val:
                        has_diff = True
                        report.conflicts_detected.append(ReconciliationConflict(
                            season=k[0],
                            round=k[1],
                            driver_id=k[2],
                            field_name=f_name,
                            fastf1_value=f_val,
                            kaggle_value=k_val,
                            resolved_value=f_val,
                            resolution_rule="FastF1 takes precedence for modern seasons (2023+)",
                        ))
                if not has_diff:
                    report.identical_count += 1
            else:
                # Driver in Kaggle not in FastF1 for modern era (e.g. reserve driver)
                reconciled_modern_rows.append(k_row.to_dict())

        df_modern_reconciled = pd.DataFrame(reconciled_modern_rows, columns=RACE_RESULTS_SCHEMA.names)
        combined = pd.concat([older_kaggle, df_modern_reconciled], ignore_index=True)
        master = combined.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")
        master = master.sort_values(by=["season", "round", "grid_position"], ascending=[True, True, True]).reset_index(drop=True)

        report.reconciled_master_count = len(master)
        report.seasons_covered = sorted(master["season"].unique().tolist())
        return master, report


# ---------------------------------------------------------------------------
# Historical Intelligence Expander
# ---------------------------------------------------------------------------

class HistoricalIntelligenceExpander:
    """Builds multi-decade driver-circuit appearances and head-to-head records."""

    def compile_driver_circuit_appearances(
        self,
        df_races: pd.DataFrame,
        df_results: pd.DataFrame,
        df_qualifying: Optional[pd.DataFrame] = None,
        df_constructors: Optional[pd.DataFrame] = None,
        existing_appearances: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Derive complete 1950-present driver-circuit appearance matrix with teammate matchups.
        
        Outputs all 38 columns adhering strictly to DRIVER_CIRCUIT_APPEARANCES_SCHEMA.
        """
        if df_results.empty or df_races.empty:
            return pd.DataFrame(columns=DRIVER_CIRCUIT_APPEARANCES_SCHEMA.names)

        # 1. Lookup map for races / events
        r_map: Dict[Tuple[int, int], Dict[str, str]] = {}
        for _, r in df_races.iterrows():
            k = (int(r["season"]), int(r["round"]))
            r_map[k] = {
                "circuit_id": str(r.get("circuit_id", "unknown")),
                "event_name": str(r.get("event_name", "")),
                "event_date": str(r.get("event_date", "")),
            }

        # 2. Lookup map for constructor names
        c_name_map: Dict[str, str] = {}
        if df_constructors is not None and not df_constructors.empty:
            for _, c in df_constructors.iterrows():
                c_name_map[str(c["constructor_id"])] = str(c.get("constructor_name", c["constructor_id"]))

        # 3. Lookup map for qualifying results
        q_map: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        if df_qualifying is not None and not df_qualifying.empty:
            for _, q in df_qualifying.iterrows():
                qk = (int(q["season"]), int(q["round"]), str(q["driver_id"]))
                q_map[qk] = {
                    "qualifying_position": int(q["qualifying_position"]) if pd.notna(q.get("qualifying_position")) and int(q.get("qualifying_position")) > 0 else None,
                    "q1_time_str": str(q.get("q1_time_str", "")).strip() if pd.notna(q.get("q1_time_str")) else "",
                    "q1_time_ms": float(q["q1_time_ms"]) if pd.notna(q.get("q1_time_ms")) else None,
                    "q2_time_str": str(q.get("q2_time_str", "")).strip() if pd.notna(q.get("q2_time_str")) else "",
                    "q2_time_ms": float(q["q2_time_ms"]) if pd.notna(q.get("q2_time_ms")) else None,
                    "q3_time_str": str(q.get("q3_time_str", "")).strip() if pd.notna(q.get("q3_time_str")) else "",
                    "q3_time_ms": float(q["q3_time_ms"]) if pd.notna(q.get("q3_time_ms")) else None,
                }

        results = df_results.copy()
        now = datetime.now(timezone.utc).isoformat()
        appearances: List[Dict[str, Any]] = []

        # Group by race event & constructor to resolve teammates
        groups = results.groupby(["season", "round", "constructor_id"])

        for (s, rnd, c_id), team_df in groups:
            season = int(s)
            round_num = int(rnd)
            constructor_id = str(c_id)
            constructor_name = c_name_map.get(constructor_id, constructor_id)

            r_info = r_map.get((season, round_num), {})
            circuit_id = r_info.get("circuit_id", "unknown")
            event_name = r_info.get("event_name", "")
            event_date = r_info.get("event_date", "")
            era = get_regulation_era(season)

            for _, d_row in team_df.iterrows():
                driver_id = str(d_row["driver_id"])
                grid = int(d_row["grid_position"]) if pd.notna(d_row.get("grid_position")) else 0
                fin = int(d_row["finish_position"]) if pd.notna(d_row.get("finish_position")) else None
                pos_text = str(d_row.get("position_text", ""))
                pts = float(d_row.get("points", 0.0)) if pd.notna(d_row.get("points")) else 0.0
                laps = int(d_row.get("laps_completed", 0)) if pd.notna(d_row.get("laps_completed")) else 0
                status = str(d_row.get("status", "Finished"))

                is_class = bool(d_row.get("is_classified", False))
                is_dnf = bool(d_row.get("is_dnf", False))
                is_dns = bool(d_row.get("is_dns", False))
                is_dsq = bool(d_row.get("is_dsq", False))

                # Positions gained/lost (positive = advanced, negative = fell back)
                pos_delta = (grid - fin) if (is_class and grid > 0 and fin is not None and fin > 0) else None

                # Fastest lap and race duration
                tot_time_ms = float(d_row.get("total_time_ms")) if pd.notna(d_row.get("total_time_ms")) else None
                fl_rank = int(d_row.get("fastest_lap_rank")) if pd.notna(d_row.get("fastest_lap_rank")) else None
                fl_time = str(d_row.get("fastest_lap_time", "")).strip() if pd.notna(d_row.get("fastest_lap_time")) else ""
                fl_spd = float(d_row.get("fastest_lap_speed")) if pd.notna(d_row.get("fastest_lap_speed")) else None

                # Qualifying data
                q_info = q_map.get((season, round_num, driver_id), {})
                q_pos = q_info.get("qualifying_position")
                q1_str = q_info.get("q1_time_str", "")
                q1_ms = q_info.get("q1_time_ms")
                q2_str = q_info.get("q2_time_str", "")
                q2_ms = q_info.get("q2_time_ms")
                q3_str = q_info.get("q3_time_str", "")
                q3_ms = q_info.get("q3_time_ms")

                # Find teammate in same constructor & event
                tm_rows = team_df[team_df["driver_id"] != driver_id]
                tm_id = None
                tm_grid = None
                tm_fin = None
                beat_tm_race = None
                beat_tm_qualy = None

                if not tm_rows.empty:
                    # Sort teammates: classified first, then best finish, then best grid
                    tm_sorted = tm_rows.sort_values(
                        by=["is_classified", "finish_position", "grid_position"],
                        ascending=[False, True, True],
                    )
                    tm_first = tm_sorted.iloc[0]
                    tm_id = str(tm_first["driver_id"])
                    tm_grid = int(tm_first["grid_position"]) if pd.notna(tm_first.get("grid_position")) and int(tm_first.get("grid_position")) > 0 else None
                    tm_fin = int(tm_first["finish_position"]) if pd.notna(tm_first.get("finish_position")) else None
                    tm_class = bool(tm_first.get("is_classified", False))

                    # Head-to-head race evaluation
                    if is_class and tm_class:
                        beat_tm_race = (fin < tm_fin) if (fin is not None and tm_fin is not None) else None
                    elif is_class and not tm_class:
                        beat_tm_race = True
                    elif not is_class and tm_class:
                        beat_tm_race = False
                    else:
                        beat_tm_race = None

                    # Head-to-head qualy evaluation
                    tm_q_info = q_map.get((season, round_num, tm_id), {})
                    tm_q_pos = tm_q_info.get("qualifying_position")
                    if q_pos is not None and tm_q_pos is not None and q_pos > 0 and tm_q_pos > 0:
                        beat_tm_qualy = (q_pos < tm_q_pos)
                    elif grid > 0 and tm_grid is not None and tm_grid > 0:
                        beat_tm_qualy = (grid < tm_grid)

                appearances.append({
                    "driver_id": driver_id,
                    "circuit_id": circuit_id,
                    "season": season,
                    "round": round_num,
                    "event_name": event_name,
                    "event_date": event_date,
                    "regulation_era": era,
                    "constructor_id": constructor_id,
                    "constructor_name": constructor_name,
                    "grid_position": grid,
                    "finish_position": fin,
                    "position_text": pos_text,
                    "positions_gained_lost": pos_delta,
                    "points": pts,
                    "laps_completed": laps,
                    "status": status,
                    "is_classified": is_class,
                    "is_dnf": is_dnf,
                    "is_dns": is_dns,
                    "is_dsq": is_dsq,
                    "qualifying_position": q_pos,
                    "q1_time_str": q1_str,
                    "q1_time_ms": q1_ms,
                    "q2_time_str": q2_str,
                    "q2_time_ms": q2_ms,
                    "q3_time_str": q3_str,
                    "q3_time_ms": q3_ms,
                    "fastest_lap_rank": fl_rank,
                    "fastest_lap_time": fl_time,
                    "fastest_lap_speed": fl_spd,
                    "total_race_time_ms": tot_time_ms,
                    "teammate_driver_id": tm_id,
                    "teammate_grid_position": tm_grid,
                    "teammate_finish_position": tm_fin,
                    "beat_teammate_race": beat_tm_race,
                    "beat_teammate_qualy": beat_tm_qualy,
                    "source_api": "kaggle_reconciled",
                    "retrieved_at": now,
                })

        df_new = pd.DataFrame(appearances, columns=DRIVER_CIRCUIT_APPEARANCES_SCHEMA.names)

        # Merge with existing appearances to preserve rich FastF1/Ergast metrics
        if existing_appearances is not None and not existing_appearances.empty:
            combined = pd.concat([existing_appearances, df_new], ignore_index=True)
            master_app = combined.drop_duplicates(subset=["driver_id", "season", "round"], keep="first")
        else:
            master_app = df_new.drop_duplicates(subset=["driver_id", "season", "round"], keep="first")

        master_app = master_app.sort_values(
            by=["season", "round", "driver_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        return master_app[DRIVER_CIRCUIT_APPEARANCES_SCHEMA.names]

    def compile_driver_circuit_aggregated(
        self,
        df_appearances: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive aggregated performance profiles across (driver_id, circuit_id).
        
        Outputs all 27 columns adhering strictly to DRIVER_CIRCUIT_AGGREGATED_SCHEMA.
        """
        if df_appearances.empty:
            return pd.DataFrame(columns=DRIVER_CIRCUIT_AGGREGATED_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        groups = df_appearances.groupby(["driver_id", "circuit_id"])

        for (d_id, c_id), g in groups:
            driver_id = str(d_id)
            circuit_id = str(c_id)
            total_starts = len(g)
            first_s = int(g["season"].min())
            latest_s = int(g["season"].max())

            classified_g = g[g["is_classified"] & (g["finish_position"] > 0)]
            valid_grid_g = g[g["grid_position"] > 0]

            wins = int((g["finish_position"] == 1).sum())
            podiums = int((g["finish_position"].between(1, 3)).sum())
            poles = int((g["grid_position"] == 1).sum())
            top_5 = int((g["finish_position"].between(1, 5)).sum())
            top_10 = int((g["finish_position"].between(1, 10)).sum())

            dnfs = int(g["is_dnf"].sum())
            dnf_rate = float(dnfs / total_starts) if total_starts > 0 else 0.0

            pts = float(g["points"].sum())
            avg_pts = float(pts / total_starts) if total_starts > 0 else 0.0

            avg_grid = float(valid_grid_g["grid_position"].mean()) if not valid_grid_g.empty else None
            avg_finish = float(classified_g["finish_position"].mean()) if not classified_g.empty else None

            gained_valid = g[g["positions_gained_lost"].notna()]
            avg_gained = float(gained_valid["positions_gained_lost"].mean()) if not gained_valid.empty else None

            best_fin = int(classified_g["finish_position"].min()) if not classified_g.empty else None
            best_grd = int(valid_grid_g["grid_position"].min()) if not valid_grid_g.empty else None

            # Teammate head-to-head metrics
            valid_race_h2h = g[g["beat_teammate_race"].notna()]
            race_h2h_wins = int((valid_race_h2h["beat_teammate_race"] == True).sum())
            race_h2h_rate = float(race_h2h_wins / len(valid_race_h2h)) if len(valid_race_h2h) > 0 else None

            valid_qualy_h2h = g[g["beat_teammate_qualy"].notna()]
            qualy_h2h_wins = int((valid_qualy_h2h["beat_teammate_qualy"] == True).sum())
            qualy_h2h_rate = float(qualy_h2h_wins / len(valid_qualy_h2h)) if len(valid_qualy_h2h) > 0 else None

            # Ground effect era (2022-2026) subset
            ge_g = g[g["season"] >= 2022]
            ge_starts = int(len(ge_g))
            ge_class = ge_g[ge_g["is_classified"] & (ge_g["finish_position"] > 0)]
            ge_avg_fin = float(ge_class["finish_position"].mean()) if not ge_class.empty else None

            records.append({
                "driver_id": driver_id,
                "circuit_id": circuit_id,
                "total_starts": total_starts,
                "first_season": first_s,
                "latest_season": latest_s,
                "wins": wins,
                "podiums": podiums,
                "poles": poles,
                "top_5_finishes": top_5,
                "top_10_finishes": top_10,
                "dnf_count": dnfs,
                "dnf_rate": dnf_rate,
                "total_points": pts,
                "avg_points_per_start": avg_pts,
                "avg_grid_position": avg_grid,
                "avg_finish_position": avg_finish,
                "avg_positions_gained": avg_gained,
                "best_finish_position": best_fin,
                "best_grid_position": best_grd,
                "teammate_race_h2h_wins": race_h2h_wins,
                "teammate_qualy_h2h_wins": qualy_h2h_wins,
                "teammate_race_h2h_rate": race_h2h_rate,
                "teammate_qualy_h2h_rate": qualy_h2h_rate,
                "ground_effect_starts": ge_starts,
                "ground_effect_avg_finish": ge_avg_fin,
                "source_api": "pace.kaggle_aggregated",
                "retrieved_at": now,
            })

        df_agg = pd.DataFrame(records, columns=DRIVER_CIRCUIT_AGGREGATED_SCHEMA.names)
        df_agg = df_agg.drop_duplicates(subset=["driver_id", "circuit_id"], keep="first")
        df_agg = df_agg.sort_values(by=["driver_id", "circuit_id"]).reset_index(drop=True)
        return df_agg[DRIVER_CIRCUIT_AGGREGATED_SCHEMA.names]

    @staticmethod
    def query_historical_profile(
        df_appearances: pd.DataFrame,
        driver_id: str,
        circuit_id: str,
        as_of_season: int,
        as_of_round: int,
    ) -> Optional[Dict[str, Any]]:
        """Query aggregated driver-circuit performance strictly prior to a target event.
        
        Provable zero-leakage guarantee:
        Only considers events where (season < as_of_season) or (season == as_of_season and round < as_of_round).
        """
        if df_appearances.empty:
            return None

        mask = (
            (df_appearances["driver_id"] == driver_id)
            & (df_appearances["circuit_id"] == circuit_id)
            & (
                (df_appearances["season"] < as_of_season)
                | (
                    (df_appearances["season"] == as_of_season)
                    & (df_appearances["round"] < as_of_round)
                )
            )
        )
        subset = df_appearances[mask]
        if subset.empty:
            return None

        total_starts = len(subset)
        classified = subset[subset["is_classified"] & (subset["finish_position"] > 0)]
        valid_grid = subset[subset["grid_position"] > 0]

        return {
            "driver_id": driver_id,
            "circuit_id": circuit_id,
            "total_starts": total_starts,
            "first_season": int(subset["season"].min()),
            "latest_season": int(subset["season"].max()),
            "wins": int((subset["finish_position"] == 1).sum()),
            "podiums": int((subset["finish_position"].between(1, 3)).sum()),
            "poles": int((subset["grid_position"] == 1).sum()),
            "top_5_finishes": int((subset["finish_position"].between(1, 5)).sum()),
            "top_10_finishes": int((subset["finish_position"].between(1, 10)).sum()),
            "dnf_count": int(subset["is_dnf"].sum()),
            "dnf_rate": float(subset["is_dnf"].sum() / total_starts),
            "avg_grid_position": float(valid_grid["grid_position"].mean()) if not valid_grid.empty else None,
            "avg_finish_position": float(classified["finish_position"].mean()) if not classified.empty else None,
            "best_finish_position": int(classified["finish_position"].min()) if not classified.empty else None,
            "best_grid_position": int(valid_grid["grid_position"].min()) if not valid_grid.empty else None,
        }


# ---------------------------------------------------------------------------
# High-Level Integration Orchestrator
# ---------------------------------------------------------------------------

def run_kaggle_historical_integration(
    source_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> KaggleImportResult:
    """Execute end-to-end Kaggle integration, validation, reconciliation, and intelligence expansion."""
    base = base_dir or DEFAULT_BASE_DIR
    importer = KaggleImporter(source_dir=source_dir, base_dir=base)
    inv = importer.manager.scan_inventory(importer.source_dir)

    if not inv.is_valid:
        # Check if Kaggle download is possible via API
        has_creds, msg = importer.manager.check_kaggle_credentials()
        if has_creds:
            importer.manager.acquire_dataset(importer.source_dir)
            inv = importer.manager.scan_inventory(importer.source_dir)

    if not inv.is_valid:
        return KaggleImportResult(
            success=False,
            inventory=inv,
            reconciliation=ReconciliationReport(),
            quality=DataQualityReport(),
            datasets_written={},
            summary_message=(
                f"Kaggle ingestion halted: missing required tables in {importer.source_dir}: "
                f"{inv.missing_required_tables}. See docs/KAGGLE_HISTORICAL_INTEGRATION.md."
            ),
        )

    # 1. Ingest Reference Tables & Expand Registry
    df_circuits = importer.import_circuits()
    df_constructors = importer.import_constructors()
    df_drivers = importer.import_drivers()
    importer.import_status()

    # Reconcile reference tables with existing datasets to avoid destructive overwrites
    circ_path = get_historical_path("circuits", base_dir=base)
    if dataset_exists(circ_path):
        df_ex_circ = load_parquet(circ_path)
        df_circuits = pd.concat([df_ex_circ, df_circuits], ignore_index=True).drop_duplicates(subset=["circuit_id"], keep="first")

    const_path = get_historical_path("constructors", base_dir=base)
    if dataset_exists(const_path):
        df_ex_const = load_parquet(const_path)
        df_constructors = pd.concat([df_ex_const, df_constructors], ignore_index=True).drop_duplicates(subset=["constructor_id"], keep="first")

    driv_path = get_historical_path("drivers", base_dir=base)
    if dataset_exists(driv_path):
        df_ex_d = load_parquet(driv_path)
        df_drivers = pd.concat([df_ex_d, df_drivers], ignore_index=True).drop_duplicates(subset=["driver_id"], keep="first")

    # 2. Ingest Races & Events
    df_races = importer.import_races()
    ev_path = get_historical_path("events", base_dir=base)
    if dataset_exists(ev_path):
        df_ex_ev = load_parquet(ev_path)
        df_races = pd.concat([df_ex_ev, df_races], ignore_index=True).drop_duplicates(subset=["season", "round"], keep="first")

    # 3. Ingest Results, Qualifying & Standings
    df_results = importer.import_results()
    df_qualifying = importer.import_qualifying()
    qual_path = get_historical_path("qualifying", base_dir=base)
    if not df_qualifying.empty and dataset_exists(qual_path):
        df_ex_q = load_parquet(qual_path)
        df_qualifying = pd.concat([df_ex_q, df_qualifying], ignore_index=True).drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    df_d_standings = importer.import_driver_standings()
    df_c_standings = importer.import_constructor_standings()

    # 4. Load Existing FastF1 Datasets for Reconciliation
    f1_results_path = get_historical_path("race_results", base_dir=base)
    df_f1_results = load_parquet(f1_results_path) if dataset_exists(f1_results_path) else pd.DataFrame()

    reconciler = HistoricalDataReconciler()
    master_results, recon_report = reconciler.reconcile_race_results(df_f1_results, df_results)

    # 5. Expand Historical Intelligence (Appearances & Teammates)
    app_path = get_driver_circuit_path("driver_circuit_appearances.parquet", base_dir=base)
    df_existing_app = load_parquet(app_path) if dataset_exists(app_path) else None

    expander = HistoricalIntelligenceExpander()
    df_appearances = expander.compile_driver_circuit_appearances(
        df_races=df_races,
        df_results=master_results,
        df_qualifying=df_qualifying,
        df_constructors=df_constructors,
        existing_appearances=df_existing_app,
    )
    df_aggregated = expander.compile_driver_circuit_aggregated(df_appearances)

    # 6. Quality Assurance & Validation
    min_s = int(master_results["season"].min()) if not master_results.empty else 0
    max_s = int(master_results["season"].max()) if not master_results.empty else 0
    quality = DataQualityReport(
        total_records_processed=len(df_results),
        valid_records_count=len(master_results),
        quarantined_records_count=len(importer.quarantined_records),
        seasons_range=(min_s, max_s),
        distinct_drivers_count=master_results["driver_id"].nunique() if not master_results.empty else 0,
        distinct_circuits_count=len(df_circuits),
        distinct_constructors_count=len(df_constructors),
    )

    # Validate driver circuit appearances and aggregated profiles
    val_app = validate_driver_circuit_appearances(df_appearances)
    val_agg = validate_driver_circuit_aggregated(df_aggregated)
    if not val_app.is_valid:
        logger.warning(f"Appearances validation warning: {val_app.summary()}")
    if not val_agg.is_valid:
        logger.warning(f"Aggregated validation warning: {val_agg.summary()}")

    # 7. Persistence (unless dry_run)
    written: Dict[str, Path] = {}
    if not dry_run:
        written["historical_race_results"] = save_parquet(
            master_results,
            get_historical_path("race_results", base_dir=base),
            schema=RACE_RESULTS_SCHEMA,
        )
        written["historical_circuits"] = save_parquet(
            df_circuits,
            get_historical_path("circuits", base_dir=base),
            schema=CIRCUITS_SCHEMA,
        )
        written["historical_constructors"] = save_parquet(
            df_constructors,
            get_historical_path("constructors", base_dir=base),
            schema=CONSTRUCTORS_SCHEMA,
        )
        written["historical_drivers"] = save_parquet(
            df_drivers,
            get_historical_path("drivers", base_dir=base),
            schema=DRIVERS_SCHEMA,
        )
        written["historical_events"] = save_parquet(
            df_races,
            get_historical_path("events", base_dir=base),
            schema=EVENTS_SCHEMA,
        )
        if not df_qualifying.empty:
            written["historical_qualifying"] = save_parquet(
                df_qualifying,
                get_historical_path("qualifying", base_dir=base),
                schema=QUALIFYING_RESULTS_SCHEMA,
            )
        if not df_d_standings.empty:
            written["historical_driver_standings"] = save_parquet(
                df_d_standings,
                get_historical_path("driver_standings", base_dir=base),
                schema=DRIVER_STANDINGS_SCHEMA,
            )
        if not df_c_standings.empty:
            written["historical_constructor_standings"] = save_parquet(
                df_c_standings,
                get_historical_path("constructor_standings", base_dir=base),
                schema=CONSTRUCTOR_STANDINGS_SCHEMA,
            )
        if not df_appearances.empty:
            written["driver_circuit_appearances"] = save_parquet(
                df_appearances,
                get_driver_circuit_path("driver_circuit_appearances.parquet", base_dir=base),
                schema=DRIVER_CIRCUIT_APPEARANCES_SCHEMA,
            )
        if not df_aggregated.empty:
            written["driver_circuit_aggregated"] = save_parquet(
                df_aggregated,
                get_driver_circuit_path("driver_circuit_aggregated.parquet", base_dir=base),
                schema=DRIVER_CIRCUIT_AGGREGATED_SCHEMA,
            )

    return KaggleImportResult(
        success=True,
        inventory=inv,
        reconciliation=recon_report,
        quality=quality,
        datasets_written=written,
        summary_message=(
            f"Successfully integrated Kaggle historical dataset: {len(master_results)} master results "
            f"spanning {min_s}-{max_s} across {quality.distinct_drivers_count} drivers and "
            f"{quality.distinct_constructors_count} constructors. "
            f"{len(df_appearances)} appearances and {len(df_aggregated)} aggregated profiles compiled."
        ),
    )
