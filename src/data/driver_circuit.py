"""Driver-Circuit Historical Intelligence module for PACE.

Collects, correlates, and aggregates the complete historical career record
for drivers across every circuit they have competed at, extending beyond recent seasons.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np

from fastf1.ergast import Ergast
from src.data.fastf1_collector import (
    extract_df,
    get_iso_timestamp,
    parse_time_to_ms,
    classify_status,
)
from src.data.schemas import (
    DRIVER_CIRCUIT_APPEARANCES_SCHEMA,
    DRIVER_CIRCUIT_AGGREGATED_SCHEMA,
    CIRCUIT_CHARACTERISTICS_SCHEMA,
)
from src.data.storage import (
    save_parquet,
    load_parquet,
    dataset_exists,
    get_driver_circuit_path,
    get_historical_path,
    DEFAULT_BASE_DIR,
)

logger = logging.getLogger(__name__)


def get_regulation_era(year: int) -> str:
    """Return the Formula 1 technical regulation era for a given championship year."""
    if year >= 2022:
        return "ground_effect_2022_2026"
    elif year >= 2014:
        return "turbo_hybrid_2014_2021"
    elif year >= 2006:
        return "v8_2006_2013"
    elif year >= 1996:
        return "v10_1996_2005"
    else:
        return "early_pre_1996"


# Curated engineering characteristics for all 39 Grand Prix circuits in historical appearances
CIRCUIT_METRICS: Dict[str, Dict[str, Any]] = {
    "albert_park": {"length_km": 5.278, "laps": 58, "overtake": "Medium", "tyre": "Medium", "downforce": "Medium", "alt_m": 10.0, "first_gp": 1996},
    "americas": {"length_km": 5.513, "laps": 56, "overtake": "Medium", "tyre": "High", "downforce": "High", "alt_m": 157.0, "first_gp": 2012},
    "bahrain": {"length_km": 5.412, "laps": 57, "overtake": "Low", "tyre": "High", "downforce": "Medium", "alt_m": 7.0, "first_gp": 2004},
    "baku": {"length_km": 6.003, "laps": 51, "overtake": "Low", "tyre": "Low", "downforce": "Medium", "alt_m": -28.0, "first_gp": 2016},
    "buddh": {"length_km": 5.125, "laps": 60, "overtake": "Medium", "tyre": "High", "downforce": "High", "alt_m": 190.0, "first_gp": 2011},
    "catalunya": {"length_km": 4.657, "laps": 66, "overtake": "Medium", "tyre": "Very High", "downforce": "High", "alt_m": 120.0, "first_gp": 1991},
    "fuji": {"length_km": 4.563, "laps": 67, "overtake": "Medium", "tyre": "High", "downforce": "Medium", "alt_m": 580.0, "first_gp": 1976},
    "hockenheimring": {"length_km": 4.574, "laps": 67, "overtake": "Medium", "tyre": "Medium", "downforce": "Medium", "alt_m": 103.0, "first_gp": 1970},
    "hungaroring": {"length_km": 4.381, "laps": 70, "overtake": "High", "tyre": "Medium", "downforce": "High", "alt_m": 238.0, "first_gp": 1986},
    "imola": {"length_km": 4.909, "laps": 63, "overtake": "High", "tyre": "Medium", "downforce": "High", "alt_m": 40.0, "first_gp": 1980},
    "indianapolis": {"length_km": 4.192, "laps": 73, "overtake": "Medium", "tyre": "Very High", "downforce": "Medium", "alt_m": 220.0, "first_gp": 1950},
    "interlagos": {"length_km": 4.309, "laps": 71, "overtake": "Low", "tyre": "Medium", "downforce": "Medium", "alt_m": 782.0, "first_gp": 1973},
    "istanbul": {"length_km": 5.338, "laps": 58, "overtake": "Medium", "tyre": "Very High", "downforce": "High", "alt_m": 130.0, "first_gp": 2005},
    "jeddah": {"length_km": 6.174, "laps": 50, "overtake": "Medium", "tyre": "Medium", "downforce": "Medium", "alt_m": 15.0, "first_gp": 2021},
    "losail": {"length_km": 5.419, "laps": 57, "overtake": "Medium", "tyre": "Very High", "downforce": "High", "alt_m": 12.0, "first_gp": 2021},
    "madring": {"length_km": 3.850, "laps": 70, "overtake": "High", "tyre": "Medium", "downforce": "High", "alt_m": 600.0, "first_gp": 1968},
    "magny_cours": {"length_km": 4.411, "laps": 70, "overtake": "High", "tyre": "Medium", "downforce": "High", "alt_m": 220.0, "first_gp": 1991},
    "marina_bay": {"length_km": 4.940, "laps": 62, "overtake": "High", "tyre": "High", "downforce": "High", "alt_m": 15.0, "first_gp": 2008},
    "miami": {"length_km": 5.412, "laps": 57, "overtake": "Medium", "tyre": "Medium", "downforce": "Medium", "alt_m": 5.0, "first_gp": 2022},
    "monaco": {"length_km": 3.337, "laps": 78, "overtake": "High", "tyre": "Low", "downforce": "High", "alt_m": 48.0, "first_gp": 1950},
    "monza": {"length_km": 5.793, "laps": 53, "overtake": "Low", "tyre": "High", "downforce": "Low", "alt_m": 162.0, "first_gp": 1950},
    "mugello": {"length_km": 5.245, "laps": 59, "overtake": "High", "tyre": "Very High", "downforce": "High", "alt_m": 292.0, "first_gp": 2020},
    "nurburgring": {"length_km": 5.148, "laps": 60, "overtake": "Medium", "tyre": "Medium", "downforce": "High", "alt_m": 620.0, "first_gp": 1951},
    "portimao": {"length_km": 4.653, "laps": 66, "overtake": "Medium", "tyre": "High", "downforce": "Medium", "alt_m": 108.0, "first_gp": 2020},
    "red_bull_ring": {"length_km": 4.318, "laps": 71, "overtake": "Low", "tyre": "Medium", "downforce": "Medium", "alt_m": 677.0, "first_gp": 1970},
    "ricard": {"length_km": 5.842, "laps": 53, "overtake": "Low", "tyre": "High", "downforce": "Medium", "alt_m": 430.0, "first_gp": 1971},
    "rodriguez": {"length_km": 4.304, "laps": 71, "overtake": "Medium", "tyre": "Low", "downforce": "High", "alt_m": 2240.0, "first_gp": 1963},
    "sepang": {"length_km": 5.543, "laps": 56, "overtake": "Low", "tyre": "Very High", "downforce": "High", "alt_m": 18.0, "first_gp": 1999},
    "shanghai": {"length_km": 5.451, "laps": 56, "overtake": "Medium", "tyre": "High", "downforce": "Medium", "alt_m": 10.0, "first_gp": 2004},
    "silverstone": {"length_km": 5.891, "laps": 52, "overtake": "Medium", "tyre": "Very High", "downforce": "High", "alt_m": 153.0, "first_gp": 1950},
    "sochi": {"length_km": 5.848, "laps": 53, "overtake": "Medium", "tyre": "Low", "downforce": "Medium", "alt_m": 2.0, "first_gp": 2014},
    "spa": {"length_km": 7.004, "laps": 44, "overtake": "Low", "tyre": "Very High", "downforce": "Medium", "alt_m": 420.0, "first_gp": 1950},
    "suzuka": {"length_km": 5.807, "laps": 53, "overtake": "High", "tyre": "Very High", "downforce": "High", "alt_m": 45.0, "first_gp": 1987},
    "valencia": {"length_km": 5.419, "laps": 57, "overtake": "Medium", "tyre": "Medium", "downforce": "Medium", "alt_m": 4.0, "first_gp": 2008},
    "vegas": {"length_km": 6.201, "laps": 50, "overtake": "Low", "tyre": "Low", "downforce": "Low", "alt_m": 610.0, "first_gp": 2023},
    "villeneuve": {"length_km": 4.361, "laps": 70, "overtake": "Low", "tyre": "Low", "downforce": "Low", "alt_m": 13.0, "first_gp": 1978},
    "yas_marina": {"length_km": 5.281, "laps": 58, "overtake": "Medium", "tyre": "Medium", "downforce": "Medium", "alt_m": 10.0, "first_gp": 2009},
    "yeongam": {"length_km": 5.615, "laps": 55, "overtake": "Medium", "tyre": "Medium", "downforce": "High", "alt_m": 5.0, "first_gp": 2010},
    "zandvoort": {"length_km": 4.259, "laps": 72, "overtake": "High", "tyre": "Very High", "downforce": "High", "alt_m": 15.0, "first_gp": 1952},
}

# Supplementary historical teammate drivers to ensure comprehensive head-to-head matching
KEY_HISTORICAL_TEAMMATES = [
    "vettel", "raikkonen", "button", "rosberg", "massa",
    "latifi", "mick_schumacher", "giovinazzi", "kvyat", "grosjean",
]


class DriverCircuitIntelligence:
    """Manages acquisition, correlation, and aggregation of complete driver-circuit histories."""

    def __init__(self, request_delay: float = 0.4, base_dir: Optional[Path] = None):
        self.ergast = Ergast()
        self.request_delay = request_delay
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.raw_dir = self.base_dir / "data" / "raw" / "driver_circuit"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _sleep(self):
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _safe_call(self, func, *args, **kwargs):
        """Execute Ergast API request with retry and exponential backoff on HTTP 429/rate limits."""
        max_retries = 6
        base_backoff = 2.0
        for attempt in range(max_retries):
            self._sleep()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "too many requests" in err_str or "rate limit" in err_str
                is_transient = is_rate_limit or "connection" in err_str or "timeout" in err_str or "reset" in err_str

                if is_transient and attempt < max_retries - 1:
                    sleep_time = base_backoff * (2 ** attempt)
                    logger.warning(
                        f"Transient Ergast API error for {func.__name__} ({e}). "
                        f"Sleeping {sleep_time:.1f}s before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Ergast call {func.__name__} failed on attempt {attempt + 1}: {e}")
                    raise
        return None

    def fetch_driver_career_races(self, driver_id: str, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch every historical race appearance for a driver across their entire career, with raw disk caching."""
        cache_file = self.raw_dir / f"{driver_id}_races.json"
        if not force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info(f"Loaded {len(cached_data)} cached career races for driver '{driver_id}' from raw storage.")
                return cached_data
            except Exception as e:
                logger.warning(f"Error reading cached races for {driver_id}: {e}. Re-fetching.")

        results = []
        offset = 0
        limit = 100

        while True:
            try:
                resp = self._safe_call(self.ergast.get_race_results, driver=driver_id, limit=limit, offset=offset)
                if resp is None or not hasattr(resp, "description") or resp.description is None or len(resp.description) == 0:
                    break

                desc_df = resp.description
                content_list = resp.content

                for idx, r in desc_df.iterrows():
                    race_df = content_list[idx] if idx < len(content_list) else pd.DataFrame()
                    if race_df.empty:
                        continue
                    # Driver's entry in this race
                    entry = race_df.iloc[0]

                    results.append({
                        "season": int(r.get("season", 0)),
                        "round": int(r.get("round", 0)),
                        "event_name": str(r.get("raceName", "")),
                        "circuit_id": str(r.get("circuitId", "")),
                        "event_date": str(r.get("raceDate", "")),
                        "driver_id": driver_id,
                        "constructor_id": str(entry.get("constructorId", "")),
                        "constructor_name": str(entry.get("constructorName", "")),
                        "grid_position": int(entry.get("grid", 0)) if pd.notna(entry.get("grid")) else 0,
                        "finish_position": int(entry.get("position", -1)) if str(entry.get("position", "")).isdigit() else -1,
                        "position_text": str(entry.get("positionText", "")),
                        "points": float(entry.get("points", 0.0)) if pd.notna(entry.get("points")) else 0.0,
                        "laps_completed": int(entry.get("laps", 0)) if pd.notna(entry.get("laps")) else 0,
                        "status": str(entry.get("status", "")),
                        "total_race_time_ms": float(entry.get("totalRaceTimeMillis")) if pd.notna(entry.get("totalRaceTimeMillis")) else None,
                        "fastest_lap_rank": int(entry.get("fastestLapRank", 0)) if str(entry.get("fastestLapRank", "")).isdigit() else 0,
                        "fastest_lap_time": str(entry.get("fastestLapTime", "")) if pd.notna(entry.get("fastestLapTime")) else "",
                        "fastest_lap_speed": float(entry.get("fastestLapAvgSpeed", 0.0)) if pd.notna(entry.get("fastestLapAvgSpeed")) else 0.0,
                    })

                count = len(desc_df)
                offset += count
                if count < limit:
                    break
            except Exception as e:
                logger.warning(f"Error fetching career races for {driver_id} at offset {offset}: {e}")
                break

        # Save to raw JSON cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save raw races cache for {driver_id}: {e}")

        return results

    def fetch_driver_career_qualifying(self, driver_id: str, force: bool = False) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Fetch every historical qualifying appearance for a driver keyed by (season, round), with raw disk caching."""
        cache_file = self.raw_dir / f"{driver_id}_qualy.json"
        if not force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    raw_cached = json.load(f)
                qualy_map = {}
                for k, v in raw_cached.items():
                    s_str, r_str = k.split("_")
                    qualy_map[(int(s_str), int(r_str))] = v
                logger.info(f"Loaded {len(qualy_map)} cached qualifying records for driver '{driver_id}' from raw storage.")
                return qualy_map
            except Exception as e:
                logger.warning(f"Error reading cached qualy for {driver_id}: {e}. Re-fetching.")

        qualy_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        offset = 0
        limit = 100

        while True:
            try:
                resp = self._safe_call(self.ergast.get_qualifying_results, driver=driver_id, limit=limit, offset=offset)
                if resp is None or not hasattr(resp, "description") or resp.description is None or len(resp.description) == 0:
                    break

                desc_df = resp.description
                content_list = resp.content

                for idx, r in desc_df.iterrows():
                    q_df = content_list[idx] if idx < len(content_list) else pd.DataFrame()
                    if q_df.empty:
                        continue
                    entry = q_df.iloc[0]
                    season = int(r.get("season", 0))
                    rnd = int(r.get("round", 0))

                    q1_str = str(entry.get("Q1", "")) if pd.notna(entry.get("Q1")) else ""
                    q2_str = str(entry.get("Q2", "")) if pd.notna(entry.get("Q2")) else ""
                    q3_str = str(entry.get("Q3", "")) if pd.notna(entry.get("Q3")) else ""
                    pos = int(entry.get("position", -1)) if str(entry.get("position", "")).isdigit() else -1

                    qualy_map[(season, rnd)] = {
                        "qualifying_position": pos,
                        "q1_time_str": q1_str,
                        "q1_time_ms": parse_time_to_ms(q1_str),
                        "q2_time_str": q2_str,
                        "q2_time_ms": parse_time_to_ms(q2_str),
                        "q3_time_str": q3_str,
                        "q3_time_ms": parse_time_to_ms(q3_str),
                    }

                count = len(desc_df)
                offset += count
                if count < limit:
                    break
            except Exception as e:
                logger.warning(f"Error fetching career qualifying for {driver_id} at offset {offset}: {e}")
                break

        # Save to raw JSON cache
        try:
            raw_to_save = {f"{k[0]}_{k[1]}": v for k, v in qualy_map.items()}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(raw_to_save, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save raw qualy cache for {driver_id}: {e}")

        return qualy_map

    def build_driver_circuit_appearances(
        self,
        driver_ids: Optional[List[str]] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Construct individual event-level historical appearance records for drivers across all circuits."""
        target_path = get_driver_circuit_path("driver_circuit_appearances.parquet", base_dir=self.base_dir)
        if not force and dataset_exists(target_path):
            logger.info("Driver-circuit appearances dataset already exists. Skipping download.")
            return load_parquet(target_path)

        if driver_ids is None:
            # Load active/historical drivers from historical drivers dataset
            drivers_path = get_historical_path("drivers", base_dir=self.base_dir)
            if dataset_exists(drivers_path):
                df_d = load_parquet(drivers_path)
                driver_ids = df_d["driver_id"].tolist()
            else:
                driver_ids = ["hamilton", "alonso", "max_verstappen", "leclerc", "norris", "russell", "sainz", "perez"]

        # Ensure key modern historical teammates are included for rich teammate H2H metrics
        full_driver_set: Set[str] = set(driver_ids).union(KEY_HISTORICAL_TEAMMATES)
        sorted_driver_ids = sorted(list(full_driver_set))

        logger.info(f"Collecting complete career circuit history for {len(sorted_driver_ids)} drivers...")
        all_appearances: List[Dict[str, Any]] = []
        now = get_iso_timestamp()

        # Step 1: Collect raw career appearances
        for d_id in sorted_driver_ids:
            logger.info(f"Querying complete career for driver: '{d_id}'...")
            races = self.fetch_driver_career_races(d_id, force=force)
            if not races:
                continue
            qualy_dict = self.fetch_driver_career_qualifying(d_id, force=force)

            for race in races:
                season = race["season"]
                rnd = race["round"]
                q_info = qualy_dict.get((season, rnd), {})

                grid = race["grid_position"]
                pos = race["finish_position"]
                status = race["status"]
                pos_text = race["position_text"]

                is_class, is_dnf, is_dns, is_dsq = classify_status(status, pos_text)

                # Positions gained/lost (grid - finish, positive = forward progress)
                pos_delta = (grid - pos) if (is_class and grid > 0 and pos > 0) else None

                all_appearances.append({
                    "driver_id": d_id,
                    "circuit_id": race["circuit_id"],
                    "season": season,
                    "round": rnd,
                    "event_name": race["event_name"],
                    "event_date": race["event_date"],
                    "regulation_era": get_regulation_era(season),
                    "constructor_id": race["constructor_id"],
                    "constructor_name": race["constructor_name"],
                    "grid_position": grid,
                    "finish_position": pos,
                    "position_text": pos_text,
                    "positions_gained_lost": pos_delta,
                    "points": race["points"],
                    "laps_completed": race["laps_completed"],
                    "status": status,
                    "is_classified": is_class,
                    "is_dnf": is_dnf,
                    "is_dns": is_dns,
                    "is_dsq": is_dsq,
                    "qualifying_position": q_info.get("qualifying_position"),
                    "q1_time_str": q_info.get("q1_time_str", ""),
                    "q1_time_ms": q_info.get("q1_time_ms"),
                    "q2_time_str": q_info.get("q2_time_str", ""),
                    "q2_time_ms": q_info.get("q2_time_ms"),
                    "q3_time_str": q_info.get("q3_time_str", ""),
                    "q3_time_ms": q_info.get("q3_time_ms"),
                    "fastest_lap_rank": race["fastest_lap_rank"],
                    "fastest_lap_time": race["fastest_lap_time"],
                    "fastest_lap_speed": race["fastest_lap_speed"],
                    "total_race_time_ms": race["total_race_time_ms"],
                    "teammate_driver_id": None,
                    "teammate_grid_position": None,
                    "teammate_finish_position": None,
                    "beat_teammate_race": None,
                    "beat_teammate_qualy": None,
                    "source_api": "fastf1.ergast",
                    "retrieved_at": now,
                })

        df_app = pd.DataFrame(all_appearances)
        if df_app.empty:
            logger.warning("No driver appearances retrieved.")
            save_parquet(df_app, target_path, schema=DRIVER_CIRCUIT_APPEARANCES_SCHEMA)
            return df_app

        # Deduplicate appearances if any duplicates exist
        df_app = df_app.drop_duplicates(subset=["driver_id", "season", "round"], keep="last").reset_index(drop=True)

        # Step 2: Compute teammate-relative performance
        team_groups = df_app.groupby(["season", "round", "constructor_id"])
        teammate_data: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

        for (season, rnd, const_id), group in team_groups:
            if len(group) == 2:
                row_a = group.iloc[0]
                row_b = group.iloc[1]
                id_a, id_b = row_a["driver_id"], row_b["driver_id"]

                # Race H2H
                pos_a, pos_b = row_a["finish_position"], row_b["finish_position"]
                class_a, class_b = row_a["is_classified"], row_b["is_classified"]
                grid_a, grid_b = row_a["grid_position"], row_b["grid_position"]

                beat_race_a = None
                beat_race_b = None
                if class_a and class_b:
                    beat_race_a = (pos_a < pos_b)
                    beat_race_b = (pos_b < pos_a)
                elif class_a and not class_b:
                    beat_race_a = True
                    beat_race_b = False
                elif not class_a and class_b:
                    beat_race_a = False
                    beat_race_b = True

                # Qualy H2H
                q_pos_a = row_a["qualifying_position"]
                q_pos_b = row_b["qualifying_position"]
                beat_q_a = None
                beat_q_b = None
                if pd.notna(q_pos_a) and pd.notna(q_pos_b) and q_pos_a > 0 and q_pos_b > 0:
                    beat_q_a = (q_pos_a < q_pos_b)
                    beat_q_b = (q_pos_b < q_pos_a)

                teammate_data[(id_a, season, rnd)] = {
                    "teammate_driver_id": id_b,
                    "teammate_grid_position": grid_b,
                    "teammate_finish_position": pos_b,
                    "beat_teammate_race": beat_race_a,
                    "beat_teammate_qualy": beat_q_a,
                }
                teammate_data[(id_b, season, rnd)] = {
                    "teammate_driver_id": id_a,
                    "teammate_grid_position": grid_a,
                    "teammate_finish_position": pos_a,
                    "beat_teammate_race": beat_race_b,
                    "beat_teammate_qualy": beat_q_b,
                }

        # Apply teammate metrics back to appearances
        for i, row in df_app.iterrows():
            key = (row["driver_id"], row["season"], row["round"])
            if key in teammate_data:
                info = teammate_data[key]
                df_app.at[i, "teammate_driver_id"] = info["teammate_driver_id"]
                df_app.at[i, "teammate_grid_position"] = info["teammate_grid_position"]
                df_app.at[i, "teammate_finish_position"] = info["teammate_finish_position"]
                df_app.at[i, "beat_teammate_race"] = info["beat_teammate_race"]
                df_app.at[i, "beat_teammate_qualy"] = info["beat_teammate_qualy"]

        save_parquet(df_app, target_path, schema=DRIVER_CIRCUIT_APPEARANCES_SCHEMA)
        logger.info(f"Saved {len(df_app)} driver-circuit appearances to {target_path}")
        return df_app

    def build_driver_circuit_aggregated(
        self,
        df_appearances: Optional[pd.DataFrame] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Calculate aggregated driver-circuit intelligence profiles from event appearances."""
        target_path = get_driver_circuit_path("driver_circuit_aggregated.parquet", base_dir=self.base_dir)
        if not force and dataset_exists(target_path):
            logger.info("Driver-circuit aggregated dataset already exists. Skipping calculation.")
            return load_parquet(target_path)

        if df_appearances is None:
            app_path = get_driver_circuit_path("driver_circuit_appearances.parquet", base_dir=self.base_dir)
            if not dataset_exists(app_path):
                df_appearances = self.build_driver_circuit_appearances(force=force)
            else:
                df_appearances = load_parquet(app_path)

        if df_appearances.empty:
            logger.warning("Empty appearances dataframe; returning empty aggregated profiles.")
            df_empty = pd.DataFrame([], columns=DRIVER_CIRCUIT_AGGREGATED_SCHEMA.names)
            save_parquet(df_empty, target_path, schema=DRIVER_CIRCUIT_AGGREGATED_SCHEMA)
            return df_empty

        logger.info(f"Aggregating driver-circuit metrics across {len(df_appearances)} appearances...")
        groups = df_appearances.groupby(["driver_id", "circuit_id"])
        records = []
        now = get_iso_timestamp()

        for (d_id, c_id), g in groups:
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

            # Teammate head-to-head
            valid_race_h2h = g[g["beat_teammate_race"].notna()]
            race_h2h_wins = int((valid_race_h2h["beat_teammate_race"] == True).sum())
            race_h2h_rate = float(race_h2h_wins / len(valid_race_h2h)) if len(valid_race_h2h) > 0 else None

            valid_qualy_h2h = g[g["beat_teammate_qualy"].notna()]
            qualy_h2h_wins = int((valid_qualy_h2h["beat_teammate_qualy"] == True).sum())
            qualy_h2h_rate = float(qualy_h2h_wins / len(valid_qualy_h2h)) if len(valid_qualy_h2h) > 0 else None

            # Current Ground Effect Era metrics (2022+)
            ge_g = g[g["season"] >= 2022]
            ge_starts = len(ge_g)
            ge_class = ge_g[ge_g["is_classified"] & (ge_g["finish_position"] > 0)]
            ge_avg_fin = float(ge_class["finish_position"].mean()) if not ge_class.empty else None

            records.append({
                "driver_id": d_id,
                "circuit_id": c_id,
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
                "source_api": "pace.driver_circuit_aggregator",
                "retrieved_at": now,
            })

        df_agg = pd.DataFrame(records)
        save_parquet(df_agg, target_path, schema=DRIVER_CIRCUIT_AGGREGATED_SCHEMA)
        logger.info(f"Saved {len(df_agg)} aggregated driver-circuit profiles to {target_path}")
        return df_agg

    def build_circuit_characteristics(
        self,
        df_appearances: Optional[pd.DataFrame] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Compile technical characteristics and metadata profiles for circuits."""
        target_path = get_driver_circuit_path("circuit_characteristics.parquet", base_dir=self.base_dir)
        if not force and dataset_exists(target_path):
            logger.info("Circuit characteristics dataset already exists. Skipping compilation.")
            return load_parquet(target_path)

        circuits_path = get_historical_path("circuits", base_dir=self.base_dir)
        events_path = get_historical_path("events", base_dir=self.base_dir)

        df_circuits = load_parquet(circuits_path) if dataset_exists(circuits_path) else pd.DataFrame()
        df_events = load_parquet(events_path) if dataset_exists(events_path) else pd.DataFrame()

        # Gather all circuit IDs from historical circuits, events, metrics, and appearances
        all_circuit_ids: Set[str] = set()
        if not df_circuits.empty and "circuit_id" in df_circuits.columns:
            all_circuit_ids.update(df_circuits["circuit_id"].dropna().unique())
        if df_appearances is not None and not df_appearances.empty and "circuit_id" in df_appearances.columns:
            all_circuit_ids.update(df_appearances["circuit_id"].dropna().unique())
        all_circuit_ids.update(CIRCUIT_METRICS.keys())

        # Map metadata from df_circuits
        circuit_meta = {}
        if not df_circuits.empty:
            for _, r in df_circuits.iterrows():
                circuit_meta[str(r["circuit_id"])] = {
                    "name": str(r.get("circuit_name", "")),
                    "country": str(r.get("country", "")),
                    "locality": str(r.get("locality", "")),
                    "lat": float(r.get("latitude", 0.0)),
                    "lng": float(r.get("longitude", 0.0)),
                }

        # Count total races held per circuit across recorded history
        race_counts: Dict[str, int] = {}
        if not df_events.empty and "circuit_id" in df_events.columns:
            race_counts = df_events["circuit_id"].value_counts().to_dict()

        records = []
        now = get_iso_timestamp()

        for c_id in sorted(list(all_circuit_ids)):
            meta = circuit_meta.get(c_id, {})
            c_name = meta.get("name", c_id.replace("_", " ").title())
            country = meta.get("country", "Unknown")
            locality = meta.get("locality", "Unknown")
            lat = meta.get("lat", 0.0)
            lng = meta.get("lng", 0.0)

            info = CIRCUIT_METRICS.get(c_id, {})
            length = float(info.get("length_km", 5.0))
            laps = int(info.get("laps", 55))
            overtake = str(info.get("overtake", "Medium"))
            tyre = str(info.get("tyre", "Medium"))
            downforce = str(info.get("downforce", "Medium"))
            alt = float(info.get("alt_m", 50.0))
            first_gp = int(info.get("first_gp", 2000))
            total_held = int(race_counts.get(c_id, 0))

            records.append({
                "circuit_id": c_id,
                "circuit_name": c_name,
                "country": country,
                "locality": locality,
                "latitude": lat,
                "longitude": lng,
                "first_grand_prix_year": first_gp,
                "total_races_held": total_held,
                "circuit_length_km": length,
                "typical_race_laps": laps,
                "overtaking_difficulty_rating": overtake,
                "tyre_stress_level": tyre,
                "downforce_level": downforce,
                "altitude_m": alt,
                "source_api": "pace.circuit_registry",
                "retrieved_at": now,
            })

        df_char = pd.DataFrame(records)
        save_parquet(df_char, target_path, schema=CIRCUIT_CHARACTERISTICS_SCHEMA)
        logger.info(f"Saved {len(df_char)} circuit characteristics profiles to {target_path}")
        return df_char
