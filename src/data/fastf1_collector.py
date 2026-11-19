"""Data collector module for PACE using FastF1 (Ergast API) and OpenF1.

Fetches race results, qualifying classifications, sprint results, lap times,
pit stops, practice sessions, drivers, constructors, circuits, and schedules.
"""

from datetime import datetime, timezone
import logging
import re
import time
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import requests

from fastf1.ergast import Ergast
from src.data.schemas import (
    DRIVERS_SCHEMA,
    CONSTRUCTORS_SCHEMA,
    CIRCUITS_SCHEMA,
    EVENTS_SCHEMA,
    SESSIONS_SCHEMA,
    RACE_RESULTS_SCHEMA,
    QUALIFYING_RESULTS_SCHEMA,
    SPRINT_RESULTS_SCHEMA,
    LAPS_SCHEMA,
    PIT_STOPS_SCHEMA,
    PRACTICE_LAPS_SCHEMA,
)
from src.data.storage import (
    save_raw_json,
    save_parquet,
    get_processed_path,
    get_raw_path,
    dataset_exists,
)

logger = logging.getLogger(__name__)


def get_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def extract_df(resp: Any) -> pd.DataFrame:
    """Extract DataFrame from Ergast response regardless of whether it is a DataFrame or multi-response."""
    if isinstance(resp, pd.DataFrame):
        return resp
    if hasattr(resp, "content") and resp.content:
        return resp.content[0]
    return pd.DataFrame()


def parse_time_to_ms(val: Any) -> Optional[float]:
    """Convert time representation (timedelta, str 'M:SS.sss' or 'SS.sss', or ms float) to milliseconds."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    if isinstance(val, pd.Timedelta):
        return float(val.total_seconds() * 1000.0)

    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "none", ""):
        return None

    # Handle '0 days 00:01:23.456000' format
    if "days" in val_str:
        try:
            td = pd.to_timedelta(val_str)
            return float(td.total_seconds() * 1000.0)
        except Exception:
            pass

    # Regex parse mm:ss.sss or ss.sss
    m = re.match(r"^(?:(\d+):)?(\d+)(?:\.(\d+))?$", val_str)
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        seconds = int(m.group(2))
        fraction = float("0." + m.group(3)) if m.group(3) else 0.0
        return float((minutes * 60 + seconds + fraction) * 1000.0)

    return None


def classify_status(status_str: str, position_text: str = "") -> Tuple[bool, bool, bool, bool]:
    """Determine classification flags (is_classified, is_dnf, is_dns, is_dsq).

    Returns:
        (is_classified, is_dnf, is_dns, is_dsq)
    """
    s = (status_str or "").strip().lower()
    p = (position_text or "").strip().upper()

    if p == "D" or "disqualif" in s:
        return False, False, False, True

    if p == "W" or "did not start" in s or "dns" in s or "withdrew" in s:
        return False, False, True, False

    if s.startswith("finished") or "+ lap" in s or "+ laps" in s or s.startswith("+"):
        return True, False, False, False

    # Any other status is a retirement / DNF
    return False, True, False, False


class FastF1Collector:
    """Orchestrates collection of F1 datasets via FastF1's Ergast interface and OpenF1."""

    def __init__(self, request_delay: float = 0.3):
        self.ergast = Ergast()
        self.request_delay = request_delay

    def _sleep(self):
        """Respect API polite delay to prevent rate limits."""
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

    def collect_drivers(self, season: int, force: bool = False) -> pd.DataFrame:
        """Collect driver biographies and metadata for a given season."""
        processed_path = get_processed_path("drivers", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Drivers dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        logger.info(f"Fetching drivers for season {season}...")
        resp = self._safe_call(self.ergast.get_driver_info, season=season)
        df_raw = extract_df(resp)

        records = []
        now = get_iso_timestamp()
        for _, row in df_raw.iterrows():
            d_id = str(row.get("driverId", ""))
            code = str(row.get("driverCode", "")) if pd.notna(row.get("driverCode")) else ""
            num = int(row.get("driverNumber", 0)) if pd.notna(row.get("driverNumber")) else -1
            given = str(row.get("givenName", ""))
            family = str(row.get("familyName", ""))
            dob = str(row.get("dateOfBirth", ""))
            nat = str(row.get("driverNationality", ""))
            url = str(row.get("driverUrl", ""))

            records.append({
                "driver_id": d_id,
                "driver_code": code,
                "driver_number": num,
                "given_name": given,
                "family_name": family,
                "full_name": f"{given} {family}".strip(),
                "date_of_birth": dob,
                "nationality": nat,
                "driver_url": url,
                "source_api": "fastf1.ergast",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=DRIVERS_SCHEMA.names)
        save_raw_json(records, "drivers", season, "drivers.json")
        save_parquet(df, processed_path, schema=DRIVERS_SCHEMA)
        return df

    def collect_constructors(self, season: int, force: bool = False) -> pd.DataFrame:
        """Collect constructor metadata for a given season."""
        processed_path = get_processed_path("constructors", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Constructors dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        logger.info(f"Fetching constructors for season {season}...")
        resp = self._safe_call(self.ergast.get_constructor_info, season=season)
        df_raw = extract_df(resp)

        records = []
        now = get_iso_timestamp()
        for _, row in df_raw.iterrows():
            c_id = str(row.get("constructorId", ""))
            name = str(row.get("constructorName", ""))
            nat = str(row.get("constructorNationality", ""))
            url = str(row.get("constructorUrl", ""))

            records.append({
                "constructor_id": c_id,
                "constructor_name": name,
                "nationality": nat,
                "constructor_url": url,
                "source_api": "fastf1.ergast",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=CONSTRUCTORS_SCHEMA.names)
        save_raw_json(records, "constructors", season, "constructors.json")
        save_parquet(df, processed_path, schema=CONSTRUCTORS_SCHEMA)
        return df

    def collect_circuits(self, season: int, force: bool = False) -> pd.DataFrame:
        """Collect circuit metadata for a given season."""
        processed_path = get_processed_path("circuits", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Circuits dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        logger.info(f"Fetching circuits for season {season}...")
        resp = self._safe_call(self.ergast.get_circuits, season=season)
        df_raw = extract_df(resp)

        records = []
        now = get_iso_timestamp()
        for _, row in df_raw.iterrows():
            c_id = str(row.get("circuitId", ""))
            name = str(row.get("circuitName", ""))
            loc = str(row.get("locality", ""))
            country = str(row.get("country", ""))
            lat = float(row.get("lat", 0.0)) if pd.notna(row.get("lat")) else 0.0
            lng = float(row.get("long", 0.0)) if pd.notna(row.get("long")) else 0.0
            url = str(row.get("circuitUrl", ""))

            records.append({
                "circuit_id": c_id,
                "circuit_name": name,
                "locality": loc,
                "country": country,
                "latitude": lat,
                "longitude": lng,
                "circuit_url": url,
                "source_api": "fastf1.ergast",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=CIRCUITS_SCHEMA.names)
        save_raw_json(records, "circuits", season, "circuits.json")
        save_parquet(df, processed_path, schema=CIRCUITS_SCHEMA)
        return df

    def collect_events_and_sessions(self, season: int, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect event schedule and individual session metadata for a season."""
        events_path = get_processed_path("events", season)
        sessions_path = get_processed_path("sessions", season)
        if not force and dataset_exists(events_path) and dataset_exists(sessions_path):
            logger.info(f"Events and sessions datasets for {season} already exist. Skipping download.")
            return pd.read_parquet(events_path), pd.read_parquet(sessions_path)

        logger.info(f"Fetching event schedule for season {season}...")
        resp = self._safe_call(self.ergast.get_race_schedule, season=season)
        sched = extract_df(resp)

        event_records = []
        session_records = []
        now = get_iso_timestamp()

        for _, row in sched.iterrows():
            rnd = int(row.get("round", 0))
            name = str(row.get("raceName", ""))
            c_id = str(row.get("circuitId", ""))
            country = str(row.get("country", ""))
            date = str(row.get("raceDate", ""))
            url = str(row.get("raceUrl", ""))

            event_records.append({
                "season": season,
                "round": rnd,
                "event_name": name,
                "circuit_id": c_id,
                "country": country,
                "event_date": date,
                "event_url": url,
                "source_api": "fastf1.ergast",
                "retrieved_at": now,
            })

            # Extract session schedule
            session_types = [
                ("FP1", "Practice 1", "fp1Date", "fp1Time"),
                ("FP2", "Practice 2", "fp2Date", "fp2Time"),
                ("FP3", "Practice 3", "fp3Date", "fp3Time"),
                ("Qualifying", "Qualifying", "qualifyingDate", "qualifyingTime"),
                ("Sprint", "Sprint", "sprintDate", "sprintTime"),
                ("Race", "Race", "raceDate", "raceTime"),
            ]
            for s_code, s_title, d_col, t_col in session_types:
                s_date = row.get(d_col)
                s_time = row.get(t_col)
                if pd.notna(s_date) and str(s_date).strip():
                    session_records.append({
                        "season": season,
                        "round": rnd,
                        "session_type": s_code,
                        "session_name": s_title,
                        "session_date": str(s_date),
                        "session_time_utc": str(s_time) if pd.notna(s_time) else "",
                        "source_api": "fastf1.ergast",
                        "retrieved_at": now,
                    })

        df_events = pd.DataFrame(event_records, columns=EVENTS_SCHEMA.names)
        df_sessions = pd.DataFrame(session_records, columns=SESSIONS_SCHEMA.names)

        save_raw_json(event_records, "events", season, "schedule.json")
        save_parquet(df_events, events_path, schema=EVENTS_SCHEMA)
        save_parquet(df_sessions, sessions_path, schema=SESSIONS_SCHEMA)
        return df_events, df_sessions

    def collect_race_results(self, season: int, max_rounds: Optional[int] = None, force: bool = False) -> pd.DataFrame:
        """Collect race results for all completed rounds in a season."""
        processed_path = get_processed_path("race_results", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Race results dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        events_df = self.collect_events_and_sessions(season, force=False)[0]
        rounds = sorted(events_df["round"].unique())
        if max_rounds:
            rounds = rounds[:max_rounds]

        records = []
        now = get_iso_timestamp()

        for rnd in rounds:
            logger.info(f"Fetching race results: {season} Round {rnd}...")
            raw_path = get_raw_path("race_results", season, f"round_{rnd}_race.json")
            df_round = pd.DataFrame()
            if not force and raw_path.exists():
                try:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        df_round = pd.DataFrame(json.load(f))
                except Exception:
                    df_round = pd.DataFrame()

            if df_round.empty:
                try:
                    resp = self._safe_call(self.ergast.get_race_results, season=season, round=rnd)
                    df_round = extract_df(resp)
                    if df_round.empty:
                        logger.info(f"Round {rnd} has no race results yet. Skipping.")
                        continue
                    save_raw_json(df_round.to_dict(orient="records"), "race_results", season, f"round_{rnd}_race.json")
                except Exception as e:
                    logger.warning(f"Error fetching race results for {season} Round {rnd}: {e}")
                    continue

            for _, row in df_round.iterrows():
                    pos_raw = row.get("position")
                    pos = int(pos_raw) if pd.notna(pos_raw) and str(pos_raw).isdigit() else -1
                    pos_text = str(row.get("positionText", ""))
                    status = str(row.get("status", ""))
                    is_class, is_dnf, is_dns, is_dsq = classify_status(status, pos_text)

                    tot_time_ms = float(row.get("totalRaceTimeMillis")) if pd.notna(row.get("totalRaceTimeMillis")) else None
                    tot_time_str = str(row.get("totalRaceTime", "")) if pd.notna(row.get("totalRaceTime")) else ""

                    fast_rank = int(row.get("fastestLapRank", 0)) if pd.notna(row.get("fastestLapRank")) and str(row.get("fastestLapRank")).isdigit() else 0
                    fast_time = str(row.get("fastestLapTime", "")) if pd.notna(row.get("fastestLapTime")) else ""
                    fast_speed = float(row.get("fastestLapAvgSpeed", 0.0)) if pd.notna(row.get("fastestLapAvgSpeed")) else 0.0

                    records.append({
                        "season": season,
                        "round": rnd,
                        "driver_id": str(row.get("driverId", "")),
                        "constructor_id": str(row.get("constructorId", "")),
                        "driver_number": int(row.get("driverNumber", 0)) if pd.notna(row.get("driverNumber")) else -1,
                        "driver_code": str(row.get("driverCode", "")) if pd.notna(row.get("driverCode")) else "",
                        "grid_position": int(row.get("grid", 0)) if pd.notna(row.get("grid")) else 0,
                        "finish_position": pos,
                        "position_text": pos_text,
                        "points": float(row.get("points", 0.0)) if pd.notna(row.get("points")) else 0.0,
                        "laps_completed": int(row.get("laps", 0)) if pd.notna(row.get("laps")) else 0,
                        "status": status,
                        "is_classified": is_class,
                        "is_dnf": is_dnf,
                        "is_dns": is_dns,
                        "is_dsq": is_dsq,
                        "total_time_ms": tot_time_ms,
                        "total_time_str": tot_time_str,
                        "fastest_lap_rank": fast_rank,
                        "fastest_lap_time": fast_time,
                        "fastest_lap_speed": fast_speed,
                        "source_api": "fastf1.ergast",
                        "retrieved_at": now,
                    })

        df = pd.DataFrame(records, columns=RACE_RESULTS_SCHEMA.names)
        save_parquet(df, processed_path, schema=RACE_RESULTS_SCHEMA)
        return df

    def collect_qualifying_results(self, season: int, max_rounds: Optional[int] = None, force: bool = False) -> pd.DataFrame:
        """Collect qualifying session classifications for a season."""
        processed_path = get_processed_path("qualifying", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Qualifying dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        events_df = self.collect_events_and_sessions(season, force=False)[0]
        rounds = sorted(events_df["round"].unique())
        if max_rounds:
            rounds = rounds[:max_rounds]

        records = []
        now = get_iso_timestamp()

        for rnd in rounds:
            logger.info(f"Fetching qualifying results: {season} Round {rnd}...")
            raw_path = get_raw_path("qualifying", season, f"round_{rnd}_qualifying.json")
            df_round = pd.DataFrame()
            if not force and raw_path.exists():
                try:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        df_round = pd.DataFrame(json.load(f))
                except Exception:
                    df_round = pd.DataFrame()

            if df_round.empty:
                try:
                    resp = self._safe_call(self.ergast.get_qualifying_results, season=season, round=rnd)
                    df_round = extract_df(resp)
                    if df_round.empty:
                        continue
                    save_raw_json(df_round.to_dict(orient="records"), "qualifying", season, f"round_{rnd}_qualifying.json")
                except Exception as e:
                    logger.warning(f"Error fetching qualifying results for {season} Round {rnd}: {e}")
                    continue

            for _, row in df_round.iterrows():
                    pos_raw = row.get("position")
                    pos = int(pos_raw) if pd.notna(pos_raw) and str(pos_raw).isdigit() else -1

                    q1_str = str(row.get("Q1", "")) if pd.notna(row.get("Q1")) else ""
                    q2_str = str(row.get("Q2", "")) if pd.notna(row.get("Q2")) else ""
                    q3_str = str(row.get("Q3", "")) if pd.notna(row.get("Q3")) else ""

                    records.append({
                        "season": season,
                        "round": rnd,
                        "driver_id": str(row.get("driverId", "")),
                        "constructor_id": str(row.get("constructorId", "")),
                        "driver_number": int(row.get("driverNumber", 0)) if pd.notna(row.get("driverNumber")) else -1,
                        "driver_code": str(row.get("driverCode", "")) if pd.notna(row.get("driverCode")) else "",
                        "qualifying_position": pos,
                        "q1_time_str": q1_str,
                        "q1_time_ms": parse_time_to_ms(q1_str),
                        "q2_time_str": q2_str,
                        "q2_time_ms": parse_time_to_ms(q2_str),
                        "q3_time_str": q3_str,
                        "q3_time_ms": parse_time_to_ms(q3_str),
                        "source_api": "fastf1.ergast",
                        "retrieved_at": now,
                    })

        df = pd.DataFrame(records, columns=QUALIFYING_RESULTS_SCHEMA.names)
        save_parquet(df, processed_path, schema=QUALIFYING_RESULTS_SCHEMA)
        return df

    def collect_sprint_results(self, season: int, max_rounds: Optional[int] = None, force: bool = False) -> pd.DataFrame:
        """Collect sprint race classifications for sprint weekend events in a season."""
        processed_path = get_processed_path("sprints", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Sprint results dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        events_df = self.collect_events_and_sessions(season, force=False)[0]
        rounds = sorted(events_df["round"].unique())
        if max_rounds:
            rounds = rounds[:max_rounds]

        records = []
        now = get_iso_timestamp()

        for rnd in rounds:
            raw_path = get_raw_path("sprints", season, f"round_{rnd}_sprint.json")
            df_round = pd.DataFrame()
            if not force and raw_path.exists():
                try:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        df_round = pd.DataFrame(json.load(f))
                except Exception:
                    df_round = pd.DataFrame()

            if df_round.empty:
                try:
                    resp = self._safe_call(self.ergast.get_sprint_results, season=season, round=rnd)
                    df_round = extract_df(resp)
                    if df_round.empty:
                        continue
                    save_raw_json(df_round.to_dict(orient="records"), "sprints", season, f"round_{rnd}_sprint.json")
                except Exception as e:
                    logger.warning(f"Error checking sprint results for {season} Round {rnd}: {e}")
                    continue

            for _, row in df_round.iterrows():
                    pos_raw = row.get("position")
                    pos = int(pos_raw) if pd.notna(pos_raw) and str(pos_raw).isdigit() else -1
                    pos_text = str(row.get("positionText", ""))
                    status = str(row.get("status", ""))
                    is_class, is_dnf, is_dns, is_dsq = classify_status(status, pos_text)

                    tot_time_ms = float(row.get("totalRaceTimeMillis")) if pd.notna(row.get("totalRaceTimeMillis")) else None
                    tot_time_str = str(row.get("totalRaceTime", "")) if pd.notna(row.get("totalRaceTime")) else ""
                    fast_lap_num = int(row.get("fastestLapNumber", 0)) if pd.notna(row.get("fastestLapNumber")) else 0
                    fast_lap_time = str(row.get("fastestLapTime", "")) if pd.notna(row.get("fastestLapTime")) else ""

                    records.append({
                        "season": season,
                        "round": rnd,
                        "driver_id": str(row.get("driverId", "")),
                        "constructor_id": str(row.get("constructorId", "")),
                        "driver_number": int(row.get("driverNumber", 0)) if pd.notna(row.get("driverNumber")) else -1,
                        "driver_code": str(row.get("driverCode", "")) if pd.notna(row.get("driverCode")) else "",
                        "grid_position": int(row.get("grid", 0)) if pd.notna(row.get("grid")) else 0,
                        "finish_position": pos,
                        "position_text": pos_text,
                        "points": float(row.get("points", 0.0)) if pd.notna(row.get("points")) else 0.0,
                        "laps_completed": int(row.get("laps", 0)) if pd.notna(row.get("laps")) else 0,
                        "status": status,
                        "is_classified": is_class,
                        "is_dnf": is_dnf,
                        "is_dns": is_dns,
                        "is_dsq": is_dsq,
                        "total_time_ms": tot_time_ms,
                        "total_time_str": tot_time_str,
                        "fastest_lap_number": fast_lap_num,
                        "fastest_lap_time": fast_lap_time,
                        "source_api": "fastf1.ergast",
                        "retrieved_at": now,
                    })

        df = pd.DataFrame(records, columns=SPRINT_RESULTS_SCHEMA.names)
        save_parquet(df, processed_path, schema=SPRINT_RESULTS_SCHEMA)
        return df

    def collect_laps_and_pits(
        self,
        season: int,
        rounds: Optional[List[int]] = None,
        max_laps_per_round: int = 500,
        force: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect race lap times and pit stops for designated rounds in a season."""
        laps_path = get_processed_path("laps", season)
        pits_path = get_processed_path("pit_stops", season)
        if not force and dataset_exists(laps_path) and dataset_exists(pits_path):
            logger.info(f"Laps and pit stops datasets for {season} already exist. Skipping download.")
            return pd.read_parquet(laps_path), pd.read_parquet(pits_path)

        if rounds is None:
            events_df = self.collect_events_and_sessions(season, force=False)[0]
            rounds = sorted(events_df["round"].unique())

        lap_records = []
        pit_records = []
        now = get_iso_timestamp()

        for rnd in rounds:
            logger.info(f"Fetching lap times & pit stops: {season} Round {rnd}...")
            self._sleep()
            # 1. Lap times
            try:
                lap_resp = self._safe_call(self.ergast.get_lap_times, season=season, round=rnd, limit=max_laps_per_round)
                df_laps = extract_df(lap_resp)
                if not df_laps.empty:
                    save_raw_json(df_laps.to_dict(orient="records"), "laps", season, f"round_{rnd}_laps.json")
                    for _, row in df_laps.iterrows():
                        time_val = row.get("time")
                        lap_records.append({
                            "season": season,
                            "round": rnd,
                            "lap_number": int(row.get("number", 0)) if pd.notna(row.get("number")) else 0,
                            "driver_id": str(row.get("driverId", "")),
                            "position": int(row.get("position", 0)) if pd.notna(row.get("position")) else 0,
                            "lap_time_str": str(time_val) if pd.notna(time_val) else "",
                            "lap_time_ms": parse_time_to_ms(time_val),
                            "source_api": "fastf1.ergast",
                            "retrieved_at": now,
                        })
            except Exception as e:
                logger.warning(f"Error fetching lap times for {season} Round {rnd}: {e}")

            # 2. Pit stops
            try:
                pit_resp = self._safe_call(self.ergast.get_pit_stops, season=season, round=rnd)
                df_pits = extract_df(pit_resp)
                if not df_pits.empty:
                    save_raw_json(df_pits.to_dict(orient="records"), "pit_stops", season, f"round_{rnd}_pits.json")
                    for _, row in df_pits.iterrows():
                        dur_val = row.get("duration")
                        pit_records.append({
                            "season": season,
                            "round": rnd,
                            "driver_id": str(row.get("driverId", "")),
                            "stop_number": int(row.get("stop", 0)) if pd.notna(row.get("stop")) else 0,
                            "lap_number": int(row.get("lap", 0)) if pd.notna(row.get("lap")) else 0,
                            "pit_time_str": str(row.get("time", "")) if pd.notna(row.get("time")) else "",
                            "duration_str": str(dur_val) if pd.notna(dur_val) else "",
                            "duration_ms": parse_time_to_ms(dur_val),
                            "source_api": "fastf1.ergast",
                            "retrieved_at": now,
                        })
            except Exception as e:
                logger.warning(f"Error fetching pit stops for {season} Round {rnd}: {e}")

        df_laps_res = pd.DataFrame(lap_records, columns=LAPS_SCHEMA.names)
        df_pits_res = pd.DataFrame(pit_records, columns=PIT_STOPS_SCHEMA.names)

        save_parquet(df_laps_res, laps_path, schema=LAPS_SCHEMA)
        save_parquet(df_pits_res, pits_path, schema=PIT_STOPS_SCHEMA)
        return df_laps_res, df_pits_res

    def collect_practice_laps(
        self,
        season: int,
        session_keys: Optional[List[int]] = None,
        max_sessions: int = 5,
        force: bool = False,
    ) -> pd.DataFrame:
        """Collect practice session laps via OpenF1 API without requiring private F1 credentials."""
        processed_path = get_processed_path("practice_laps", season)
        if not force and dataset_exists(processed_path):
            logger.info(f"Practice laps dataset for {season} already exists. Skipping download.")
            return pd.read_parquet(processed_path)

        records = []
        now = get_iso_timestamp()

        try:
            self._sleep()
            sess_url = f"https://api.openf1.org/v1/sessions?year={season}"
            resp = requests.get(sess_url, timeout=15)
            if resp.status_code == 200:
                all_sessions = resp.json()
                practice_sessions = [
                    s for s in all_sessions
                    if "Practice" in s.get("session_name", "")
                ]
                if session_keys:
                    practice_sessions = [s for s in practice_sessions if s.get("session_key") in session_keys]
                elif max_sessions:
                    practice_sessions = practice_sessions[:max_sessions]

                for ps in practice_sessions:
                    s_key = ps.get("session_key")
                    s_name = ps.get("session_name")
                    s_type = "FP1" if "1" in s_name else ("FP2" if "2" in s_name else "FP3")
                    logger.info(f"Fetching OpenF1 practice laps: {season} {ps.get('country_name')} - {s_name} (key {s_key})...")
                    self._sleep()

                    laps_url = f"https://api.openf1.org/v1/laps?session_key={s_key}"
                    laps_resp = requests.get(laps_url, timeout=20)
                    if laps_resp.status_code == 200:
                        laps_data = laps_resp.json()
                        save_raw_json(laps_data, "practice_laps", season, f"session_{s_key}_practice.json")
                        for lap in laps_data:
                            lap_dur = lap.get("lap_duration")
                            records.append({
                                "season": season,
                                "round": 1,
                                "session_type": s_type,
                                "driver_number": int(lap.get("driver_number", -1)),
                                "lap_number": int(lap.get("lap_number", 0)),
                                "lap_time_seconds": float(lap_dur) if lap_dur is not None else None,
                                "sector_1_seconds": float(lap.get("duration_sector_1")) if lap.get("duration_sector_1") is not None else None,
                                "sector_2_seconds": float(lap.get("duration_sector_2")) if lap.get("duration_sector_2") is not None else None,
                                "sector_3_seconds": float(lap.get("duration_sector_3")) if lap.get("duration_sector_3") is not None else None,
                                "is_pit_out_lap": bool(lap.get("is_pit_out_lap", False)),
                                "i1_speed": float(lap.get("i1_speed")) if lap.get("i1_speed") is not None else None,
                                "i2_speed": float(lap.get("i2_speed")) if lap.get("i2_speed") is not None else None,
                                "st_speed": float(lap.get("st_speed")) if lap.get("st_speed") is not None else None,
                                "source_api": "openf1",
                                "retrieved_at": now,
                            })
        except Exception as e:
            logger.warning(f"Error fetching practice laps for season {season}: {e}")

        df = pd.DataFrame(records, columns=PRACTICE_LAPS_SCHEMA.names)
        save_parquet(df, processed_path, schema=PRACTICE_LAPS_SCHEMA)
        return df
