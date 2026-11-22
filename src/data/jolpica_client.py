"""PACE: Jolpica F1 API Client & Ingestion Engine.

Provides a production-grade, Ergast-compatible HTTP client for Jolpica F1 API
(https://api.jolpi.ca/ergast/f1/) with pagination, retry backoff, HTTP 429 rate limit
handling, local JSON disk caching, canonical entity resolution, and strongly-typed
Parquet persistence conforming to PACE PyArrow schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import requests

from src.data.canonical_registry import (
    CanonicalEntityRegistry,
    get_canonical_registry,
    resolve_driver,
    resolve_constructor,
    resolve_circuit,
)
from src.data.schemas import (
    DRIVER_STANDINGS_SCHEMA,
    CONSTRUCTOR_STANDINGS_SCHEMA,
    RACE_RESULTS_SCHEMA,
    QUALIFYING_RESULTS_SCHEMA,
    SPRINT_RESULTS_SCHEMA,
    EVENTS_SCHEMA,
)
from src.data.storage import (
    DEFAULT_BASE_DIR,
    get_historical_path,
    get_jolpica_dir,
    get_jolpica_path,
    get_jolpica_cache_dir,
    save_parquet,
    load_parquet,
    dataset_exists,
)

logger = logging.getLogger(__name__)

DEFAULT_JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_USER_AGENT = "PACE-F1-Analytics/1.0 (https://github.com/dash-pdf/PACE)"


# ---------------------------------------------------------------------------
# Exceptions & Configuration
# ---------------------------------------------------------------------------

class JolpicaAPIError(Exception):
    """Base exception for Jolpica API interaction failures."""
    pass


class JolpicaRateLimitError(JolpicaAPIError):
    """Raised when Jolpica returns HTTP 429 and retries are exhausted."""
    pass


class JolpicaConnectionError(JolpicaAPIError):
    """Raised when transport connection fails after all retries."""
    pass


@dataclass
class JolpicaConfig:
    """Configuration container for Jolpica API client."""
    base_url: str = DEFAULT_JOLPICA_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 12.0
    max_retries: int = 4
    backoff_base: float = 1.0
    max_backoff: float = 30.0
    rate_limit_pause: float = 0.25
    cache_dir: Optional[Path] = None
    use_cache: bool = True


@dataclass
class JolpicaStats:
    """Operational telemetry counters for Jolpica client requests."""
    requests_made: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries_attempted: int = 0
    rate_limits_encountered: int = 0
    errors_encountered: int = 0
    records_ingested: int = 0

    def summary(self) -> str:
        return (
            f"JolpicaStats(requests={self.requests_made}, cache_hits={self.cache_hits}, "
            f"cache_misses={self.cache_misses}, retries={self.retries_attempted}, "
            f"rate_limits={self.rate_limits_encountered}, errors={self.errors_encountered})"
        )


@dataclass
class JolpicaIngestResult:
    """Summary of an ingestion run across one or more seasons."""
    success: bool
    seasons: List[int]
    driver_standings_count: int = 0
    constructor_standings_count: int = 0
    race_results_count: int = 0
    qualifying_count: int = 0
    sprint_count: int = 0
    calendar_count: int = 0
    unresolved_entities: Dict[str, Set[str]] = field(default_factory=lambda: {"drivers": set(), "constructors": set(), "circuits": set()})
    datasets_written: Dict[str, Path] = field(default_factory=dict)
    summary_message: str = ""


# ---------------------------------------------------------------------------
# Jolpica API HTTP Client
# ---------------------------------------------------------------------------

class JolpicaClient:
    """Production-grade Ergast-compatible REST client for the Jolpica F1 API.
    
    Features:
    - Automated pagination over large result sets
    - Configurable retry mechanism with exponential backoff and jitter
    - HTTP 429 rate limit adherence and Retry-After header parsing
    - Robust local JSON disk caching preventing redundant remote traffic
    - Offline-compatible design for unit and integration testing
    """

    def __init__(
        self,
        config: Optional[JolpicaConfig] = None,
        session: Optional[requests.Session] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or JolpicaConfig()
        self.session = session or requests.Session()
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.stats = JolpicaStats()

        # Cache directory setup
        if self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir)
        else:
            self.cache_dir = get_jolpica_cache_dir(base_dir=self.base_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, endpoint_url: str, params: Optional[Dict[str, Any]] = None) -> Path:
        """Derive reproducible, filesystem-safe cache path for an endpoint request."""
        clean_url = endpoint_url.replace(self.config.base_url, "").strip("/").replace("/", "_")
        if not clean_url:
            clean_url = "root"

        param_str = ""
        if params:
            sorted_params = sorted([(k, str(v)) for k, v in params.items()])
            param_repr = "&".join(f"{k}={v}" for k, v in sorted_params)
            param_hash = hashlib.sha256(param_repr.encode("utf-8")).hexdigest()[:12]
            param_str = f"_{param_hash}"

        filename = f"{clean_url}{param_str}.json"
        # Sanitize filename
        safe_filename = re.sub(r"[^\w\-.]", "_", filename)
        return self.cache_dir / safe_filename

    def request_endpoint(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Execute a GET request against Jolpica with caching, retries, and rate limit backoff."""
        # Ensure endpoint starts with base_url
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            url = endpoint
        else:
            clean_ep = endpoint.lstrip("/")
            url = f"{self.config.base_url.rstrip('/')}/{clean_ep}"

        # Ensure .json extension is present for Ergast compatibility
        if not url.endswith(".json") and "?" not in url:
            url = f"{url}.json"
        elif not url.endswith(".json") and "?" in url:
            base_part, query_part = url.split("?", 1)
            if not base_part.endswith(".json"):
                url = f"{base_part}.json?{query_part}"

        cache_path = self._get_cache_path(url, params)

        # 1. Check local cache
        if self.config.use_cache and not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.stats.cache_hits += 1
                logger.debug(f"Jolpica cache hit: {cache_path.name}")
                return data
            except Exception as e:
                logger.warning(f"Corrupt cache file {cache_path}, refetching: {e}")

        # 2. Remote HTTP Dispatch with Retries
        self.stats.cache_misses += 1
        headers = {"User-Agent": self.config.user_agent, "Accept": "application/json"}
        attempt = 0

        while attempt <= self.config.max_retries:
            attempt += 1
            self.stats.requests_made += 1

            try:
                # Polite rate-limit delay between active network dispatches
                if self.config.rate_limit_pause > 0:
                    time.sleep(self.config.rate_limit_pause)

                response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)

                # HTTP 429 Rate Limit Handling
                if response.status_code == 429:
                    self.stats.rate_limits_encountered += 1
                    if attempt > self.config.max_retries:
                        raise JolpicaRateLimitError(f"Exceeded max retries on HTTP 429 from Jolpica API: {url}")
                    self.stats.retries_attempted += 1
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait_sec = float(retry_after)
                    else:
                        wait_sec = min(self.config.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), self.config.max_backoff)

                    logger.warning(f"Jolpica HTTP 429 Rate Limit hit. Backing off for {wait_sec:.2f}s (attempt {attempt}/{self.config.max_retries})")
                    time.sleep(wait_sec)
                    continue

                # 5xx Server Error Handling
                if 500 <= response.status_code < 600:
                    if attempt > self.config.max_retries:
                        raise JolpicaAPIError(f"HTTP {response.status_code} from Jolpica after {self.config.max_retries} retries: {url}")
                    self.stats.retries_attempted += 1
                    wait_sec = min(self.config.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), self.config.max_backoff)
                    logger.warning(f"Jolpica HTTP {response.status_code} server error. Retrying in {wait_sec:.2f}s (attempt {attempt}/{self.config.max_retries})")
                    time.sleep(wait_sec)
                    continue

                response.raise_for_status()

                # Parse JSON
                try:
                    payload = response.json()
                except ValueError as e:
                    raise JolpicaAPIError(f"Malformed JSON response from Jolpica at {url}: {e}")

                # Save to cache
                if self.config.use_cache:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed writing cache file {cache_path}: {e}")

                return payload

            except (requests.ConnectionError, requests.Timeout) as e:
                self.stats.errors_encountered += 1
                if attempt > self.config.max_retries:
                    raise JolpicaConnectionError(f"Jolpica API transport unreachable after {self.config.max_retries} attempts: {e}")
                self.stats.retries_attempted += 1
                wait_sec = min(self.config.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), self.config.max_backoff)
                logger.warning(f"Network transport error querying Jolpica ({e.__class__.__name__}): {e}. Retrying in {wait_sec:.2f}s (attempt {attempt}/{self.config.max_retries})")
                time.sleep(wait_sec)

        raise JolpicaAPIError(f"Failed querying Jolpica endpoint {url} after {self.config.max_retries} attempts.")

    def paginate_endpoint(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        extract_key: Optional[Tuple[str, ...]] = None,
        page_size: int = 100,
        max_pages: int = 50,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Automatically retrieve and aggregate all pages from a paginated Ergast endpoint."""
        query_params = dict(params or {})
        query_params["limit"] = page_size
        offset = int(query_params.get("offset", 0))

        all_items: List[Dict[str, Any]] = []
        page_count = 0

        while page_count < max_pages:
            page_count += 1
            query_params["offset"] = offset

            payload = self.request_endpoint(endpoint, params=query_params, force_refresh=force_refresh)
            mr = payload.get("MRData", {})
            total = int(mr.get("total", 0))

            # Extract data items according to hierarchy
            items: List[Dict[str, Any]] = []
            if extract_key:
                cur = mr
                for k in extract_key:
                    if isinstance(cur, dict):
                        cur = cur.get(k, {})
                    elif isinstance(cur, list) and cur:
                        cur = cur[0].get(k, {}) if isinstance(cur[0], dict) else []
                    else:
                        cur = []
                if isinstance(cur, list):
                    items = cur
            else:
                # Automatic fallback: search top dictionary values
                for v in mr.values():
                    if isinstance(v, dict):
                        for sub_v in v.values():
                            if isinstance(sub_v, list):
                                items = sub_v
                                break

            if not items:
                break

            all_items.extend(items)

            # Check termination
            if offset + page_size >= total or len(items) == 0:
                break

            offset += page_size

        return all_items

    # -----------------------------------------------------------------------
    # Endpoint-Specific High-Level Queries
    # -----------------------------------------------------------------------

    def fetch_driver_standings(self, season: int, round_num: Optional[int] = None, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch driver championship standings for a given season or round."""
        if round_num is not None:
            ep = f"{season}/{round_num}/driverStandings"
        else:
            ep = f"{season}/driverStandings"

        return self.paginate_endpoint(
            endpoint=ep,
            extract_key=("StandingsTable", "StandingsLists", "DriverStandings"),
            page_size=100,
            force_refresh=force,
        )

    def fetch_constructor_standings(self, season: int, round_num: Optional[int] = None, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch constructor championship standings for a given season or round."""
        if round_num is not None:
            ep = f"{season}/{round_num}/constructorStandings"
        else:
            ep = f"{season}/constructorStandings"

        return self.paginate_endpoint(
            endpoint=ep,
            extract_key=("StandingsTable", "StandingsLists", "ConstructorStandings"),
            page_size=100,
            force_refresh=force,
        )

    def fetch_race_results(self, season: int, round_num: Optional[int] = None, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch Grand Prix race results for a season or specific round."""
        if round_num is not None:
            ep = f"{season}/{round_num}/results"
        else:
            ep = f"{season}/results"

        payload = self.request_endpoint(ep, force_refresh=force)
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races

    def fetch_qualifying_results(self, season: int, round_num: Optional[int] = None, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch qualifying session results for a season or round."""
        if round_num is not None:
            ep = f"{season}/{round_num}/qualifying"
        else:
            ep = f"{season}/qualifying"

        payload = self.request_endpoint(ep, force_refresh=force)
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races

    def fetch_sprint_results(self, season: int, round_num: Optional[int] = None, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch sprint race results for a season or round."""
        if round_num is not None:
            ep = f"{season}/{round_num}/sprint"
        else:
            ep = f"{season}/sprint"

        try:
            payload = self.request_endpoint(ep, force_refresh=force)
            races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            return races
        except Exception as e:
            logger.debug(f"No sprint results for {season} round {round_num}: {e}")
            return []

    def fetch_schedule(self, season: int, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch Grand Prix calendar and event schedules for a season."""
        ep = f"{season}"
        payload = self.request_endpoint(ep, force_refresh=force)
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races

    def check_connectivity(self) -> Tuple[bool, str]:
        """Test active network connectivity to the Jolpica API."""
        try:
            # Query recent season info with small limit
            data = self.request_endpoint("2024.json", params={"limit": 1}, force_refresh=True)
            total = data.get("MRData", {}).get("total", "unknown")
            return True, f"Successfully connected to Jolpica API at {self.config.base_url} (2024 total races: {total})"
        except Exception as e:
            return False, f"Failed connecting to Jolpica API: {e}"


# ---------------------------------------------------------------------------
# Jolpica Ingestion Engine & Schema Normalizer
# ---------------------------------------------------------------------------

class JolpicaIngestionEngine:
    """Transforms raw Jolpica API payloads into strongly-typed PACE Parquet datasets.
    
    Adheres strictly to:
    - DRIVER_STANDINGS_SCHEMA
    - CONSTRUCTOR_STANDINGS_SCHEMA
    - RACE_RESULTS_SCHEMA
    - QUALIFYING_RESULTS_SCHEMA
    - SPRINT_RESULTS_SCHEMA
    - EVENTS_SCHEMA
    """

    def __init__(
        self,
        client: Optional[JolpicaClient] = None,
        registry: Optional[CanonicalEntityRegistry] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.client = client or JolpicaClient(base_dir=self.base_dir)
        self.registry = registry or get_canonical_registry()
        self.unresolved: Dict[str, Set[str]] = {
            "drivers": set(),
            "constructors": set(),
            "circuits": set(),
        }

    # -----------------------------------------------------------------------
    # Canonical Resolution Helpers
    # -----------------------------------------------------------------------

    def resolve_driver_id(self, raw_driver_id: str, full_name: str = "") -> str:
        """Resolve Jolpica/Ergast driverId to PACE canonical driver identifier."""
        d_clean = raw_driver_id.strip()
        # Direct ergast match in registry
        for d in self.registry.list_drivers():
            if d.canonical_id == d_clean or d.ergast_id == d_clean:
                return d.canonical_id
            if full_name and d.full_name.strip().lower() == full_name.strip().lower():
                return d.canonical_id

        # Ambiguity-safe resolution via registry
        res = self.registry.resolve_driver(d_clean, strict=False)
        if res.is_resolved and res.canonical_id:
            return str(res.canonical_id)

        # Track unresolved entity
        self.unresolved["drivers"].add(d_clean)
        return d_clean

    def resolve_constructor_id(self, raw_constructor_id: str, name: str = "") -> str:
        """Resolve Jolpica/Ergast constructorId to PACE canonical constructor identifier."""
        c_clean = raw_constructor_id.strip()
        for c in self.registry.list_constructors():
            if c.canonical_id == c_clean or c.ergast_id == c_clean:
                return c.canonical_id
            if name and c.name.strip().lower() == name.strip().lower():
                return c.canonical_id

        res = self.registry.resolve_constructor(c_clean, strict=False)
        if res.is_resolved and res.canonical_id:
            return str(res.canonical_id)

        self.unresolved["constructors"].add(c_clean)
        return c_clean

    def resolve_circuit_id(self, raw_circuit_id: str, name: str = "") -> str:
        """Resolve Jolpica/Ergast circuitId to PACE canonical circuit identifier."""
        circ_clean = raw_circuit_id.strip()
        for circ in self.registry.list_circuits():
            if circ.canonical_id == circ_clean or circ.ergast_id == circ_clean:
                return circ.canonical_id
            if name and circ.name.strip().lower() == name.strip().lower():
                return circ.canonical_id

        res = self.registry.resolve_circuit(circ_clean, strict=False)
        if res.is_resolved and res.canonical_id:
            return str(res.canonical_id)

        self.unresolved["circuits"].add(circ_clean)
        return circ_clean

    # -----------------------------------------------------------------------
    # Ingestion Methods
    # -----------------------------------------------------------------------

    def ingest_driver_standings_for_round(self, season: int, round_num: int, force: bool = False) -> pd.DataFrame:
        """Ingest driver standings for a specific completed round."""
        raw_items = self.client.fetch_driver_standings(season, round_num=round_num, force=force)
        if not raw_items:
            return pd.DataFrame(columns=DRIVER_STANDINGS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for item in raw_items:
            pos_val = item.get("position")
            pos = int(pos_val) if pos_val is not None and str(pos_val).isdigit() else 999
            pos_text = str(item.get("positionText", str(pos)))
            pts = float(item.get("points", 0.0))
            wins = int(item.get("wins", 0))

            driver_info = item.get("Driver", {})
            raw_d_id = str(driver_info.get("driverId", "")).strip()
            fn = str(driver_info.get("givenName", "")).strip()
            sn = str(driver_info.get("familyName", "")).strip()
            full_name = f"{fn} {sn}".strip()
            canon_d_id = self.resolve_driver_id(raw_d_id, full_name=full_name)

            records.append({
                "season": int(season),
                "round": int(round_num),
                "driver_id": canon_d_id,
                "points": pts,
                "position": pos,
                "position_text": pos_text,
                "wins": wins,
                "source_api": "jolpica.ergast",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=DRIVER_STANDINGS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def ingest_constructor_standings_for_round(self, season: int, round_num: int, force: bool = False) -> pd.DataFrame:
        """Ingest constructor standings for a specific completed round."""
        raw_items = self.client.fetch_constructor_standings(season, round_num=round_num, force=force)
        if not raw_items:
            return pd.DataFrame(columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for item in raw_items:
            pos_val = item.get("position")
            pos = int(pos_val) if pos_val is not None and str(pos_val).isdigit() else 999
            pos_text = str(item.get("positionText", str(pos)))
            pts = float(item.get("points", 0.0))
            wins = int(item.get("wins", 0))

            const_info = item.get("Constructor", {})
            raw_c_id = str(const_info.get("constructorId", "")).strip()
            c_name = str(const_info.get("name", "")).strip()
            canon_c_id = self.resolve_constructor_id(raw_c_id, name=c_name)

            records.append({
                "season": int(season),
                "round": int(round_num),
                "constructor_id": canon_c_id,
                "points": pts,
                "position": pos,
                "position_text": pos_text,
                "wins": wins,
                "source_api": "jolpica.ergast",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "constructor_id"], keep="first")

    def ingest_full_season_standings(self, season: int, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest full progression of driver and constructor standings across all completed rounds."""
        # 1. Fetch calendar to determine total completed rounds
        schedule = self.client.fetch_schedule(season, force=force)
        if not schedule:
            logger.warning(f"No calendar schedule found from Jolpica for season {season}")
            return pd.DataFrame(columns=DRIVER_STANDINGS_SCHEMA.names), pd.DataFrame(columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)

        total_rounds = len(schedule)
        d_frames: List[pd.DataFrame] = []
        c_frames: List[pd.DataFrame] = []

        for rnd in range(1, total_rounds + 1):
            try:
                df_d = self.ingest_driver_standings_for_round(season, rnd, force=force)
                if not df_d.empty:
                    d_frames.append(df_d)
            except Exception as e:
                logger.debug(f"Season {season} round {rnd} driver standings not available: {e}")

            try:
                df_c = self.ingest_constructor_standings_for_round(season, rnd, force=force)
                if not df_c.empty:
                    c_frames.append(df_c)
            except Exception as e:
                logger.debug(f"Season {season} round {rnd} constructor standings not available: {e}")

        df_drivers = pd.concat(d_frames, ignore_index=True) if d_frames else pd.DataFrame(columns=DRIVER_STANDINGS_SCHEMA.names)
        df_constructors = pd.concat(c_frames, ignore_index=True) if c_frames else pd.DataFrame(columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)

        return df_drivers, df_constructors

    def ingest_season_race_results(self, season: int, force: bool = False) -> pd.DataFrame:
        """Ingest all Grand Prix race results for a season into RACE_RESULTS_SCHEMA."""
        races = self.client.fetch_race_results(season, force=force)
        if not races:
            return pd.DataFrame(columns=RACE_RESULTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for race in races:
            rnd = int(race.get("round", 1))

            for res in race.get("Results", []):
                driver_info = res.get("Driver", {})
                raw_d_id = str(driver_info.get("driverId", "")).strip()
                full_name = f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip()
                canon_d_id = self.resolve_driver_id(raw_d_id, full_name=full_name)

                const_info = res.get("Constructor", {})
                raw_c_id = str(const_info.get("constructorId", "")).strip()
                canon_c_id = self.resolve_constructor_id(raw_c_id, name=str(const_info.get("name", "")))

                d_num_val = res.get("number")
                d_num = int(d_num_val) if d_num_val and str(d_num_val).isdigit() else -1

                grid_val = res.get("grid")
                grid = int(grid_val) if grid_val and str(grid_val).isdigit() else 0

                pos_val = res.get("position")
                fin_pos = int(pos_val) if pos_val and str(pos_val).isdigit() else None
                pos_text = str(res.get("positionText", ""))
                pts = float(res.get("points", 0.0))
                laps = int(res.get("laps", 0)) if str(res.get("laps", "")).isdigit() else 0
                status = str(res.get("status", "Finished"))

                s_lower = status.lower()
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

                # Total race time
                time_info = res.get("Time", {})
                time_ms_val = time_info.get("millis")
                tot_time_ms = int(time_ms_val) if time_ms_val and str(time_ms_val).isdigit() else None
                time_str = str(time_info.get("time", ""))

                # Fastest lap
                fl_info = res.get("FastestLap", {})
                fl_rank_val = fl_info.get("rank")
                fl_rank = int(fl_rank_val) if fl_rank_val and str(fl_rank_val).isdigit() else None
                fl_num_val = fl_info.get("lap")
                fl_num = int(fl_num_val) if fl_num_val and str(fl_num_val).isdigit() else None
                fl_time = str(fl_info.get("Time", {}).get("time", ""))
                fl_spd_val = fl_info.get("AverageSpeed", {}).get("speed")
                fl_spd = float(fl_spd_val) if fl_spd_val is not None else None

                records.append({
                    "season": int(season),
                    "round": rnd,
                    "driver_id": canon_d_id,
                    "constructor_id": canon_c_id,
                    "driver_number": d_num,
                    "grid_position": grid,
                    "finish_position": fin_pos,
                    "position_text": pos_text,
                    "points": pts,
                    "laps_completed": laps,
                    "status": status,
                    "is_classified": is_classified,
                    "is_dnf": is_dnf,
                    "is_dns": is_dns,
                    "is_dsq": is_dsq,
                    "total_time_ms": tot_time_ms,
                    "time_text": time_str,
                    "fastest_lap_rank": fl_rank,
                    "fastest_lap_num": fl_num,
                    "fastest_lap_time": fl_time,
                    "fastest_lap_speed": fl_spd,
                    "source_api": "jolpica.ergast",
                    "retrieved_at": now,
                })

        df = pd.DataFrame(records, columns=RACE_RESULTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def ingest_season_qualifying(self, season: int, force: bool = False) -> pd.DataFrame:
        """Ingest qualifying session results for a season into QUALIFYING_RESULTS_SCHEMA."""
        races = self.client.fetch_qualifying_results(season, force=force)
        if not races:
            return pd.DataFrame(columns=QUALIFYING_RESULTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for race in races:
            rnd = int(race.get("round", 1))

            for q_res in race.get("QualifyingResults", []):
                driver_info = q_res.get("Driver", {})
                raw_d_id = str(driver_info.get("driverId", "")).strip()
                full_name = f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip()
                canon_d_id = self.resolve_driver_id(raw_d_id, full_name=full_name)

                const_info = q_res.get("Constructor", {})
                raw_c_id = str(const_info.get("constructorId", "")).strip()
                canon_c_id = self.resolve_constructor_id(raw_c_id, name=str(const_info.get("name", "")))

                d_num_val = q_res.get("number")
                d_num = int(d_num_val) if d_num_val and str(d_num_val).isdigit() else -1

                pos_val = q_res.get("position")
                pos = int(pos_val) if pos_val and str(pos_val).isdigit() else 999

                q1_str = str(q_res.get("Q1", "")).strip()
                q2_str = str(q_res.get("Q2", "")).strip()
                q3_str = str(q_res.get("Q3", "")).strip()

                def parse_ms(t_str: str) -> Optional[int]:
                    if not t_str or ":" not in t_str:
                        return None
                    try:
                        parts = t_str.split(":")
                        m = int(parts[0])
                        s_parts = parts[1].split(".")
                        s = int(s_parts[0])
                        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
                        return m * 60000 + s * 1000 + ms
                    except Exception:
                        return None

                records.append({
                    "season": int(season),
                    "round": rnd,
                    "driver_id": canon_d_id,
                    "constructor_id": canon_c_id,
                    "driver_number": d_num,
                    "position": pos,
                    "q1_time_ms": parse_ms(q1_str),
                    "q1_time_str": q1_str,
                    "q2_time_ms": parse_ms(q2_str),
                    "q2_time_str": q2_str,
                    "q3_time_ms": parse_ms(q3_str),
                    "q3_time_str": q3_str,
                    "laps_completed": 0,
                    "source_api": "jolpica.ergast",
                    "retrieved_at": now,
                })

        df = pd.DataFrame(records, columns=QUALIFYING_RESULTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def ingest_season_sprints(self, season: int, force: bool = False) -> pd.DataFrame:
        """Ingest sprint results for a season into SPRINT_RESULTS_SCHEMA."""
        races = self.client.fetch_sprint_results(season, force=force)
        if not races:
            return pd.DataFrame(columns=SPRINT_RESULTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for race in races:
            rnd = int(race.get("round", 1))

            for res in race.get("SprintResults", []):
                driver_info = res.get("Driver", {})
                raw_d_id = str(driver_info.get("driverId", "")).strip()
                full_name = f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip()
                canon_d_id = self.resolve_driver_id(raw_d_id, full_name=full_name)

                const_info = res.get("Constructor", {})
                raw_c_id = str(const_info.get("constructorId", "")).strip()
                canon_c_id = self.resolve_constructor_id(raw_c_id, name=str(const_info.get("name", "")))

                d_num_val = res.get("number")
                d_num = int(d_num_val) if d_num_val and str(d_num_val).isdigit() else -1

                grid_val = res.get("grid")
                grid = int(grid_val) if grid_val and str(grid_val).isdigit() else 0

                pos_val = res.get("position")
                fin_pos = int(pos_val) if pos_val and str(pos_val).isdigit() else None
                pos_text = str(res.get("positionText", ""))
                pts = float(res.get("points", 0.0))
                laps = int(res.get("laps", 0)) if str(res.get("laps", "")).isdigit() else 0
                status = str(res.get("status", "Finished"))

                s_lower = status.lower()
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

                time_info = res.get("Time", {})
                time_ms_val = time_info.get("millis")
                tot_time_ms = int(time_ms_val) if time_ms_val and str(time_ms_val).isdigit() else None
                time_str = str(time_info.get("time", ""))

                fl_info = res.get("FastestLap", {})
                fl_rank_val = fl_info.get("rank")
                fl_rank = int(fl_rank_val) if fl_rank_val and str(fl_rank_val).isdigit() else None
                fl_time = str(fl_info.get("Time", {}).get("time", ""))
                fl_spd_val = fl_info.get("AverageSpeed", {}).get("speed")
                fl_spd = float(fl_spd_val) if fl_spd_val is not None else None

                records.append({
                    "season": int(season),
                    "round": rnd,
                    "driver_id": canon_d_id,
                    "constructor_id": canon_c_id,
                    "driver_number": d_num,
                    "grid_position": grid,
                    "finish_position": fin_pos,
                    "position_text": pos_text,
                    "points": pts,
                    "laps_completed": laps,
                    "status": status,
                    "is_classified": is_classified,
                    "is_dnf": is_dnf,
                    "is_dns": is_dns,
                    "is_dsq": is_dsq,
                    "total_time_ms": tot_time_ms,
                    "time_text": time_str,
                    "fastest_lap_rank": fl_rank,
                    "fastest_lap_time": fl_time,
                    "fastest_lap_speed": fl_spd,
                    "source_api": "jolpica.ergast",
                    "retrieved_at": now,
                })

        df = pd.DataFrame(records, columns=SPRINT_RESULTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round", "driver_id"], keep="first")

    def ingest_season_calendar(self, season: int, force: bool = False) -> pd.DataFrame:
        """Ingest Grand Prix events calendar for a season into EVENTS_SCHEMA."""
        schedule = self.client.fetch_schedule(season, force=force)
        if not schedule:
            return pd.DataFrame(columns=EVENTS_SCHEMA.names)

        now = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for race in schedule:
            rnd = int(race.get("round", 1))
            ev_name = str(race.get("raceName", "")).strip()
            circ_info = race.get("Circuit", {})
            raw_c_id = str(circ_info.get("circuitId", "")).strip()
            circ_name = str(circ_info.get("circuitName", "")).strip()
            canon_c_id = self.resolve_circuit_id(raw_c_id, name=circ_name)

            loc_info = circ_info.get("Location", {})
            country = str(loc_info.get("country", "")).strip()
            locality = str(loc_info.get("locality", "")).strip()
            ev_date = str(race.get("date", "")).strip()

            records.append({
                "season": int(season),
                "round": rnd,
                "event_name": ev_name,
                "circuit_id": canon_c_id,
                "country": country,
                "locality": locality,
                "event_date": ev_date,
                "source_api": "jolpica.ergast",
                "retrieved_at": now,
            })

        df = pd.DataFrame(records, columns=EVENTS_SCHEMA.names)
        return df.drop_duplicates(subset=["season", "round"], keep="first")

    # -----------------------------------------------------------------------
    # Multi-Season Orchestrator & Idempotent Persistence
    # -----------------------------------------------------------------------

    def execute_season_ingestion(
        self,
        seasons: Union[int, List[int]],
        dry_run: bool = False,
        force: bool = False,
    ) -> JolpicaIngestResult:
        """Execute full ingestion, canonicalization, and persistence across target seasons."""
        target_seasons = [seasons] if isinstance(seasons, int) else list(seasons)
        written: Dict[str, Path] = {}

        total_d_standings = 0
        total_c_standings = 0
        total_races = 0
        total_qual = 0
        total_sprints = 0
        total_cal = 0

        all_d_standings: List[pd.DataFrame] = []
        all_c_standings: List[pd.DataFrame] = []

        for s in target_seasons:
            logger.info(f"Executing Jolpica ingestion for season {s}...")

            # 1. Standings
            df_d_stand, df_c_stand = self.ingest_full_season_standings(s, force=force)
            if not df_d_stand.empty:
                all_d_standings.append(df_d_stand)
                total_d_standings += len(df_d_stand)
                if not dry_run:
                    p = get_jolpica_path(f"jolpica_driver_standings_{s}.parquet", base_dir=self.base_dir)
                    written[f"driver_standings_{s}"] = save_parquet(df_d_stand, p, schema=DRIVER_STANDINGS_SCHEMA)

            if not df_c_stand.empty:
                all_c_standings.append(df_c_stand)
                total_c_standings += len(df_c_stand)
                if not dry_run:
                    p = get_jolpica_path(f"jolpica_constructor_standings_{s}.parquet", base_dir=self.base_dir)
                    written[f"constructor_standings_{s}"] = save_parquet(df_c_stand, p, schema=CONSTRUCTOR_STANDINGS_SCHEMA)

            # 2. Race Results
            df_rr = self.ingest_season_race_results(s, force=force)
            if not df_rr.empty:
                total_races += len(df_rr)
                if not dry_run:
                    p = get_jolpica_path(f"jolpica_race_results_{s}.parquet", base_dir=self.base_dir)
                    written[f"race_results_{s}"] = save_parquet(df_rr, p, schema=RACE_RESULTS_SCHEMA)

            # 3. Qualifying Results
            df_q = self.ingest_season_qualifying(s, force=force)
            if not df_q.empty:
                total_qual += len(df_q)
                if not dry_run:
                    p = get_jolpica_path(f"jolpica_qualifying_{s}.parquet", base_dir=self.base_dir)
                    written[f"qualifying_{s}"] = save_parquet(df_q, p, schema=QUALIFYING_RESULTS_SCHEMA)

            # 4. Sprints
            df_sp = self.ingest_season_sprints(s, force=force)
            if not df_sp.empty:
                total_sprints += len(df_sp)
                if not dry_run:
                    p = get_jolpica_path(f"jolpica_sprints_{s}.parquet", base_dir=self.base_dir)
                    written[f"sprints_{s}"] = save_parquet(df_sp, p, schema=SPRINT_RESULTS_SCHEMA)

            # 5. Calendar
            df_cal = self.ingest_season_calendar(s, force=force)
            if not df_cal.empty:
                total_cal += len(df_cal)
                if not dry_run:
                    p = get_jolpica_path(f"jolpica_calendar_{s}.parquet", base_dir=self.base_dir)
                    written[f"calendar_{s}"] = save_parquet(df_cal, p, schema=EVENTS_SCHEMA)

        # Reconcile master compiled standings without duplicates
        if not dry_run and all_d_standings:
            master_d_path = get_historical_path("driver_standings", base_dir=self.base_dir)
            existing_d = load_parquet(master_d_path) if dataset_exists(master_d_path) else pd.DataFrame()
            combined_d = pd.concat([existing_d] + all_d_standings, ignore_index=True) if not existing_d.empty else pd.concat(all_d_standings, ignore_index=True)
            reconciled_d = combined_d.drop_duplicates(subset=["season", "round", "driver_id"], keep="last").sort_values(by=["season", "round", "position"]).reset_index(drop=True)
            written["historical_driver_standings"] = save_parquet(reconciled_d, master_d_path, schema=DRIVER_STANDINGS_SCHEMA)

        if not dry_run and all_c_standings:
            master_c_path = get_historical_path("constructor_standings", base_dir=self.base_dir)
            existing_c = load_parquet(master_c_path) if dataset_exists(master_c_path) else pd.DataFrame()
            combined_c = pd.concat([existing_c] + all_c_standings, ignore_index=True) if not existing_c.empty else pd.concat(all_c_standings, ignore_index=True)
            reconciled_c = combined_c.drop_duplicates(subset=["season", "round", "constructor_id"], keep="last").sort_values(by=["season", "round", "position"]).reset_index(drop=True)
            written["historical_constructor_standings"] = save_parquet(reconciled_c, master_c_path, schema=CONSTRUCTOR_STANDINGS_SCHEMA)

        msg = (
            f"Jolpica ingestion complete across seasons {target_seasons}: {total_d_standings} driver standings, "
            f"{total_c_standings} constructor standings, {total_races} race results, {total_qual} qualifying, "
            f"{total_sprints} sprint results, and {total_cal} calendar events."
        )

        return JolpicaIngestResult(
            success=True,
            seasons=target_seasons,
            driver_standings_count=total_d_standings,
            constructor_standings_count=total_c_standings,
            race_results_count=total_races,
            qualifying_count=total_qual,
            sprint_count=total_sprints,
            calendar_count=total_cal,
            unresolved_entities=self.unresolved,
            datasets_written=written,
            summary_message=msg,
        )
