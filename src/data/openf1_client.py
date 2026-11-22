"""PACE: OpenF1 Race Intelligence Client & Ingestion Engine.

Provides an authoritative client for OpenF1 (https://api.openf1.org/v1/), implementing
request throttling, rate limit adherence (3 req/sec, 30 req/min), exponential backoff
with jitter, HTTP 429 and Retry-After handling, local disk JSON caching, canonical
entity resolution, PyArrow schema normalization, and selective telemetry ingestion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pyarrow as pa
import requests

from src.data.canonical_registry import (
    CanonicalEntityRegistry,
    get_canonical_registry,
    normalize_key,
)
from src.data.schemas import (
    OPENF1_MEETINGS_SCHEMA,
    OPENF1_SESSIONS_SCHEMA,
    OPENF1_DRIVERS_SCHEMA,
    OPENF1_INTERVALS_SCHEMA,
    OPENF1_STINTS_SCHEMA,
    OPENF1_RACE_CONTROL_SCHEMA,
    OPENF1_WEATHER_SCHEMA,
    OPENF1_OVERTAKES_SCHEMA,
    OPENF1_POSITION_SCHEMA,
    OPENF1_PIT_SCHEMA,
    OPENF1_CAR_DATA_SCHEMA,
    OPENF1_LOCATION_SCHEMA,
)
from src.data.storage import (
    DEFAULT_BASE_DIR,
    get_openf1_dir,
    get_openf1_path,
    get_openf1_cache_dir,
    save_parquet,
    load_parquet,
    dataset_exists,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OpenF1Error(Exception):
    """Base exception for all OpenF1 client and ingestion errors."""
    pass


class OpenF1APIError(OpenF1Error):
    """Raised on non-transient HTTP errors (4xx/5xx)."""
    def __init__(self, status_code: int, message: str, url: str = ""):
        self.status_code = status_code
        self.url = url
        super().__init__(f"OpenF1 API Error [{status_code}] at {url}: {message}")


class OpenF1RateLimitError(OpenF1Error):
    """Raised when rate limits (HTTP 429) persist after backoff retries."""
    pass


class OpenF1ConnectionError(OpenF1Error):
    """Raised on network connection failures and timeouts."""
    pass


# ---------------------------------------------------------------------------
# Configuration & Statistics
# ---------------------------------------------------------------------------

@dataclass
class OpenF1Config:
    """Configuration options for OpenF1 API interactions."""
    base_url: str = "https://api.openf1.org/v1"
    timeout: float = 12.0
    max_retries: int = 4
    backoff_base: float = 0.5
    rate_limit_pause: float = 0.35              # Stay under 3 req/sec (< 2.85 req/sec)
    max_requests_per_minute: int = 30           # Free tier rate limit
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    user_agent: str = "PACE-F1-Analytics/1.0 (https://github.com/dash-pdf/PACE)"
    live_mode: bool = False                     # Historical-only by default


@dataclass
class OpenF1Stats:
    """Runtime metrics tracking OpenF1 requests and caching efficiency."""
    requests_made: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries_attempted: int = 0
    rate_limits_encountered: int = 0
    bytes_downloaded: int = 0
    errors_encountered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retries_attempted": self.retries_attempted,
            "rate_limits_encountered": self.rate_limits_encountered,
            "bytes_downloaded": self.bytes_downloaded,
            "errors_encountered": self.errors_encountered,
            "cache_hit_ratio_pct": (
                round((self.cache_hits / (self.cache_hits + self.cache_misses)) * 100.0, 2)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class OpenF1RateLimiter:
    """Enforces client-side rate limits: max 3 req/sec and max 30 req/min."""

    def __init__(self, min_pause: float = 0.35, max_per_minute: int = 30) -> None:
        self.min_pause = min_pause
        self.max_per_minute = max_per_minute
        self.last_request_time: float = 0.0
        self.rolling_window: deque[float] = deque()

    def wait_if_needed(self) -> None:
        """Throttle requests to respect both per-second and per-minute constraints."""
        now = time.time()

        # 1. Enforce minimum pause between consecutive requests (burst limiter)
        if self.min_pause > 0.0:
            elapsed = now - self.last_request_time
            if elapsed < self.min_pause:
                sleep_duration = self.min_pause - elapsed
                time.sleep(sleep_duration)
                now = time.time()

        # 2. Enforce rolling 60-second request volume cap
        if self.max_per_minute > 0:
            # Purge timestamps older than 60 seconds
            while self.rolling_window and (now - self.rolling_window[0] > 60.0):
                self.rolling_window.popleft()

            if len(self.rolling_window) >= self.max_per_minute:
                # Wait until oldest request timestamp exits the 60s window
                oldest = self.rolling_window[0]
                wait_time = max(0.01, 60.0 - (now - oldest) + 0.1)
                logger.warning(f"OpenF1 rate limiter: 30 req/min reached. Throttling for {wait_time:.2f}s...")
                time.sleep(wait_time)
                now = time.time()
                while self.rolling_window and (now - self.rolling_window[0] > 60.0):
                    self.rolling_window.popleft()

        self.last_request_time = now
        self.rolling_window.append(now)


# ---------------------------------------------------------------------------
# OpenF1 Core API Client
# ---------------------------------------------------------------------------

class OpenF1Client:
    """Authoritative client for the OpenF1 REST API with throttling and disk caching."""

    def __init__(
        self,
        config: Optional[OpenF1Config] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or OpenF1Config()
        self.session = session or requests.Session()
        self.stats = OpenF1Stats()
        self.rate_limiter = OpenF1RateLimiter(
            min_pause=self.config.rate_limit_pause,
            max_per_minute=self.config.max_requests_per_minute,
        )

        self.cache_dir = self.config.cache_dir or get_openf1_cache_dir()
        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Path:
        """Derive reproducible SHA-256 cache filename for a given request."""
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        norm_qs = urlencode(sorted(clean_params.items()))
        unique_sig = f"{endpoint}?{norm_qs}".encode("utf-8")
        h = hashlib.sha256(unique_sig).hexdigest()
        clean_ep = endpoint.replace("/", "_")
        return self.cache_dir / f"openf1_{clean_ep}_{h[:16]}.json"

    def _read_cache(self, cache_file: Path) -> Optional[Any]:
        """Attempt to read and parse JSON payload from local disk cache."""
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.stats.cache_hits += 1
            return data
        except Exception as e:
            logger.warning(f"Corrupted cache file {cache_file}: {e}. Removing.")
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def _write_cache(self, cache_file: Path, data: Any) -> None:
        """Persist JSON response to disk atomically."""
        tmp_file = cache_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            tmp_file.replace(cache_file)
        except Exception as e:
            logger.warning(f"Failed to persist cache file {cache_file}: {e}")
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    def request_endpoint(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch records from OpenF1 endpoint with rate limiting, retries, and disk cache."""
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_file = self._get_cache_path(endpoint, clean_params)

        # 1. Check local disk cache
        if self.config.use_cache and not force_refresh:
            cached_data = self._read_cache(cache_file)
            if cached_data is not None:
                return cached_data

        self.stats.cache_misses += 1
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
        }

        # 2. Retry loop with exponential backoff and jitter
        has_comparison = any(">" in str(k) or "<" in str(k) for k in clean_params.keys())

        for attempt in range(1, self.config.max_retries + 1):
            self.rate_limiter.wait_if_needed()
            self.stats.requests_made += 1

            try:
                # OpenF1 expects literal comparison operators (e.g. speed>=200) rather
                # than percent-encoded '%3E%3D' which standard requests URL encoding produces.
                if has_comparison and hasattr(self.session, "prepare_request") and not hasattr(self.session, "_mock_return_value"):
                    req = requests.Request("GET", url, params=clean_params, headers=headers)
                    prep = self.session.prepare_request(req)
                    if prep.url:
                        prep.url = (
                            prep.url.replace("%3E%3D=", ">=")
                            .replace("%3E%3D", ">=")
                            .replace("%3C%3D=", "<=")
                            .replace("%3C%3D", "<=")
                            .replace("%3E=", ">=")
                            .replace("%3C=", "<=")
                            .replace("%3E", ">")
                            .replace("%3C", "<")
                        )
                    response = self.session.send(prep, timeout=self.config.timeout)
                else:
                    response = self.session.get(
                        url,
                        params=clean_params,
                        headers=headers,
                        timeout=self.config.timeout,
                    )
            except (requests.ConnectionError, requests.Timeout) as e:
                self.stats.retries_attempted += 1
                logger.warning(f"Connection failure on {url} (attempt {attempt}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries:
                    self.stats.errors_encountered += 1
                    raise OpenF1ConnectionError(f"Persistent network error querying {url}: {e}") from e
                sleep_sec = self.config.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.05, 0.25)
                time.sleep(sleep_sec)
                continue

            status = response.status_code

            # Success
            if status == 200:
                self.stats.bytes_downloaded += len(response.content)
                try:
                    data = response.json()
                    if isinstance(data, list):
                        records = data
                    elif isinstance(data, dict):
                        records = [data]
                    else:
                        records = []

                    if self.config.use_cache:
                        self._write_cache(cache_file, records)
                    return records
                except ValueError as e:
                    self.stats.errors_encountered += 1
                    raise OpenF1APIError(status, f"Malformed JSON payload: {e}", url=url) from e

            # OpenF1 returns HTTP 404 when no results match a filter (e.g. empty results)
            if status == 404:
                try:
                    detail = response.json().get("detail", "")
                    if "No results found" in detail:
                        records: List[Dict[str, Any]] = []
                        if self.config.use_cache:
                            self._write_cache(cache_file, records)
                        return records
                except Exception:
                    pass
                records = []
                if self.config.use_cache:
                    self._write_cache(cache_file, records)
                return records

            # HTTP 429: Rate Limit Exceeded
            if status == 429:
                self.stats.rate_limits_encountered += 1
                self.stats.retries_attempted += 1
                retry_after_str = response.headers.get("Retry-After")
                if retry_after_str and retry_after_str.isdigit():
                    wait_sec = float(retry_after_str) + 0.1
                else:
                    wait_sec = self.config.backoff_base * (2 ** attempt) + random.uniform(0.2, 0.5)

                logger.warning(f"HTTP 429 Rate Limited on {url}. Backing off for {wait_sec:.2f}s (attempt {attempt})...")
                if attempt == self.config.max_retries:
                    self.stats.errors_encountered += 1
                    raise OpenF1RateLimitError(f"Exhausted retries on HTTP 429 for {url}")
                time.sleep(wait_sec)
                continue

            # HTTP 5xx Server Errors
            if 500 <= status < 600:
                self.stats.retries_attempted += 1
                wait_sec = self.config.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.3)
                logger.warning(f"HTTP {status} server error on {url}. Retrying in {wait_sec:.2f}s...")
                if attempt == self.config.max_retries:
                    self.stats.errors_encountered += 1
                    raise OpenF1APIError(status, f"Persistent server error: {response.text[:200]}", url=url)
                time.sleep(wait_sec)
                continue

            # Other 4xx client errors
            self.stats.errors_encountered += 1
            raise OpenF1APIError(status, f"Client error: {response.text[:200]}", url=url)

        raise OpenF1APIError(0, "Unexpected loop termination in request_endpoint", url=url)

    # -----------------------------------------------------------------------
    # Endpoint Access Methods
    # -----------------------------------------------------------------------

    def fetch_meetings(self, year: Optional[int] = None, meeting_key: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch Grand Prix meetings and pre-season tests."""
        params: Dict[str, Any] = {}
        if year is not None:
            params["year"] = int(year)
        if meeting_key is not None:
            params["meeting_key"] = int(meeting_key)
        return self.request_endpoint("meetings", params)

    def fetch_sessions(
        self,
        year: Optional[int] = None,
        meeting_key: Optional[int] = None,
        session_key: Optional[int] = None,
        session_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch session metadata (Practice, Qualifying, Sprint, Race)."""
        params: Dict[str, Any] = {}
        if year is not None:
            params["year"] = int(year)
        if meeting_key is not None:
            params["meeting_key"] = int(meeting_key)
        if session_key is not None:
            params["session_key"] = int(session_key)
        if session_type is not None:
            params["session_type"] = session_type
        return self.request_endpoint("sessions", params)

    def fetch_drivers(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch driver entries for a specific session."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if driver_number is not None:
            params["driver_number"] = int(driver_number)
        return self.request_endpoint("drivers", params)

    def fetch_intervals(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch real-time intervals and gap-to-leader records."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if driver_number is not None:
            params["driver_number"] = int(driver_number)
        return self.request_endpoint("intervals", params)

    def fetch_laps(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch lap times, sector durations, and speed trap observations."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if driver_number is not None:
            params["driver_number"] = int(driver_number)
        return self.request_endpoint("laps", params)

    def fetch_position(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch chronological track running position changes."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if driver_number is not None:
            params["driver_number"] = int(driver_number)
        return self.request_endpoint("position", params)

    def fetch_stints(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch tyre compound stints and lap ranges."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if driver_number is not None:
            params["driver_number"] = int(driver_number)
        return self.request_endpoint("stints", params)

    def fetch_pit(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch pit stop timing and duration."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if driver_number is not None:
            params["driver_number"] = int(driver_number)
        return self.request_endpoint("pit", params)

    def fetch_race_control(
        self,
        session_key: int,
        flag: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch race control messages (Flags, Safety Car, VSC, Incidents)."""
        params: Dict[str, Any] = {"session_key": int(session_key)}
        if flag is not None:
            params["flag"] = flag
        if category is not None:
            params["category"] = category
        return self.request_endpoint("race_control", params)

    def fetch_weather(self, session_key: int) -> List[Dict[str, Any]]:
        """Fetch session environmental conditions (air/track temp, wind, rain)."""
        return self.request_endpoint("weather", {"session_key": int(session_key)})

    def fetch_overtakes(self, session_key: int) -> List[Dict[str, Any]]:
        """Fetch recorded on-track overtaking maneuvers."""
        return self.request_endpoint("overtakes", {"session_key": int(session_key)})

    def fetch_session_result(self, session_key: int) -> List[Dict[str, Any]]:
        """Fetch session classified results and finishing status."""
        return self.request_endpoint("session_result", {"session_key": int(session_key)})

    def fetch_car_data(
        self,
        session_key: int,
        driver_number: int,
        speed_min: Optional[int] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch high-frequency telemetry (speed, rpm, gear, throttle, brake, drs)."""
        params: Dict[str, Any] = {
            "session_key": int(session_key),
            "driver_number": int(driver_number),
        }
        if speed_min is not None:
            params["speed>="] = int(speed_min)
        if date_start is not None:
            params["date>="] = date_start
        if date_end is not None:
            params["date<="] = date_end
        return self.request_endpoint("car_data", params)

    def fetch_location(
        self,
        session_key: int,
        driver_number: int,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch high-frequency approximate car coordinates (x, y, z)."""
        params: Dict[str, Any] = {
            "session_key": int(session_key),
            "driver_number": int(driver_number),
        }
        if date_start is not None:
            params["date>="] = date_start
        if date_end is not None:
            params["date<="] = date_end
        return self.request_endpoint("location", params)

    def check_connectivity(self) -> Tuple[bool, str]:
        """Verify network connectivity and OpenF1 API availability."""
        try:
            records = self.fetch_meetings(year=2023)
            return True, f"Online. Retrieved {len(records)} meetings for 2023."
        except Exception as e:
            return False, f"Offline: {e}"


# ---------------------------------------------------------------------------
# Ingestion Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class OpenF1IngestResult:
    """Summary of OpenF1 historical ingestion operation."""
    success: bool
    year: int
    meetings_ingested: int = 0
    sessions_ingested: int = 0
    records_by_category: Dict[str, int] = field(default_factory=dict)
    unresolved_entities: Dict[str, Set[str]] = field(default_factory=dict)
    datasets_written: Dict[str, str] = field(default_factory=dict)
    summary_message: str = ""
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Ingestion Engine & PyArrow Normalizer
# ---------------------------------------------------------------------------

class OpenF1IngestionEngine:
    """Ingests OpenF1 datasets, normalizes canonical entities, and persists Parquet files."""

    def __init__(
        self,
        client: Optional[OpenF1Client] = None,
        registry: Optional[CanonicalEntityRegistry] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.client = client or OpenF1Client()
        self.registry = registry or get_canonical_registry()
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.unresolved: Dict[str, Set[str]] = {
            "drivers": set(),
            "constructors": set(),
            "circuits": set(),
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -----------------------------------------------------------------------
    # Canonical Resolution Helpers
    # -----------------------------------------------------------------------

    def resolve_circuit_key(self, circuit_short_name: str, location: str, country_name: str) -> str:
        """Resolve OpenF1 circuit metadata to canonical circuit ID."""
        for query in [circuit_short_name, location, country_name]:
            if query:
                res = self.registry.resolve_circuit(query)
                if res.is_resolved and res.canonical_id:
                    return res.canonical_id

        # Fallback to normalized short name
        fallback_id = normalize_key(circuit_short_name or location or country_name)
        self.unresolved["circuits"].add(f"{circuit_short_name} ({location}, {country_name})")
        return fallback_id

    def resolve_constructor_name(self, team_name: str) -> str:
        """Resolve OpenF1 team name to canonical constructor ID."""
        if not team_name:
            return "unknown"
        res = self.registry.resolve_constructor(team_name)
        if res.is_resolved and res.canonical_id:
            return res.canonical_id

        norm = normalize_key(team_name)
        self.unresolved["constructors"].add(team_name)
        return norm

    def resolve_driver_identity(
        self,
        driver_number: int,
        full_name: str = "",
        acronym: str = "",
        year: Optional[int] = None,
    ) -> str:
        """Resolve driver identity using number, name, acronym, and season context."""
        # Try full name first if available (most specific)
        if full_name:
            res_name = self.registry.resolve_driver(full_name, season=year)
            if res_name.is_resolved and res_name.canonical_id:
                return res_name.canonical_id

        # Try 3-letter acronym
        if acronym:
            res_code = self.registry.resolve_driver(acronym, season=year)
            if res_code.is_resolved and res_code.canonical_id:
                return res_code.canonical_id

        # Try driver number in season
        if driver_number > 0:
            res_num = self.registry.resolve_driver(driver_number, season=year)
            if res_num.is_resolved and res_num.canonical_id:
                return res_num.canonical_id

        norm = normalize_key(full_name or acronym or f"driver_{driver_number}")
        self.unresolved["drivers"].add(f"#{driver_number} {full_name} ({acronym})")
        return norm

    # -----------------------------------------------------------------------
    # Ingestion Methods
    # -----------------------------------------------------------------------

    def ingest_meetings_for_year(self, year: int, force: bool = False) -> pd.DataFrame:
        """Ingest all Grand Prix meetings for a championship year."""
        raw_meetings = self.client.fetch_meetings(year=year)
        if not raw_meetings:
            return pd.DataFrame(columns=OPENF1_MEETINGS_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for m in raw_meetings:
            c_name = m.get("circuit_short_name") or ""
            loc = m.get("location") or ""
            country = m.get("country_name") or ""
            canonical_cid = self.resolve_circuit_key(c_name, loc, country)

            rows.append({
                "meeting_key": int(m.get("meeting_key", 0)),
                "meeting_name": str(m.get("meeting_name") or ""),
                "meeting_official_name": str(m.get("meeting_official_name") or ""),
                "location": loc,
                "country_key": int(m.get("country_key", 0)),
                "country_code": str(m.get("country_code") or ""),
                "country_name": country,
                "circuit_key": int(m.get("circuit_key", 0)),
                "circuit_short_name": c_name,
                "circuit_id": canonical_cid,
                "year": int(m.get("year", year)),
                "date_start": str(m.get("date_start") or ""),
                "date_end": str(m.get("date_end") or ""),
                "gmt_offset": str(m.get("gmt_offset") or ""),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })

        df = pd.DataFrame(rows)
        # Ensure correct column ordering
        return df[OPENF1_MEETINGS_SCHEMA.names]

    def ingest_sessions_for_year(self, year: int, force: bool = False) -> pd.DataFrame:
        """Ingest all session metadata for a championship year."""
        raw_sessions = self.client.fetch_sessions(year=year)
        if not raw_sessions:
            return pd.DataFrame(columns=OPENF1_SESSIONS_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for s in raw_sessions:
            c_name = s.get("circuit_short_name") or ""
            loc = s.get("location") or ""
            country = s.get("country_name") or ""
            canonical_cid = self.resolve_circuit_key(c_name, loc, country)

            rows.append({
                "session_key": int(s.get("session_key", 0)),
                "meeting_key": int(s.get("meeting_key", 0)),
                "session_name": str(s.get("session_name") or ""),
                "session_type": str(s.get("session_type") or ""),
                "year": int(s.get("year", year)),
                "circuit_key": int(s.get("circuit_key", 0)),
                "circuit_short_name": c_name,
                "circuit_id": canonical_cid,
                "country_name": country,
                "location": loc,
                "date_start": str(s.get("date_start") or ""),
                "date_end": str(s.get("date_end") or ""),
                "gmt_offset": str(s.get("gmt_offset") or ""),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })

        df = pd.DataFrame(rows)
        return df[OPENF1_SESSIONS_SCHEMA.names]

    def build_session_driver_map(self, session_key: int, year: int) -> Dict[int, Dict[str, str]]:
        """Construct a lookup map from driver_number to canonical identity for a session."""
        raw_drivers = self.client.fetch_drivers(session_key=session_key)
        driver_map: Dict[int, Dict[str, str]] = {}

        for d in raw_drivers:
            num = int(d.get("driver_number", 0))
            fn = str(d.get("full_name") or "")
            acronym = str(d.get("name_acronym") or "")
            team = str(d.get("team_name") or "")

            c_did = self.resolve_driver_identity(num, full_name=fn, acronym=acronym, year=year)
            c_cid = self.resolve_constructor_name(team)

            driver_map[num] = {
                "driver_id": c_did,
                "constructor_id": c_cid,
                "name_acronym": acronym,
                "full_name": fn,
                "team_name": team,
            }
        return driver_map

    def ingest_session_drivers(self, session_key: int, year: int) -> pd.DataFrame:
        """Ingest driver roster for a specific session."""
        raw_drivers = self.client.fetch_drivers(session_key=session_key)
        if not raw_drivers:
            return pd.DataFrame(columns=OPENF1_DRIVERS_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for d in raw_drivers:
            num = int(d.get("driver_number", 0))
            fn = str(d.get("full_name") or "")
            acronym = str(d.get("name_acronym") or "")
            team = str(d.get("team_name") or "")

            c_did = self.resolve_driver_identity(num, full_name=fn, acronym=acronym, year=year)
            c_cid = self.resolve_constructor_name(team)

            rows.append({
                "session_key": int(d.get("session_key", session_key)),
                "meeting_key": int(d.get("meeting_key", 0)),
                "driver_number": num,
                "driver_id": c_did,
                "broadcast_name": str(d.get("broadcast_name") or ""),
                "full_name": fn,
                "name_acronym": acronym,
                "team_name": team,
                "constructor_id": c_cid,
                "team_colour": str(d.get("team_colour") or ""),
                "first_name": str(d.get("first_name") or ""),
                "last_name": str(d.get("last_name") or ""),
                "country_code": str(d.get("country_code") or ""),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })
        df = pd.DataFrame(rows)
        return df[OPENF1_DRIVERS_SCHEMA.names]

    @staticmethod
    def _safe_float(val: Any, default: float = np.nan) -> float:
        """Parse float safely, handling strings like '+1 LAP', None, etc."""
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
        """Parse int safely, returning default if invalid or None."""
        if val is None or pd.isna(val):
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def ingest_session_intervals(
        self,
        session_key: int,
        driver_map: Optional[Dict[int, Dict[str, str]]] = None,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Ingest gap to leader and interval records."""
        raw_intervals = self.client.fetch_intervals(session_key=session_key)
        if not raw_intervals:
            return pd.DataFrame(columns=OPENF1_INTERVALS_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for inv in raw_intervals:
            num = int(inv.get("driver_number", 0))
            d_id = (
                driver_map[num]["driver_id"]
                if driver_map and num in driver_map
                else self.resolve_driver_identity(num, year=year)
            )

            rows.append({
                "session_key": int(inv.get("session_key", session_key)),
                "meeting_key": int(inv.get("meeting_key", 0)),
                "date": str(inv.get("date") or ""),
                "driver_number": num,
                "driver_id": d_id,
                "gap_to_leader": self._safe_float(inv.get("gap_to_leader")),
                "interval": self._safe_float(inv.get("interval")),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })
        df = pd.DataFrame(rows)
        return df[OPENF1_INTERVALS_SCHEMA.names]

    def ingest_session_stints(
        self,
        session_key: int,
        driver_map: Optional[Dict[int, Dict[str, str]]] = None,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Ingest tyre stint sequences and tyre age observations."""
        raw_stints = self.client.fetch_stints(session_key=session_key)
        if not raw_stints:
            return pd.DataFrame(columns=OPENF1_STINTS_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for s in raw_stints:
            num = int(s.get("driver_number", 0))
            d_id = (
                driver_map[num]["driver_id"]
                if driver_map and num in driver_map
                else self.resolve_driver_identity(num, year=year)
            )
            l_start = int(s.get("lap_start", 0)) if s.get("lap_start") is not None else 0
            l_end = int(s.get("lap_end", 0)) if s.get("lap_end") is not None else 0
            stint_len = max(0, l_end - l_start + 1) if l_end >= l_start and l_start > 0 else 0

            rows.append({
                "session_key": int(s.get("session_key", session_key)),
                "meeting_key": int(s.get("meeting_key", 0)),
                "stint_number": int(s.get("stint_number", 0)),
                "driver_number": num,
                "driver_id": d_id,
                "compound": str(s.get("compound") or "UNKNOWN").upper(),
                "lap_start": l_start,
                "lap_end": l_end,
                "stint_length": stint_len,
                "tyre_age_at_start": int(s.get("tyre_age_at_start", 0)) if s.get("tyre_age_at_start") is not None else 0,
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })
        df = pd.DataFrame(rows)
        return df[OPENF1_STINTS_SCHEMA.names]

    def ingest_session_race_control(
        self,
        session_key: int,
        driver_map: Optional[Dict[int, Dict[str, str]]] = None,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Ingest race control messages (Flags, Safety Car, Red Flags, Incidents)."""
        raw_rc = self.client.fetch_race_control(session_key=session_key)
        if not raw_rc:
            return pd.DataFrame(columns=OPENF1_RACE_CONTROL_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for rc in raw_rc:
            raw_dnum = rc.get("driver_number")
            d_num: Optional[int] = int(raw_dnum) if raw_dnum is not None else None
            d_id: Optional[str] = None
            if d_num is not None:
                d_id = (
                    driver_map[d_num]["driver_id"]
                    if driver_map and d_num in driver_map
                    else self.resolve_driver_identity(d_num, year=year)
                )

            raw_lap = rc.get("lap_number")
            lap_num: Optional[int] = int(raw_lap) if raw_lap is not None else None

            raw_sec = rc.get("sector")
            sector: Optional[int] = int(raw_sec) if raw_sec is not None else None

            rows.append({
                "session_key": int(rc.get("session_key", session_key)),
                "meeting_key": int(rc.get("meeting_key", 0)),
                "date": str(rc.get("date") or ""),
                "lap_number": lap_num,
                "category": str(rc.get("category") or "Other"),
                "flag": str(rc.get("flag") or "CLEAR"),
                "scope": str(rc.get("scope") or "Track"),
                "sector": sector,
                "driver_number": d_num,
                "driver_id": d_id,
                "message": str(rc.get("message") or ""),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })
        df = pd.DataFrame(rows)
        return df[OPENF1_RACE_CONTROL_SCHEMA.names]

    def ingest_session_weather(self, session_key: int) -> pd.DataFrame:
        """Ingest session weather observations (air/track temperature, wind, humidity, rain)."""
        raw_w = self.client.fetch_weather(session_key=session_key)
        if not raw_w:
            return pd.DataFrame(columns=OPENF1_WEATHER_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for w in raw_w:
            rows.append({
                "session_key": int(w.get("session_key", session_key)),
                "meeting_key": int(w.get("meeting_key", 0)),
                "date": str(w.get("date") or ""),
                "air_temperature": self._safe_float(w.get("air_temperature")),
                "track_temperature": self._safe_float(w.get("track_temperature")),
                "humidity": self._safe_float(w.get("humidity")),
                "pressure": self._safe_float(w.get("pressure")),
                "wind_speed": self._safe_float(w.get("wind_speed")),
                "wind_direction": self._safe_int(w.get("wind_direction"), 0),
                "rainfall": self._safe_int(w.get("rainfall"), 0),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })
        df = pd.DataFrame(rows)
        return df[OPENF1_WEATHER_SCHEMA.names]

    def ingest_session_overtakes(
        self,
        session_key: int,
        driver_map: Optional[Dict[int, Dict[str, str]]] = None,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Ingest on-track overtaking events."""
        raw_ov = self.client.fetch_overtakes(session_key=session_key)
        if not raw_ov:
            return pd.DataFrame(columns=OPENF1_OVERTAKES_SCHEMA.names)

        retrieved_at = self._now_iso()
        rows = []
        for ov in raw_ov:
            ing_num = int(ov.get("overtaking_driver_number", 0))
            en_num = int(ov.get("overtaken_driver_number", 0))

            ing_id = (
                driver_map[ing_num]["driver_id"]
                if driver_map and ing_num in driver_map
                else self.resolve_driver_identity(ing_num, year=year)
            )
            en_id = (
                driver_map[en_num]["driver_id"]
                if driver_map and en_num in driver_map
                else self.resolve_driver_identity(en_num, year=year)
            )

            rows.append({
                "session_key": int(ov.get("session_key", session_key)),
                "meeting_key": int(ov.get("meeting_key", 0)),
                "date": str(ov.get("date") or ""),
                "overtaking_driver_number": ing_num,
                "overtaking_driver_id": ing_id,
                "overtaken_driver_number": en_num,
                "overtaken_driver_id": en_id,
                "position": int(ov.get("position", 0)),
                "source_api": "openf1",
                "retrieved_at": retrieved_at,
            })
        df = pd.DataFrame(rows)
        return df[OPENF1_OVERTAKES_SCHEMA.names]

    def ingest_selective_car_data(
        self,
        session_key: int,
        driver_numbers: List[int],
        speed_min: Optional[int] = None,
        sample_step: int = 1,
        driver_map: Optional[Dict[int, Dict[str, str]]] = None,
        year: int = 2023,
    ) -> pd.DataFrame:
        """Selectively ingest high-frequency telemetry for specified driver numbers."""
        retrieved_at = self._now_iso()
        all_rows = []

        for d_num in driver_numbers:
            d_id = (
                driver_map[d_num]["driver_id"]
                if driver_map and d_num in driver_map
                else self.resolve_driver_identity(d_num, year=year)
            )
            raw = self.client.fetch_car_data(session_key=session_key, driver_number=d_num, speed_min=speed_min)
            if not raw:
                continue

            # Apply configurable subsampling
            sampled = raw[::max(1, sample_step)]
            for r in sampled:
                all_rows.append({
                    "session_key": int(r.get("session_key", session_key)),
                    "meeting_key": int(r.get("meeting_key", 0)),
                    "date": str(r.get("date") or ""),
                    "driver_number": d_num,
                    "driver_id": d_id,
                    "speed": int(r.get("speed", 0)),
                    "rpm": int(r.get("rpm", 0)),
                    "n_gear": int(r.get("n_gear", 0)),
                    "throttle": int(r.get("throttle", 0)),
                    "brake": int(r.get("brake", 0)),
                    "drs": int(r.get("drs", 0)),
                    "source_api": "openf1",
                    "retrieved_at": retrieved_at,
                })

        if not all_rows:
            return pd.DataFrame(columns=OPENF1_CAR_DATA_SCHEMA.names)
        df = pd.DataFrame(all_rows)
        return df[OPENF1_CAR_DATA_SCHEMA.names]

    # -----------------------------------------------------------------------
    # Comprehensive Session Ingestion & Persistence
    # -----------------------------------------------------------------------

    def execute_session_ingestion(
        self,
        session_key: int,
        meeting_key: Optional[int] = None,
        year: int = 2023,
        force: bool = False,
        include_telemetry_drivers: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        """Ingest and persist complete race intelligence datasets for a session."""
        session_dir = get_openf1_dir(self.base_dir) / str(year) / f"session_{session_key}"
        session_dir.mkdir(parents=True, exist_ok=True)

        counts: Dict[str, int] = {}

        # 1. Drivers & Driver Mapping
        driver_map = self.build_session_driver_map(session_key=session_key, year=year)
        df_drivers = self.ingest_session_drivers(session_key=session_key, year=year)
        p_d = session_dir / "openf1_drivers.parquet"
        save_parquet(df_drivers, p_d, schema=OPENF1_DRIVERS_SCHEMA)
        counts["drivers"] = len(df_drivers)

        # 2. Intervals
        p_inv = session_dir / "openf1_intervals.parquet"
        if force or not p_inv.exists():
            df_inv = self.ingest_session_intervals(session_key=session_key, driver_map=driver_map, year=year)
            save_parquet(df_inv, p_inv, schema=OPENF1_INTERVALS_SCHEMA)
            counts["intervals"] = len(df_inv)
        else:
            counts["intervals"] = len(load_parquet(p_inv))

        # 3. Tyre Stints
        p_st = session_dir / "openf1_stints.parquet"
        if force or not p_st.exists():
            df_st = self.ingest_session_stints(session_key=session_key, driver_map=driver_map, year=year)
            save_parquet(df_st, p_st, schema=OPENF1_STINTS_SCHEMA)
            counts["stints"] = len(df_st)
        else:
            counts["stints"] = len(load_parquet(p_st))

        # 4. Race Control
        p_rc = session_dir / "openf1_race_control.parquet"
        if force or not p_rc.exists():
            df_rc = self.ingest_session_race_control(session_key=session_key, driver_map=driver_map, year=year)
            save_parquet(df_rc, p_rc, schema=OPENF1_RACE_CONTROL_SCHEMA)
            counts["race_control"] = len(df_rc)
        else:
            counts["race_control"] = len(load_parquet(p_rc))

        # 5. Weather
        p_w = session_dir / "openf1_weather.parquet"
        if force or not p_w.exists():
            df_w = self.ingest_session_weather(session_key=session_key)
            save_parquet(df_w, p_w, schema=OPENF1_WEATHER_SCHEMA)
            counts["weather"] = len(df_w)
        else:
            counts["weather"] = len(load_parquet(p_w))

        # 6. Overtakes
        p_ov = session_dir / "openf1_overtakes.parquet"
        if force or not p_ov.exists():
            df_ov = self.ingest_session_overtakes(session_key=session_key, driver_map=driver_map, year=year)
            save_parquet(df_ov, p_ov, schema=OPENF1_OVERTAKES_SCHEMA)
            counts["overtakes"] = len(df_ov)
        else:
            counts["overtakes"] = len(load_parquet(p_ov))

        # 7. Optional Selective Telemetry
        if include_telemetry_drivers:
            df_telemetry = self.ingest_selective_car_data(
                session_key=session_key,
                driver_numbers=include_telemetry_drivers,
                sample_step=5,
                driver_map=driver_map,
                year=year,
            )
            p_tel = session_dir / "openf1_car_data_sampled.parquet"
            save_parquet(df_telemetry, p_tel, schema=OPENF1_CAR_DATA_SCHEMA)
            counts["car_data_sampled"] = len(df_telemetry)

        return counts
