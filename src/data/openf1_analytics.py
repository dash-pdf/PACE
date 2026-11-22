"""PACE: OpenF1 Race Intelligence Analytics & Feature Engineering Engine.

Processes OpenF1 datasets (intervals, stints, race control, weather, overtakes, telemetry)
to extract validated race intelligence features: traffic exposure, undercut/overcut dynamics,
neutralisation event periods, weather gradients, and overtaking profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class NeutralisationEvent:
    """Reconstructed Safety Car, VSC, or Red Flag period."""
    event_type: str                         # 'SAFETY_CAR', 'VSC', 'RED_FLAG'
    start_date: str
    end_date: Optional[str]
    start_lap: Optional[int]
    end_lap: Optional[int]
    duration_seconds: Optional[float]
    laps_affected: int
    message: str


@dataclass
class DriverStintProfile:
    """Tyre compound and stint usage summary for a single driver."""
    driver_id: str
    driver_number: int
    stint_number: int
    compound: str
    lap_start: int
    lap_end: int
    stint_length: int
    tyre_age_at_start: int
    pitted_under_neutralisation: bool = False


@dataclass
class DriverRaceIntelligenceSummary:
    """Comprehensive race progression and performance intelligence for a single driver."""
    driver_id: str
    driver_number: int
    constructor_id: str
    total_laps: int
    stints_count: int
    compounds_used: List[str]
    traffic_exposure_pct: float             # % of intervals with gap_ahead <= 1.0s
    mean_gap_to_leader: float
    min_gap_to_leader: float
    position_volatility: float              # Standard deviation of running position
    overtakes_made: int
    times_overtaken: int
    net_overtakes: int
    pit_stops_under_sc_vsc: int


@dataclass
class SessionWeatherProfile:
    """Environmental condition analysis across a session."""
    session_key: int
    mean_track_temp: float
    min_track_temp: float
    max_track_temp: float
    track_temp_delta: float
    mean_air_temp: float
    mean_humidity: float
    mean_wind_speed: float
    rainfall_detected: bool
    wet_onset_timestamps: List[str]


# ---------------------------------------------------------------------------
# Race Intelligence Engine
# ---------------------------------------------------------------------------

class OpenF1RaceIntelligence:
    """Calculates validated race intelligence features from OpenF1 datasets."""

    def __init__(
        self,
        df_intervals: Optional[pd.DataFrame] = None,
        df_stints: Optional[pd.DataFrame] = None,
        df_race_control: Optional[pd.DataFrame] = None,
        df_weather: Optional[pd.DataFrame] = None,
        df_overtakes: Optional[pd.DataFrame] = None,
        df_drivers: Optional[pd.DataFrame] = None,
    ) -> None:
        self.df_intervals = df_intervals if df_intervals is not None else pd.DataFrame()
        self.df_stints = df_stints if df_stints is not None else pd.DataFrame()
        self.df_rc = df_race_control if df_race_control is not None else pd.DataFrame()
        self.df_weather = df_weather if df_weather is not None else pd.DataFrame()
        self.df_overtakes = df_overtakes if df_overtakes is not None else pd.DataFrame()
        self.df_drivers = df_drivers if df_drivers is not None else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Race Control & Neutralisation Reconstruction
    # -----------------------------------------------------------------------

    def extract_neutralisation_periods(self) -> List[NeutralisationEvent]:
        """Reconstruct continuous Safety Car, VSC, and Red Flag periods from messages."""
        if self.df_rc.empty:
            return []

        df_sorted = self.df_rc.sort_values(by="date").reset_index(drop=True)
        events: List[NeutralisationEvent] = []

        active_event: Optional[Dict[str, Any]] = None

        for _, row in df_sorted.iterrows():
            msg = str(row.get("message") or "").upper()
            cat = str(row.get("category") or "").upper()
            flag = str(row.get("flag") or "").upper()
            dt_str = str(row.get("date") or "")
            lap = row.get("lap_number")
            lap_num = int(lap) if pd.notna(lap) else None

            # Detect start of Virtual Safety Car (check before physical SC to avoid substring overlap)
            if "VIRTUAL SAFETY CAR DEPLOYED" in msg or "VSC DEPLOYED" in msg or (cat == "SAFETYCAR" and "VIRTUAL" in msg):
                if active_event:
                    self._finalize_event(active_event, dt_str, lap_num, events)
                active_event = {
                    "type": "VSC",
                    "start_date": dt_str,
                    "start_lap": lap_num,
                    "message": row.get("message"),
                }

            # Detect start of Physical Safety Car
            elif ("SAFETY CAR DEPLOYED" in msg or (cat == "SAFETYCAR" and "DEPLOYED" in msg)) and "VIRTUAL" not in msg:
                if active_event:
                    # Close existing
                    self._finalize_event(active_event, dt_str, lap_num, events)
                active_event = {
                    "type": "SAFETY_CAR",
                    "start_date": dt_str,
                    "start_lap": lap_num,
                    "message": row.get("message"),
                }

            # Detect start of Red Flag
            elif flag == "RED" or "RED FLAG" in msg:
                if active_event:
                    self._finalize_event(active_event, dt_str, lap_num, events)
                active_event = {
                    "type": "RED_FLAG",
                    "start_date": dt_str,
                    "start_lap": lap_num,
                    "message": row.get("message"),
                }

            # Detect neutralisation end
            elif active_event and (
                "SAFETY CAR IN THIS LAP" in msg
                or "TRACK CLEAR" in msg
                or "VSC ENDING" in msg
                or "GREEN FLAG" in msg
                or flag == "GREEN"
                or flag == "CLEAR"
            ):
                self._finalize_event(active_event, dt_str, lap_num, events)
                active_event = None

        if active_event:
            self._finalize_event(active_event, None, None, events)

        return events

    def _finalize_event(
        self,
        event_dict: Dict[str, Any],
        end_date: Optional[str],
        end_lap: Optional[int],
        events_list: List[NeutralisationEvent],
    ) -> None:
        start_date = event_dict["start_date"]
        duration_s = None
        if start_date and end_date:
            try:
                t0 = pd.to_datetime(start_date)
                t1 = pd.to_datetime(end_date)
                duration_s = max(0.0, float((t1 - t0).total_seconds()))
            except Exception:
                pass

        s_lap = event_dict.get("start_lap")
        e_lap = end_lap
        laps_aff = max(1, (e_lap - s_lap + 1)) if (s_lap and e_lap and e_lap >= s_lap) else 1

        events_list.append(NeutralisationEvent(
            event_type=event_dict["type"],
            start_date=start_date,
            end_date=end_date,
            start_lap=s_lap,
            end_lap=e_lap,
            duration_seconds=duration_s,
            laps_affected=laps_aff,
            message=str(event_dict.get("message") or ""),
        ))

    # -----------------------------------------------------------------------
    # Tyre & Pit Strategy Intelligence
    # -----------------------------------------------------------------------

    def analyze_stints(self, neutralisations: Optional[List[NeutralisationEvent]] = None) -> List[DriverStintProfile]:
        """Analyze tyre stint lengths, compounds, and pit stops under neutralisation."""
        if self.df_stints.empty:
            return []

        sc_periods = neutralisations if neutralisations is not None else self.extract_neutralisation_periods()
        sc_laps: Set[int] = set()
        for ev in sc_periods:
            if ev.start_lap and ev.end_lap:
                for l in range(ev.start_lap, ev.end_lap + 1):
                    sc_laps.add(l)

        profiles: List[DriverStintProfile] = []
        for _, row in self.df_stints.iterrows():
            d_id = str(row.get("driver_id") or "")
            num = int(row.get("driver_number", 0))
            st_num = int(row.get("stint_number", 0))
            compound = str(row.get("compound") or "UNKNOWN")
            l_start = int(row.get("lap_start", 0)) if pd.notna(row.get("lap_start")) else 0
            l_end = int(row.get("lap_end", 0)) if pd.notna(row.get("lap_end")) else 0
            length = int(row.get("stint_length", 0))
            age = int(row.get("tyre_age_at_start", 0))

            # Did the pit stop (stint start) occur under neutralisation?
            pitted_sc = bool(l_start in sc_laps) if sc_laps and l_start > 1 else False

            profiles.append(DriverStintProfile(
                driver_id=d_id,
                driver_number=num,
                stint_number=st_num,
                compound=compound,
                lap_start=l_start,
                lap_end=l_end,
                stint_length=length,
                tyre_age_at_start=age,
                pitted_under_neutralisation=pitted_sc,
            ))

        return profiles

    # -----------------------------------------------------------------------
    # Traffic Exposure & Interval Intelligence
    # -----------------------------------------------------------------------

    def calculate_traffic_exposure(self, drs_threshold_s: float = 1.0) -> Dict[str, float]:
        """Compute traffic exposure percentage (% of intervals within DRS range <= 1.0s of car ahead)."""
        if self.df_intervals.empty or "driver_id" not in self.df_intervals.columns:
            return {}

        result: Dict[str, float] = {}
        for d_id, group in self.df_intervals.groupby("driver_id"):
            valid_intervals = group["interval"].dropna()
            if valid_intervals.empty:
                result[str(d_id)] = 0.0
                continue

            in_traffic = (valid_intervals <= drs_threshold_s) & (valid_intervals >= 0.0)
            pct = float((in_traffic.sum() / len(valid_intervals)) * 100.0)
            result[str(d_id)] = round(pct, 2)

        return result

    # -----------------------------------------------------------------------
    # Weather Profile Intelligence
    # -----------------------------------------------------------------------

    def analyze_session_weather(self, session_key: int = 0) -> SessionWeatherProfile:
        """Compute summary statistics for track temperature, ambient weather, and rainfall."""
        if self.df_weather.empty:
            return SessionWeatherProfile(
                session_key=session_key,
                mean_track_temp=0.0,
                min_track_temp=0.0,
                max_track_temp=0.0,
                track_temp_delta=0.0,
                mean_air_temp=0.0,
                mean_humidity=0.0,
                mean_wind_speed=0.0,
                rainfall_detected=False,
                wet_onset_timestamps=[],
            )

        track_temps = self.df_weather["track_temperature"].dropna()
        air_temps = self.df_weather["air_temperature"].dropna()
        humidity = self.df_weather["humidity"].dropna()
        wind_speed = self.df_weather["wind_speed"].dropna()
        rainfall = self.df_weather["rainfall"].dropna()

        mean_tr = float(track_temps.mean()) if not track_temps.empty else 0.0
        min_tr = float(track_temps.min()) if not track_temps.empty else 0.0
        max_tr = float(track_temps.max()) if not track_temps.empty else 0.0
        delta_tr = max_tr - min_tr

        mean_air = float(air_temps.mean()) if not air_temps.empty else 0.0
        mean_hum = float(humidity.mean()) if not humidity.empty else 0.0
        mean_wind = float(wind_speed.mean()) if not wind_speed.empty else 0.0

        rain_detected = bool((rainfall > 0).any()) if not rainfall.empty else False
        rain_rows = self.df_weather[self.df_weather["rainfall"] > 0]
        wet_stamps = [str(d) for d in rain_rows["date"].tolist()]

        s_key = int(self.df_weather["session_key"].iloc[0]) if "session_key" in self.df_weather.columns else session_key

        return SessionWeatherProfile(
            session_key=s_key,
            mean_track_temp=round(mean_tr, 2),
            min_track_temp=round(min_tr, 2),
            max_track_temp=round(max_tr, 2),
            track_temp_delta=round(delta_tr, 2),
            mean_air_temp=round(mean_air, 2),
            mean_humidity=round(mean_hum, 2),
            mean_wind_speed=round(mean_wind, 2),
            rainfall_detected=rain_detected,
            wet_onset_timestamps=wet_stamps[:5],
        )

    # -----------------------------------------------------------------------
    # Comprehensive Driver Summary
    # -----------------------------------------------------------------------

    def compile_driver_race_intelligence(self) -> pd.DataFrame:
        """Compile complete driver-level race progression and intelligence summary."""
        neutralisations = self.extract_neutralisation_periods()
        stints = self.analyze_stints(neutralisations=neutralisations)
        traffic_map = self.calculate_traffic_exposure()

        # Overtakes mapping
        overtakes_made: Dict[str, int] = {}
        times_overtaken: Dict[str, int] = {}

        if not self.df_overtakes.empty and "overtaking_driver_id" in self.df_overtakes.columns:
            for ing in self.df_overtakes["overtaking_driver_id"].dropna():
                overtakes_made[str(ing)] = overtakes_made.get(str(ing), 0) + 1
            for en in self.df_overtakes["overtaken_driver_id"].dropna():
                times_overtaken[str(en)] = times_overtaken.get(str(en), 0) + 1

        # Stints aggregation by driver
        stints_by_driver: Dict[str, List[DriverStintProfile]] = {}
        for st in stints:
            stints_by_driver.setdefault(st.driver_id, []).append(st)

        # Drivers metadata
        d_meta: Dict[str, Dict[str, Any]] = {}
        if not self.df_drivers.empty and "driver_id" in self.df_drivers.columns:
            for _, r in self.df_drivers.iterrows():
                d_id = str(r["driver_id"])
                d_meta[d_id] = {
                    "driver_number": int(r.get("driver_number", 0)),
                    "constructor_id": str(r.get("constructor_id", "unknown")),
                }

        # Build summaries
        summaries: List[Dict[str, Any]] = []
        all_drivers = set(traffic_map.keys()).union(set(stints_by_driver.keys())).union(set(d_meta.keys()))

        for d_id in sorted(list(all_drivers)):
            if not d_id or d_id == "unknown":
                continue

            meta = d_meta.get(d_id, {"driver_number": 0, "constructor_id": "unknown"})
            d_stints = stints_by_driver.get(d_id, [])
            compounds = [s.compound for s in d_stints]
            sc_pits = sum(1 for s in d_stints if s.pitted_under_neutralisation)

            # Total laps completed from stints
            total_laps = max([s.lap_end for s in d_stints]) if d_stints else 0

            # Gap to leader statistics
            mean_gap = 0.0
            min_gap = 0.0
            pos_vol = 0.0
            if not self.df_intervals.empty and "driver_id" in self.df_intervals.columns:
                sub_inv = self.df_intervals[self.df_intervals["driver_id"] == d_id]
                if not sub_inv.empty:
                    valid_gaps = sub_inv["gap_to_leader"].dropna()
                    if not valid_gaps.empty:
                        mean_gap = float(valid_gaps.mean())
                        min_gap = float(valid_gaps.min())

            ot_made = overtakes_made.get(d_id, 0)
            ot_taken = times_overtaken.get(d_id, 0)

            summaries.append({
                "driver_id": d_id,
                "driver_number": meta["driver_number"],
                "constructor_id": meta["constructor_id"],
                "total_laps": total_laps,
                "stints_count": len(d_stints),
                "compounds_used": ", ".join(compounds) if compounds else "NONE",
                "traffic_exposure_pct": traffic_map.get(d_id, 0.0),
                "mean_gap_to_leader_s": round(mean_gap, 2),
                "min_gap_to_leader_s": round(min_gap, 2),
                "overtakes_made": ot_made,
                "times_overtaken": ot_taken,
                "net_overtakes": ot_made - ot_taken,
                "pit_stops_under_sc": sc_pits,
            })

        df = pd.DataFrame(summaries)
        if df.empty:
            return pd.DataFrame(columns=[
                "driver_id", "driver_number", "constructor_id", "total_laps",
                "stints_count", "compounds_used", "traffic_exposure_pct",
                "mean_gap_to_leader_s", "min_gap_to_leader_s", "overtakes_made",
                "times_overtaken", "net_overtakes", "pit_stops_under_sc"
            ])
        return df.sort_values(by="mean_gap_to_leader_s").reset_index(drop=True)
