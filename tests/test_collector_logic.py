"""Tests for collector helper functions."""

import pandas as pd
from src.data.fastf1_collector import parse_time_to_ms, classify_status


def test_parse_time_to_ms():
    """Verify conversion of various time formats to milliseconds."""
    # Milliseconds float/int
    assert parse_time_to_ms(54321.0) == 54321.0

    # Timedelta
    td = pd.Timedelta("90.123s")
    assert parse_time_to_ms(td) == 90123.0

    # mm:ss.sss string
    assert parse_time_to_ms("1:23.456") == 83456.0

    # ss.sss string
    assert parse_time_to_ms("25.885") == 25885.0

    # Null / empty
    assert parse_time_to_ms(None) is None
    assert parse_time_to_ms("") is None
    assert parse_time_to_ms("nan") is None


def test_classify_status():
    """Verify classification of finishing statuses."""
    # Finished
    is_class, is_dnf, is_dns, is_dsq = classify_status("Finished", "1")
    assert is_class is True
    assert is_dnf is False
    assert is_dns is False
    assert is_dsq is False

    # Lapped finisher (+1 Lap)
    is_class, is_dnf, is_dns, is_dsq = classify_status("+1 Lap", "12")
    assert is_class is True
    assert is_dnf is False

    # Mechanical retirement
    is_class, is_dnf, is_dns, is_dsq = classify_status("Engine", "R")
    assert is_class is False
    assert is_dnf is True

    # Collision retirement
    is_class, is_dnf, is_dns, is_dsq = classify_status("Collision damage", "R")
    assert is_class is False
    assert is_dnf is True

    # Disqualification
    is_class, is_dnf, is_dns, is_dsq = classify_status("Disqualified", "D")
    assert is_class is False
    assert is_dsq is True

    # Did not start
    is_class, is_dnf, is_dns, is_dsq = classify_status("Did not start", "W")
    assert is_class is False
    assert is_dns is True
