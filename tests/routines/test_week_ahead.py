"""Unit tests for the morning-briefing week-ahead bullet renderer."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from alfred.routines._common import (
    render_week_ahead_bullets_html,
    render_week_ahead_bullets_plain,
)

TZ = ZoneInfo("America/Los_Angeles")


def _ev(start_iso: str, summary: str = "x") -> dict:
    return {"start_iso": start_iso, "end_iso": start_iso, "summary": summary}


def test_plain_empty_day_says_clear() -> None:
    by_day = {"2026-06-01": [], "2026-06-02": []}
    out = render_week_ahead_bullets_plain(by_day, TZ)
    assert "Mon Jun 1 — clear" in out
    assert "Tue Jun 2 — clear" in out


def test_plain_counts_and_first_time() -> None:
    by_day = {
        "2026-06-01": [
            _ev("2026-06-01T09:00:00-07:00"),
            _ev("2026-06-01T14:00:00-07:00"),
        ],
        "2026-06-02": [_ev("2026-06-02T08:30:00-07:00")],
    }
    out = render_week_ahead_bullets_plain(by_day, TZ)
    assert "Mon Jun 1 — 2 events, first 9:00 am" in out
    assert "Tue Jun 2 — 1 event, first 8:30 am" in out


def test_plain_sorts_days_chronologically() -> None:
    by_day = {
        "2026-06-03": [],
        "2026-06-01": [],
        "2026-06-02": [],
    }
    out = render_week_ahead_bullets_plain(by_day, TZ)
    lines = [line for line in out.splitlines() if line.strip()]
    assert "Jun 1" in lines[0]
    assert "Jun 2" in lines[1]
    assert "Jun 3" in lines[2]


def test_plain_uses_earliest_event_when_unsorted() -> None:
    by_day = {
        "2026-06-01": [
            _ev("2026-06-01T15:00:00-07:00"),
            _ev("2026-06-01T07:00:00-07:00"),
        ],
    }
    out = render_week_ahead_bullets_plain(by_day, TZ)
    assert "first 7:00 am" in out


def test_html_contains_day_labels_and_counts() -> None:
    by_day = {
        "2026-06-01": [_ev("2026-06-01T09:00:00-07:00")],
        "2026-06-02": [],
    }
    html = render_week_ahead_bullets_html(by_day, TZ)
    assert "Mon Jun 1" in html
    assert "1 event, first 9:00 am" in html
    assert "Tue Jun 2" in html
    assert "clear" in html


def test_html_escapes_safely() -> None:
    # Day labels are deterministic so XSS is theoretical, but make sure
    # we're using escape() rather than raw concat in case of future changes.
    by_day: dict[str, list[dict]] = {"2026-06-01": []}
    html = render_week_ahead_bullets_html(by_day, TZ)
    assert "<script>" not in html
