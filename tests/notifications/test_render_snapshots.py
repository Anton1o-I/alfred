"""Snapshot-style tests for tiered chore HTML and the Disappointed persona.

Strings are checked for distinguishing substrings, not byte-for-byte
equality, so legitimate style tweaks don't churn the tests. The intent
is to lock in: (a) tier prefixes land on the right rows,
(b) tier-2/3 rows get the red marker color, (c) roast_lines override the
fallback label per chore_id, (d) tier ordering (most-severe first), and
(e) the persona override shows up in the From metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from alfred.agents.tasks.renderers import (
    ShameTierTable,
    render_chores_html,
    render_chores_plain,
)
from alfred.core.persona import Persona, resolve_persona

TIERS: list[dict[str, Any]] = [
    {"tier": 1, "min_days": 3, "max_days": 6, "fallback_label": "{days}d pending",
     "row_prefix": "·"},
    {"tier": 2, "min_days": 7, "max_days": 13, "fallback_label": "{days}d overdue",
     "row_prefix": "!"},
    {"tier": 3, "min_days": 14, "max_days": None,
     "fallback_label": "{days}d — two weeks pending", "row_prefix": "!!"},
]


@dataclass
class _FakeChore:
    title: str
    assignee: str
    shame_after_days: int = 3
    id: str = "c1"


@dataclass
class _FakeStatus:
    overdue_days: int
    chore: _FakeChore


def _categorized(overdues: list[_FakeStatus]) -> dict[str, list[Any]]:
    return {
        "overdue": sorted(overdues, key=lambda s: -s.overdue_days),
        "due_today": [],
        "due_tomorrow": [],
        "later": [],
    }


def test_html_tier_1_uses_fallback_label() -> None:
    table = ShameTierTable(TIERS)
    statuses = [_FakeStatus(5, _FakeChore("Take out trash", "primary", id="t"))]
    html = render_chores_html(_categorized(statuses), None, shame_tiers=table)
    assert "5d pending" in html
    assert "Take out trash" in html
    # Tier 1 row prefix appears.
    assert "·" in html
    # Tier 1 uses the neutral marker color, not the tier-2 red.
    assert "#b91c1c" not in html


def test_html_tier_2_uses_firm_copy_and_red() -> None:
    table = ShameTierTable(TIERS)
    statuses = [_FakeStatus(10, _FakeChore("Take out trash", "primary", id="t"))]
    html = render_chores_html(_categorized(statuses), None, shame_tiers=table)
    assert "10d overdue" in html
    assert "#b91c1c" in html
    assert "!" in html


def test_html_tier_3_uses_severe_fallback() -> None:
    table = ShameTierTable(TIERS)
    statuses = [_FakeStatus(21, _FakeChore("Take out trash", "primary", id="t"))]
    html = render_chores_html(_categorized(statuses), None, shame_tiers=table)
    assert "21d — two weeks pending" in html
    assert "!!" in html


def test_html_no_section_headers() -> None:
    """Section headers were removed — roast + prefix + color carry severity."""
    table = ShameTierTable(TIERS)
    statuses = [
        _FakeStatus(5, _FakeChore("low", "primary", id="a")),
        _FakeStatus(10, _FakeChore("mid", "primary", id="b")),
        _FakeStatus(21, _FakeChore("severe", "primary", id="c")),
    ]
    html = render_chores_html(_categorized(statuses), None, shame_tiers=table)
    plain = render_chores_plain(_categorized(statuses), None, shame_tiers=table)
    # No bucket banners.
    for banner in ("OVERDUE", "Long overdue", "Still pending", "<h3"):
        assert banner not in html
        assert banner not in plain


def test_tier_ordering_most_severe_first() -> None:
    table = ShameTierTable(TIERS)
    statuses = [
        _FakeStatus(5, _FakeChore("low_title", "primary", id="a")),
        _FakeStatus(10, _FakeChore("mid_title", "primary", id="b")),
        _FakeStatus(21, _FakeChore("severe_title", "primary", id="c")),
    ]
    html = render_chores_html(_categorized(statuses), None, shame_tiers=table)
    # Tier 3 chore appears before tier 2, which appears before tier 1.
    assert html.index("severe_title") < html.index("mid_title")
    assert html.index("mid_title") < html.index("low_title")


def test_roast_lines_override_fallback_html() -> None:
    table = ShameTierTable(TIERS)
    statuses = [_FakeStatus(10, _FakeChore("Take out trash", "primary", id="trash"))]
    roast = "The kitchen has begun composing its own smell."
    html = render_chores_html(
        _categorized(statuses), None, shame_tiers=table, roast_lines={"trash": roast}
    )
    assert roast in html
    # Static fallback should NOT appear when roast is present.
    assert "10d overdue" not in html


def test_roast_lines_override_fallback_plain() -> None:
    table = ShameTierTable(TIERS)
    statuses = [_FakeStatus(10, _FakeChore("Take out trash", "primary", id="trash"))]
    roast = "The kitchen has begun composing its own smell."
    plain = render_chores_plain(
        _categorized(statuses), None, shame_tiers=table, roast_lines={"trash": roast}
    )
    assert roast in plain
    assert "10d overdue" not in plain


def test_missing_roast_falls_back_to_label() -> None:
    table = ShameTierTable(TIERS)
    statuses = [
        _FakeStatus(10, _FakeChore("Has roast", "primary", id="a")),
        _FakeStatus(8, _FakeChore("No roast", "primary", id="b")),
    ]
    html = render_chores_html(
        _categorized(statuses),
        None,
        shame_tiers=table,
        roast_lines={"a": "roast for a"},
    )
    assert "roast for a" in html
    # Chore b has no roast → fallback label appears.
    assert "8d overdue" in html


def test_plain_includes_tier_prefix() -> None:
    table = ShameTierTable(TIERS)
    statuses = [_FakeStatus(10, _FakeChore("Take out trash", "primary", id="t"))]
    plain = render_chores_plain(_categorized(statuses), None, shame_tiers=table)
    assert "!" in plain
    assert "10d overdue" in plain


def test_legacy_render_unchanged_when_no_tiers() -> None:
    """No shame_tiers arg → identical legacy ⚠ output (backward-compat)."""
    statuses = [_FakeStatus(10, _FakeChore("Take out trash", "primary", id="t"))]
    html = render_chores_html(_categorized(statuses), None)
    assert "10d late" in html  # legacy label
    assert ">Overdue<" in html


# ── Persona resolution ─────────────────────────────────────────────────────


def test_resolve_persona_returns_yaml_values() -> None:
    cfg = {
        "tasks-shame": {
            "display_name": "Alfred · Disappointed",
            "tagline": "AI chore tracker · overdue note",
        }
    }
    name, tagline = resolve_persona(Persona.TASKS_SHAME, cfg)
    assert name == "Alfred · Disappointed"
    assert "overdue" in tagline


def test_resolve_persona_falls_back_to_empty() -> None:
    name, tagline = resolve_persona(Persona.TASKS_SHAME, {})
    assert name == ""
    assert tagline == ""


@pytest.mark.asyncio
async def test_send_to_family_uses_persona_override(monkeypatch, tmp_path) -> None:
    """`persona_override` swaps the From-name on the outbound metadata."""
    from alfred.core.config import (
        NotificationConfig,
        RecipientConfig,
    )
    from alfred.notifications.service import NotificationService
    from scaffold.audit.logger import AuditLogger
    from scaffold.core.constants import NotificationChannel
    from scaffold.storage.database import Database

    db = Database(str(tmp_path / "t.db"))
    await db.initialize()
    audit = AuditLogger(db)

    cfg = NotificationConfig(
        email_enabled=True,
        recipients=[
            RecipientConfig(
                user_id="primary",
                preferred_channel=NotificationChannel.EMAIL,
                email="test@example.com",
            )
        ],
        personas={
            "tasks-shame": {
                "display_name": "Alfred · Disappointed",
                "tagline": "overdue chores note",
            }
        },
    )

    sent: list[Any] = []

    class _StubChannel:
        async def send(self, notification):  # type: ignore[no-untyped-def]
            sent.append(notification)
            from alfred.notifications.models import NotificationResult
            return NotificationResult(success=True, channel="email")

    svc = NotificationService(
        channels={NotificationChannel.EMAIL: _StubChannel()},
        config=cfg,
        audit_logger=audit,
        db=db,
    )
    await svc.send_to_family(
        body="b",
        subject="s",
        html_body="<p>x</p>",
        from_name="Alfred · Scheduler",  # baseline; should be overridden
        persona_override=Persona.TASKS_SHAME,
    )
    await db.close()

    assert len(sent) == 1
    assert sent[0].metadata == {"from_name": "Alfred · Disappointed"}
