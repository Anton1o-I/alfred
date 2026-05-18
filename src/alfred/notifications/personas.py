"""Persona identities for outbound mail.

A persona bundles a From-name and a signature tagline. Most outbound mail
resolves a persona implicitly from the originating agent
(`NotificationConfig.email_from_names_by_agent` /
`email_taglines_by_agent`). Some flows need to override that — e.g. the
daily briefing splits its tier-2+ shame chores into a separate
"Alfred · Disappointed" note. The `Persona` enum is the single switch
point so callers don't pass raw strings around.
"""

from __future__ import annotations

from enum import StrEnum


class Persona(StrEnum):
    """Distinct outbound mail identities.

    The string value is the key used in `NotificationConfig.personas`
    (config/notifications.yaml under `personas:`). Adding a new persona
    means adding the enum member here AND a matching block in YAML.
    """

    TASKS_SHAME = "tasks-shame"


def resolve_persona(
    persona: Persona,
    personas_config: dict[str, dict[str, str]],
) -> tuple[str, str]:
    """Return (display_name, tagline) for a persona.

    Falls back to ("", "") if the persona is not declared in YAML — the
    signature renderer treats both-empty as "no signature".
    """
    block = personas_config.get(persona.value, {})
    return block.get("display_name", ""), block.get("tagline", "")
