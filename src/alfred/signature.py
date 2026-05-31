"""Render a small, agent-appropriate signature for outbound email.

Plain-text and HTML variants are produced from the same persona + tagline
pair so they read consistently in any mail client. Returns empty strings
when no signature is configured for the agent.
"""

from __future__ import annotations

from html import escape

_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def render_signature(persona: str, tagline: str) -> tuple[str, str]:
    """Return (plain, html) signature snippets. Empty if both inputs are empty."""
    if not persona and not tagline:
        return "", ""

    # NOTE: no leading '—' or '--' line and no border-top divider on the
    # HTML side. Both are Gmail-mobile signature-detection triggers that
    # cause the "Show trimmed content" heuristic to collapse the body
    # above. We rely on extra top margin + smaller, dim text for the
    # visual signature break instead.
    plain_lines: list[str] = []
    if persona:
        plain_lines.append(persona)
    if tagline:
        plain_lines.append(tagline)
    plain = "\n".join(plain_lines)

    container = (
        "margin: 40px 0 0; "
        f"font-family: {_FONT_STACK}; "
        "font-size: 12px; line-height: 1.45; color: #86868b;"
    )
    name_style = "font-weight: 600; color: #6e6e73;"
    parts = [f'<div style="{container}">']
    if persona:
        parts.append(f'<div style="{name_style}">{escape(persona)}</div>')
    if tagline:
        parts.append(f"<div>{escape(tagline)}</div>")
    parts.append("</div>")
    html = "".join(parts)
    return plain, html


def append_to_body(body: str, signature_plain: str) -> str:
    """Append a plain-text signature with a blank-line separator."""
    if not signature_plain:
        return body
    return f"{body.rstrip()}\n\n{signature_plain}\n"


def append_to_html(html_body: str, signature_html: str) -> str:
    """Append the HTML signature before </body>; fall back to concatenation."""
    if not signature_html:
        return html_body
    end_idx = html_body.rfind("</body>")
    if end_idx != -1:
        return html_body[:end_idx] + signature_html + html_body[end_idx:]
    return html_body + signature_html
