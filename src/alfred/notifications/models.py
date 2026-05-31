"""Notification data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from scaffold.core.constants import NotificationChannel


class Notification(BaseModel):
    """A notification to send to a family member."""

    recipient: str
    subject: str | None = None
    body: str
    html_body: str | None = None  # When set, email channel sends multipart/alternative
    priority: str = "normal"
    channel: NotificationChannel
    metadata: dict[str, Any] = {}


class NotificationResult(BaseModel):
    """Result of a notification delivery attempt."""

    success: bool
    channel: str
    error: str | None = None
    message_id: str | None = None  # RFC 822 Message-Id of the sent message (email)
