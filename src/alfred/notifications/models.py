"""Notification data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from alfred.core.constants import NotificationChannel


class Notification(BaseModel):
    """A notification to send to a family member."""

    recipient: str
    subject: str | None = None
    body: str
    priority: str = "normal"
    channel: NotificationChannel
    metadata: dict[str, Any] = {}


class NotificationResult(BaseModel):
    """Result of a notification delivery attempt."""

    success: bool
    channel: str
    error: str | None = None
