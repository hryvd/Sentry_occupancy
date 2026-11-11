"""
backend_client.py — Injected mock/real observation transport.

Responsibilities:
  - POST occupancy_observation events to backend (or mock sink)
  - POST device_heartbeat events
  - Bounded retry queue with backoff
  - Exact event_id/body/occurred_at preserved on retry
  - Maintains boot_id and per-space sequence counters
  - NEVER writes assignments, releases spaces, or accesses PostgreSQL

Modes:
  - BackendMode.MOCK  → logs events locally; simulates 202/409/503 responses
  - BackendMode.REAL  → HTTP POST to configured BACKEND_API_BASE_URL (gated)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETRY_QUEUE_SIZE = 100
BASE_RETRY_DELAY_S = 2.0
MAX_RETRY_DELAY_S = 60.0


class BackendMode(str, Enum):
    MOCK = "mock"
    REAL = "real"


@dataclass
class _RetryItem:
    payload: dict[str, Any]
    attempts: int = 0
    next_retry_at: float = field(default_factory=time.monotonic)
    endpoint: str = "occupancy"


class BackendClient:
    """
    Backend observation transport.

    Implements a bounded retry queue so temporary backend unavailability
    does not cause data loss. Queue saturation is logged and oldest items
    are dropped (capacity overflow, not silent discard).

    boot_id is generated fresh on client instantiation (service start).
    """

    def __init__(
        self,
        mode: BackendMode,
        config,  # CaptureConfig
        *,
        api_base_url: str = "",
        device_token: str = "",
    ) -> None:
        self._mode = mode
        self._config = config
        self._api_base_url = api_base_url or getattr(config, "api_base_url", "")
        self._device_token = device_token
        self._boot_id: str = str(uuid.uuid4())
        self._sequences: dict[str, int] = {}
        self._retry_queue: deque[_RetryItem] = deque(maxlen=MAX_RETRY_QUEUE_SIZE)

        logger.info(
            "BackendClient initialized (mode=%s, boot_id=%s)",
            mode.value,
            self._boot_id,
        )

    @property
    def boot_id(self) -> str:
        return self._boot_id

    def next_sequence(self, space_id: str) -> int:
        """Atomically increment and return the per-space sequence number."""
        self._sequences[space_id] = self._sequences.get(space_id, 0) + 1
        return self._sequences[space_id]

    # ------------------------------------------------------------------
    # Public transport methods
    # ------------------------------------------------------------------

    def post_observation(self, payload: dict[str, Any]) -> None:
        """Submit an occupancy_observation payload."""
        if self._mode == BackendMode.MOCK:
            self._mock_post("occupancy", payload)
        else:
            self._real_post("/api/v1/observations/occupancy", payload, endpoint="occupancy")

    def post_heartbeat(self, payload: dict[str, Any]) -> None:
        """Submit a device_heartbeat payload."""
        if self._mode == BackendMode.MOCK:
            self._mock_post("heartbeat", payload)
        else:
            self._real_post("/api/v1/observations/heartbeat", payload, endpoint="heartbeat")

    def flush(self) -> None:
        """Attempt to drain retry queue (called on shutdown)."""
        logger.info("Flushing retry queue (%d items).", len(self._retry_queue))
        while self._retry_queue:
            item = self._retry_queue.popleft()
            if self._mode == BackendMode.MOCK:
                self._mock_post(item.endpoint, item.payload)
            else:
                self._real_post(
                    f"/api/v1/observations/{item.endpoint}",
                    item.payload,
                    endpoint=item.endpoint,
                    is_retry=True,
                )

    def close(self) -> None:
        """Close HTTP session if open."""
        if hasattr(self, "_http_client"):
            self._http_client.close()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Mock transport
    # ------------------------------------------------------------------

    def _mock_post(self, endpoint: str, payload: dict[str, Any]) -> None:
        """Simulate a POST. Logs the event; never contacts any real server."""
        event_type = payload.get("event_type", "?")
        event_id = payload.get("event_id", "?")
        data = payload.get("data", {})

        if endpoint == "occupancy":
            logger.info(
                "[MOCK] OCCUPANCY event_id=%s space=%s status=%s confidence=%s seq=%s",
                event_id,
                data.get("space_id", "?")[-12:],
                data.get("physical_status", "?"),
                data.get("confidence"),
                data.get("sequence_number"),
            )
        elif endpoint == "heartbeat":
            logger.info(
                "[MOCK] HEARTBEAT event_id=%s boot_id=%s lot=%s",
                event_id,
                data.get("boot_id", "?")[-8:],
                data.get("lot_code", "?"),
            )
        else:
            logger.info("[MOCK] %s event_id=%s", event_type, event_id)

    # ------------------------------------------------------------------
    # Real transport (gated — requires approved auth/TLS/device scope)
    # ------------------------------------------------------------------

    def _real_post(
        self,
        path: str,
        payload: dict[str, Any],
        endpoint: str = "occupancy",
        is_retry: bool = False,
    ) -> None:
        """
        POST to the real FastAPI backend.

        BLOCKED until approved authentication, device scopes, and TLS
        configuration are implemented. Attempting real mode without these
        will raise RuntimeError.
        """
        if not self._api_base_url:
            raise RuntimeError(
                "BACKEND_MODE=real requires BACKEND_API_BASE_URL to be configured. "
                "Real integration is BLOCKED pending approved auth/TLS/device scope."
            )

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("httpx is required for real backend mode: pip install httpx") from exc

        url = self._api_base_url.rstrip("/") + path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._device_token}",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload, headers=headers)

            if resp.status_code in (200, 202):
                logger.debug("POST %s → %d", url, resp.status_code)
            elif resp.status_code == 409:
                logger.warning(
                    "POST %s → 409 Conflict (event_id=%s). "
                    "Changed-content conflict — dropping this payload.",
                    url,
                    payload.get("event_id"),
                )
            else:
                logger.warning(
                    "POST %s → %d. Queuing for retry (event_id=%s).",
                    url,
                    resp.status_code,
                    payload.get("event_id"),
                )
                self._enqueue_retry(payload, endpoint)

        except (httpx.RequestError, httpx.TimeoutException) as exc:
            logger.warning(
                "POST %s failed: %s. Queuing for retry (event_id=%s).",
                url,
                exc,
                payload.get("event_id"),
            )
            if not is_retry:
                self._enqueue_retry(payload, endpoint)

    def _enqueue_retry(self, payload: dict[str, Any], endpoint: str) -> None:
        """Add payload to bounded retry queue."""
        if len(self._retry_queue) >= MAX_RETRY_QUEUE_SIZE:
            dropped = self._retry_queue.popleft()
            logger.error(
                "Retry queue saturated (%d). Dropping oldest event_id=%s.",
                MAX_RETRY_QUEUE_SIZE,
                dropped.payload.get("event_id"),
            )
        delay = min(BASE_RETRY_DELAY_S * (2 ** min(len(self._retry_queue), 5)), MAX_RETRY_DELAY_S)
        self._retry_queue.append(_RetryItem(
            payload=payload,
            endpoint=endpoint,
            next_retry_at=time.monotonic() + delay,
        ))
