"""Load repo-root ``config.yaml``.

Secrets stay in the process environment or ``.env``. ``CLOUD_URL`` and
``EDGE_ENABLE_UDP`` override the file so Docker Compose can rewire the edge
without editing YAML.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from shared.paths import REPO_ROOT

# Load before reading env overrides so a local ``.env`` applies. Compose-set
# variables already exist and are not overwritten.
load_dotenv(REPO_ROOT / ".env")


class EdgeSettings(BaseModel):
    """UDP ingest, windowing, and cloud POST settings for the edge process."""

    http_host: str = "0.0.0.0"
    http_port: int = 8000
    udp_host: str = "0.0.0.0"
    udp_port: int = 9000
    udp_pcm_port: int = 9001
    enable_udp: bool = True
    cloud_url: str = "http://127.0.0.1:8001"
    cloud_post_timeout_s: float = 8.0
    max_nodes: int = 5
    max_history: int = 220
    live_s: float = 2.5
    rate_s: float = 1.5
    escalate_min_confidence: float = 0.50
    window_s: float = 2.0
    hop_s: float = 1.0
    sensor_hz: float = 50.0
    pcm_rate_hz: float = 16000.0
    pcm_ring_s: float = 4.0


class ServerSettings(BaseModel):
    """HTTP and escalation-ladder settings for the cloud process."""

    http_host: str = "0.0.0.0"
    http_port: int = 8001
    senior_wait_s: float = 60.0
    family_wait_s: float = 60.0
    careline_at_s: float = 180.0
    tick_s: float = 0.5
    telegram_timeout_s: float = 8.0


class AlertSettings(BaseModel):
    """Shared cooldown after a gated fall is posted."""

    cooldown_s: float = 3.0


class Settings(BaseModel):
    """Root settings object matching ``config.yaml``."""

    edge: EdgeSettings = Field(default_factory=EdgeSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    alert: AlertSettings = Field(default_factory=AlertSettings)


def config_path() -> Path:
    """Return the YAML path from ``OPOYO_CONFIG`` or the repo root.

    Returns:
        Absolute or relative path to ``config.yaml``.
    """
    raw = os.environ.get("OPOYO_CONFIG")
    if raw:
        return Path(raw)
    return REPO_ROOT / "config.yaml"


def _as_bool(raw: str) -> bool:
    """Parse a loose boolean from an environment string."""
    return raw.strip().lower() not in {"0", "false", "no", ""}


def load_settings(path: Path | None = None) -> Settings:
    """Parse YAML and apply environment overrides.

    Args:
        path: Optional file to load. Defaults to :func:`config_path`. Missing
            files yield built-in defaults.

    Returns:
        Validated settings. ``CLOUD_URL`` and ``EDGE_ENABLE_UDP`` win over YAML
        when set, which is how Compose points the edge at ``http://server:8001``.
    """
    file = path or config_path()
    data: dict = {}
    if file.is_file():
        loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    settings = Settings.model_validate(data)
    cloud = os.environ.get("CLOUD_URL")
    if cloud:
        settings.edge.cloud_url = cloud.rstrip("/")
    else:
        settings.edge.cloud_url = settings.edge.cloud_url.rstrip("/")
    udp = os.environ.get("EDGE_ENABLE_UDP")
    if udp is not None:
        settings.edge.enable_udp = _as_bool(udp)
    return settings


CFG = load_settings()
