from __future__ import annotations

import pytest

from edge.app import create_app as create_edge
from edge.app import default_edge_config
from server.app import create_app as create_cloud
from shared.env import require_bool, require_env, require_float


def test_require_env_missing(monkeypatch):
    monkeypatch.delenv("EDGE_CLOUD_URL", raising=False)
    with pytest.raises(RuntimeError, match="EDGE_CLOUD_URL"):
        require_env("EDGE_CLOUD_URL")


def test_require_env_blank_fails(monkeypatch):
    monkeypatch.setenv("EDGE_CLOUD_URL", "   ")
    with pytest.raises(RuntimeError, match="EDGE_CLOUD_URL"):
        require_env("EDGE_CLOUD_URL")


def test_require_float_invalid(monkeypatch):
    monkeypatch.setenv("EDGE_ESCALATE_MIN_CONFIDENCE", "high")
    with pytest.raises(RuntimeError, match="EDGE_ESCALATE_MIN_CONFIDENCE"):
        require_float("EDGE_ESCALATE_MIN_CONFIDENCE")


def test_require_bool_invalid(monkeypatch):
    monkeypatch.setenv("EDGE_ENABLE_UDP", "maybe")
    with pytest.raises(RuntimeError, match="EDGE_ENABLE_UDP"):
        require_bool("EDGE_ENABLE_UDP")


def test_default_edge_config_requires_all(monkeypatch):
    monkeypatch.delenv("EDGE_CLOUD_URL", raising=False)
    monkeypatch.delenv("EDGE_ESCALATE_MIN_CONFIDENCE", raising=False)
    with pytest.raises(RuntimeError, match="missing required environment variable"):
        default_edge_config()


def test_live_edge_app_fails_without_env(monkeypatch):
    monkeypatch.delenv("EDGE_CLOUD_URL", raising=False)
    monkeypatch.delenv("EDGE_ESCALATE_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("EDGE_ENABLE_UDP", raising=False)
    with pytest.raises(RuntimeError, match="missing required environment variable"):
        create_edge()


def test_live_cloud_app_fails_without_telegram(monkeypatch):
    monkeypatch.setattr("server.app.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID_NEXT_OF_KIN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID_SECONDARY", raising=False)
    monkeypatch.delenv("SENIOR_PHONE", raising=False)
    with pytest.raises(RuntimeError, match="missing required environment variable"):
        create_cloud()


def test_default_edge_config_reads_env(monkeypatch):
    monkeypatch.setenv("EDGE_CLOUD_URL", "http://cloud:8001")
    monkeypatch.setenv("EDGE_ESCALATE_MIN_CONFIDENCE", "0.91")
    cfg = default_edge_config()
    assert cfg.cloud_url == "http://cloud:8001"
    assert cfg.escalate_min_confidence == 0.91
