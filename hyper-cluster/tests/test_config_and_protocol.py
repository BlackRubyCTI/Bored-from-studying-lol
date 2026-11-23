"""Configuration, logging and protocol unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import Settings, parse_dotenv
from core.logging_setup import get_logger, log_context, setup_logging
from core.protocol import ChatMessage, ChatRequest, RequestMode, WorkerHello, WorkerStatus


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_parse_dotenv_handles_quotes_comments_and_export() -> None:
    raw = """
# a comment
export HYPER_HEAD_PORT=9000
HYPER_API_KEY="quoted-value"
HYPER_LOG_JSON=0
HYPER_CORS_ORIGINS=["http://a", "http://b"]
HYPER_EMPTY=
"""
    parsed = parse_dotenv(raw)
    assert parsed["HYPER_HEAD_PORT"] == "9000"
    assert parsed["HYPER_API_KEY"] == "quoted-value"
    assert parsed["HYPER_LOG_JSON"] == "0"
    assert json.loads(parsed["HYPER_CORS_ORIGINS"]) == ["http://a", "http://b"]
    assert parsed["HYPER_EMPTY"] == ""


def test_settings_from_env_reads_prefixed_variables() -> None:
    env = {
        "HYPER_CLUSTER_NAME": "test-cluster",
        "HYPER_HEAD_PORT": "9999",
        "HYPER_HOLD_ENABLED": "false",
        "HYPER_COUNCIL_SIZE": "5",
        "HYPER_SPECULATIVE_EDGE_PROVIDERS": "groq, cerebras",
    }
    settings = Settings.from_env(env)
    assert settings.cluster_name == "test-cluster"
    assert settings.head_port == 9999
    assert settings.hold_enabled is False
    assert settings.council_size == 5
    assert settings.speculative_edge_providers == ["groq", "cerebras"]


def test_settings_redacts_secrets() -> None:
    settings = Settings(api_key="super-secret", mesh_token="mesh-secret", admin_token="admin-secret")
    redacted = settings.redacted()
    assert redacted["api_key"] == "***"
    assert redacted["mesh_token"] == "***"
    assert redacted["admin_token"] == "***"
    assert "super-secret" not in json.dumps(redacted)


def test_settings_ensure_dirs(tmp_path: Path) -> None:
    settings = Settings(data_dir=str(tmp_path / "cluster"))
    settings.ensure_dirs()
    assert settings.data_path.is_dir()
    assert settings.state_path.is_dir()
    assert settings.logs_path.is_dir()


def test_settings_with_overrides_returns_copy() -> None:
    settings = Settings()
    derived = settings.with_overrides(head_port=1234)
    assert derived.head_port == 1234
    assert settings.head_port != 1234


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_json_formatter_emits_machine_readable_lines(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", json_mode=True, force=True)
    logger = get_logger("test.logger")
    with log_context(request="req_123"):
        logger.info("hello", extra={"hc_worker": "wk_1"})
    captured = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(captured)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["request"] == "req_123"
    assert payload["worker"] == "wk_1"


def test_secret_looking_fields_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", json_mode=True, force=True)
    logger = get_logger("test.logger")
    logger.info("cfg", extra={"hc_api_key": "leak-me"})
    captured = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(captured)
    assert payload["api_key"] == "***"


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
def test_chat_request_requires_messages() -> None:
    with pytest.raises(ValueError):
        ChatRequest(messages=[])


def test_chat_request_round_trips() -> None:
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="hello")],
        mode=RequestMode.DEEP,
        max_tokens=64,
    )
    payload = request.model_dump(mode="json")
    restored = ChatRequest(**payload)
    assert restored.messages[0].content == "hello"
    assert restored.mode is RequestMode.DEEP
    assert request.to_prompt().startswith("User: hello")


def test_worker_hello_rejects_unknown_protocol_major() -> None:
    with pytest.raises(ValueError):
        WorkerHello(worker_id="wk", protocol_version="9.0")
    hello = WorkerHello(worker_id="wk", protocol_version="1.0")
    assert hello.protocol_version == "1.0"


def test_worker_status_values_are_stable() -> None:
    assert WorkerStatus.IDLE.value == "idle"
    assert WorkerStatus.DRAINING.value == "draining"
    assert WorkerStatus.LOADING_MODEL.value == "loading_model"
