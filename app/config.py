from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_RAG_CONFIG_CACHE: dict[str, Any] | None = None
_APP_CONFIG_CACHE: dict[str, Any] | None = None
_BOT_RESPONSES_CACHE: dict[str, Any] | None = None
_SIMULATOR_PROMPTS_CACHE: dict[str, Any] | None = None
_TELEGRAM_RESPONSES_CACHE: dict[str, Any] | None = None
_VK_RESPONSES_CACHE: dict[str, Any] | None = None


def load_rag_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load RAG configuration once and cache it."""
    global _RAG_CONFIG_CACHE
    if _RAG_CONFIG_CACHE is not None:
        return _RAG_CONFIG_CACHE

    config_path = Path(path) if path else Path("configs/rag_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"RAG config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _RAG_CONFIG_CACHE = yaml.safe_load(fp) or {}
    return _RAG_CONFIG_CACHE


def load_app_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load app configuration once and cache it."""
    global _APP_CONFIG_CACHE
    if _APP_CONFIG_CACHE is not None:
        return _APP_CONFIG_CACHE

    config_path = Path(path) if path else Path("configs/app_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"App config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _APP_CONFIG_CACHE = yaml.safe_load(fp) or {}
    return _APP_CONFIG_CACHE


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load combined configuration for backward compatibility."""
    # Объединяем RAG и app конфигурации
    rag_config = load_rag_config()
    app_config = load_app_config()

    # Объединяем конфиги (app_config перезаписывает rag_config при конфликтах)
    combined = {**rag_config, **app_config}
    return combined


def load_bot_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load bot responses configuration once and cache it."""
    global _BOT_RESPONSES_CACHE
    if _BOT_RESPONSES_CACHE is not None:
        return _BOT_RESPONSES_CACHE

    config_path = Path(path) if path else Path("bot_responses.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Bot responses config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _BOT_RESPONSES_CACHE = yaml.safe_load(fp) or {}
    return _BOT_RESPONSES_CACHE


def load_telegram_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load Telegram responses configuration from app_config."""
    app_config = load_app_config()
    return app_config.get("telegram_responses", {})


def load_vk_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load VK responses configuration from app_config."""
    app_config = load_app_config()
    return app_config.get("vk_responses", {})


def load_simulator_prompts(path: str | Path | None = None) -> dict[str, Any]:
    """Load simulator prompts configuration from separate simulator_config."""
    global _SIMULATOR_PROMPTS_CACHE
    if _SIMULATOR_PROMPTS_CACHE is not None:
        return _SIMULATOR_PROMPTS_CACHE

    config_path = Path(path) if path else Path("configs/simulator_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Simulator config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _SIMULATOR_PROMPTS_CACHE = yaml.safe_load(fp) or {}
    return _SIMULATOR_PROMPTS_CACHE


def reset_cache() -> None:
    global _RAG_CONFIG_CACHE, _APP_CONFIG_CACHE, _BOT_RESPONSES_CACHE, _SIMULATOR_PROMPTS_CACHE, _TELEGRAM_RESPONSES_CACHE, _VK_RESPONSES_CACHE
    _RAG_CONFIG_CACHE = None
    _APP_CONFIG_CACHE = None
    _BOT_RESPONSES_CACHE = None
    _SIMULATOR_PROMPTS_CACHE = None
    _TELEGRAM_RESPONSES_CACHE = None
    _VK_RESPONSES_CACHE = None
