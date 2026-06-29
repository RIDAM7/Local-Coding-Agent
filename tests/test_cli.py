"""Phase 6 CLI tests. Settings/factory and preflight are mocked — no network."""

from unittest.mock import AsyncMock

import pytest

from agent import cli
from agent.cli import build_parser, _normalize_argv, main, ROLES
from agent.config import settings
from agent.exceptions.errors import PreflightError


# --- Entry point / metadata ---------------------------------------------------

def test_cli_main_is_importable_and_callable():
    import agent.cli as mod
    assert callable(mod.main)


def test_main_module_delegates_to_cli():
    import main as main_module
    assert main_module.main is cli.main


# --- Parser ------------------------------------------------------------------

def test_run_subcommand_parses_with_phase5_flags():
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(["run", "make a calculator", "--yes", "--dry-run"]))
    assert args.subcommand == "run"
    assert args.auto is True
    assert args.dry_run is True
    assert " ".join(args.task) == "make a calculator"


def test_bare_task_shorthand_maps_to_run():
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(["create calculator.py", "--auto"]))
    assert args.subcommand == "run"
    assert args.auto is True
    assert " ".join(args.task) == "create calculator.py"


def test_config_check_parses():
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(["config", "check"]))
    assert args.subcommand == "config"
    assert args.config_action == "check"


def test_models_parses():
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(["models"]))
    assert args.subcommand == "models"


def test_empty_argv_defaults_to_run():
    parser = build_parser()
    args = parser.parse_args(_normalize_argv([]))
    assert args.subcommand == "run"


# --- models: routing table for all 7 roles, no secrets -----------------------

def test_models_renders_all_roles_without_secrets(capsys, monkeypatch):
    monkeypatch.setattr(cli, "resolve_provider", lambda role: "openai" if role == "coder" else "ollama")
    monkeypatch.setattr(cli, "resolve_model", lambda role: f"model-{role}")
    monkeypatch.setattr(settings, "openai_api_key", "sk-FAKESECRET-DO-NOT-LEAK")

    rc = main(["models"])
    out = capsys.readouterr().out

    assert rc == 0
    for role in ROLES:
        assert role in out
    assert "model-planner" in out and "model-coder" in out
    assert "sk-FAKESECRET-DO-NOT-LEAK" not in out  # secret never printed


# --- config check: table + health + pass/fail --------------------------------

def test_config_check_renders_table_and_passes(capsys, monkeypatch):
    monkeypatch.setattr(cli, "resolve_provider", lambda role: "ollama")
    monkeypatch.setattr(cli, "resolve_model", lambda role: f"m-{role}")
    monkeypatch.setattr(cli, "_ollama_models", AsyncMock(return_value=["m-planner", "m-coder"]))
    monkeypatch.setattr(cli, "_model_present", lambda model, available: True)
    monkeypatch.setattr(cli, "tooling_check", lambda: None)
    monkeypatch.setattr(cli, "preflight_check", AsyncMock(return_value=None))
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-SHOULD-NOT-APPEAR")

    rc = main(["config", "check"])
    out = capsys.readouterr().out

    assert rc == 0
    for role in ROLES:
        assert role in out
    assert "config check: OK" in out
    # credentials are shown as set/missing only — never the value.
    assert ("set" in out) or ("missing" in out)
    assert "sk-ant-SHOULD-NOT-APPEAR" not in out


def test_config_check_exits_nonzero_when_preflight_fails(capsys, monkeypatch):
    monkeypatch.setattr(cli, "resolve_provider", lambda role: "openai")
    monkeypatch.setattr(cli, "resolve_model", lambda role: f"m-{role}")
    monkeypatch.setattr(cli, "tooling_check", lambda: None)
    monkeypatch.setattr(
        cli, "preflight_check",
        AsyncMock(side_effect=PreflightError(
            "role 'coder' (openai): missing API key. Set OPENAI_API_KEY in your .env.")),
    )

    rc = main(["config", "check"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "config check: FAILED" in out
    # the actionable, secret-free message is surfaced
    assert "OPENAI_API_KEY" in out
