"""Tests for env-driven config — see docs/changes/0005."""

from __future__ import annotations

import importlib

import src.config as config_mod


def test_defaults_when_env_unset(monkeypatch, tmp_path):
    for var in ("HR_PDF_DIRECTORY", "HR_PERSIST_DIRECTORY",
                "HR_CHUNK_SIZE", "HR_CHUNK_OVERLAP", "HR_TOP_K", "HR_CANDIDATE_K"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HR_PERSIST_DIRECTORY", str(tmp_path))  # don't touch real ~/.hybrid_retrieval
    monkeypatch.chdir(tmp_path)  # avoid loading the repo's .env

    importlib.reload(config_mod)
    s = config_mod.settings
    assert s.chunk_size == 1200
    assert s.chunk_overlap == 150
    assert s.top_k == 5
    assert s.candidate_k is None


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("HR_PDF_DIRECTORY", "/tmp/foo")
    monkeypatch.setenv("HR_PERSIST_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("HR_CHUNK_SIZE", "777")
    monkeypatch.setenv("HR_TOP_K", "11")
    monkeypatch.setenv("HR_CANDIDATE_K", "33")
    monkeypatch.chdir(tmp_path)

    importlib.reload(config_mod)
    s = config_mod.settings
    assert str(s.pdf_directory) == "/tmp/foo"
    assert s.chunk_size == 777
    assert s.top_k == 11
    assert s.candidate_k == 33


def test_persist_directory_is_created(monkeypatch, tmp_path):
    target = tmp_path / "newly_made"
    assert not target.exists()
    monkeypatch.setenv("HR_PERSIST_DIRECTORY", str(target))
    monkeypatch.chdir(tmp_path)

    importlib.reload(config_mod)
    assert target.is_dir()


def test_dotenv_loaded_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("HR_TOP_K", raising=False)
    monkeypatch.setenv("HR_PERSIST_DIRECTORY", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HR_TOP_K=42\n# comment\n")

    importlib.reload(config_mod)
    assert config_mod.settings.top_k == 42


def test_shell_env_wins_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("HR_TOP_K", "9")
    monkeypatch.setenv("HR_PERSIST_DIRECTORY", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HR_TOP_K=42\n")

    importlib.reload(config_mod)
    assert config_mod.settings.top_k == 9
