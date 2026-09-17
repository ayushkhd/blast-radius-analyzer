"""Tests for blast_radius.llm.factory."""

import os
import pathlib
from typing import Any

import pytest

from blast_radius import config
from blast_radius.llm import anthropic_provider
from blast_radius.llm import factory
from blast_radius.llm import openai_provider

_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
  """Keeps the machine's own keys and .env out of the tests."""
  for name in _KEYS:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profile"))
  monkeypatch.chdir(tmp_path)


def _settings(**overrides: Any) -> config.Settings:
  return config.Settings(_env_file=None, **overrides)


def test_none_means_no_provider_even_with_a_key(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

  assert factory.create(_settings(llm_provider="none")) is None


def test_openai_is_the_default_and_asks_luna(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

  provider = factory.create(_settings())

  assert isinstance(provider, openai_provider.OpenAIProvider)
  assert provider.model == "gpt-5.6-luna"


def test_anthropic_is_built_when_asked_for(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

  provider = factory.create(_settings(llm_provider="anthropic"))

  assert isinstance(provider, anthropic_provider.AnthropicProvider)
  assert provider.model == "claude-opus-5"


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_missing_credential_degrades_to_no_provider_with_one_warning(
    name: str, caplog: pytest.LogCaptureFixture
):
  provider = factory.create(_settings(llm_provider=name))

  assert provider is None
  assert caplog.text.count("running without a language model") == 1


def test_key_in_dotenv_reaches_the_sdk_without_touching_settings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  (tmp_path / ".env").write_text(
      "# demo key\nexport OPENAI_API_KEY='sk-test-from-dotenv'\n",
      encoding="utf-8",
  )
  # setenv then delenv registers the variable with monkeypatch, so whatever
  # load_dotenv puts in the environment is removed again after the test.
  monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
  monkeypatch.delenv("OPENAI_API_KEY")

  provider = factory.create(_settings())

  assert isinstance(provider, openai_provider.OpenAIProvider)
  assert os.environ["OPENAI_API_KEY"] == "sk-test-from-dotenv"
  assert "sk-test-from-dotenv" not in _settings().model_dump_json()


def test_load_dotenv_never_overrides_the_real_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  path = tmp_path / "custom.env"
  path.write_text('OPENAI_API_KEY="from-file"\nnot a line\n', encoding="utf-8")
  monkeypatch.setenv("OPENAI_API_KEY", "from-environment")

  factory.load_dotenv(path)

  assert os.environ["OPENAI_API_KEY"] == "from-environment"


def test_load_dotenv_ignores_a_missing_file(tmp_path: pathlib.Path):
  factory.load_dotenv(tmp_path / "absent.env")
