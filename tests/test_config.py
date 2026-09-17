"""Tests for blast_radius.config."""

import os
import pathlib

import pydantic
import pytest

from blast_radius import config


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
  """Keeps the machine's own BLAST_* variables and .env out of the tests."""
  for name in list(os.environ):
    if name.startswith("BLAST_"):
      monkeypatch.delenv(name)
  monkeypatch.chdir(tmp_path)


def test_defaults_enable_every_retrieval_stage():
  settings = config.Settings()

  assert settings.enable_keyword
  assert settings.enable_dense
  assert settings.enable_rerank
  assert settings.llm_provider == "openai"
  assert settings.llm_model is None


def test_threat_weights_sum_to_one_by_default():
  settings = config.Settings()

  total = (
      settings.weight_severity
      + settings.weight_epss
      + settings.weight_known_exploited
  )

  assert total == pytest.approx(1.0)


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setenv("BLAST_ENABLE_RERANK", "false")
  monkeypatch.setenv("BLAST_LLM_PROVIDER", "none")
  monkeypatch.setenv("BLAST_DATA_DIR", "/srv/exports")

  settings = config.Settings()

  assert not settings.enable_rerank
  assert settings.llm_provider == "none"
  assert settings.data_dir == pathlib.Path("/srv/exports")


def test_dotenv_file_in_the_working_directory_is_read(tmp_path: pathlib.Path):
  (tmp_path / ".env").write_text("BLAST_CONTEXT_COUNT=3\n", encoding="utf-8")

  assert config.Settings().context_count == 3


def test_export_paths_are_derived_from_the_data_directory():
  settings = config.Settings(data_dir=pathlib.Path("exports"))

  assert settings.assets_path == pathlib.Path(
      "exports/asset_data_scrubbed.json"
  )
  assert settings.vulns_path == pathlib.Path("exports/vulns_data_scrubbed.json")


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_max_chars": 0},
        {"candidate_count": 0},
        {"fusion_k": -1},
        {"weight_epss": -0.1},
        {"internet_facing_multiplier": 0.5},
        {"llm_provider": "someone-else"},
    ],
)
def test_out_of_range_values_are_rejected(overrides: dict[str, object]):
  with pytest.raises(pydantic.ValidationError):
    config.Settings(**overrides)  # type: ignore[arg-type]
