"""Tests for blast_radius.cli and ``python -m blast_radius``."""

from collections.abc import Iterator
import json
import logging
import os
import pathlib
import runpy
import shutil
from typing import Any, NoReturn

import fastapi
import pytest
import uvicorn

import blast_radius
from blast_radius import cli
from blast_radius import config
from blast_radius import models
from blast_radius import store as store_lib
from blast_radius.evaluation import runner
from blast_radius.llm import anthropic_provider

# A QID of the synthetic exports; tests/fixtures/make_fixtures.py defines it.
_QID_OPENSSH = "710001"
_ASK = ["ask", f"QID {_QID_OPENSSH}"]


@pytest.fixture(autouse=True)
def _clean_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  """Keeps the machine's own BLAST_* variables and .env out of the tests.

  The command builds ``config.Settings()`` itself, from the environment and
  the working directory, so both are replaced. The retrieval models are set
  to the model-free ones, so that no command can download anything.

  Args:
    monkeypatch: pytest's patcher, which undoes all of this afterwards.
    tmp_path: The test's own directory, which holds no ``.env``.
  """
  for name in list(os.environ):
    if name.startswith("BLAST_"):
      monkeypatch.delenv(name)
  monkeypatch.setenv("BLAST_EMBEDDING_MODEL", "hashing-256")
  monkeypatch.setenv("BLAST_RERANK_MODEL", "lexical")
  monkeypatch.setenv("BLAST_LLM_PROVIDER", "none")
  monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
  """Undoes the logging set-up that every run of ``main`` performs."""
  root = logging.getLogger()
  handlers = list(root.handlers)
  level = root.level
  yield
  for handler in list(root.handlers):
    root.removeHandler(handler)
  for handler in handlers:
    root.addHandler(handler)
  root.setLevel(level)


@pytest.fixture(name="exports_dir")
def fixture_exports_dir(
    tmp_path: pathlib.Path,
    assets_path: pathlib.Path,
    vulns_path: pathlib.Path,
) -> pathlib.Path:
  """Returns a data directory holding the synthetic exports.

  They are copied under the file names the settings look for.

  Args:
    tmp_path: The test's own directory.
    assets_path: The synthetic asset export.
    vulns_path: The synthetic vulnerability export.
  """
  directory = tmp_path / "exports"
  directory.mkdir()
  settings = config.Settings(_env_file=None, data_dir=directory)
  shutil.copy(assets_path, settings.assets_path)
  shutil.copy(vulns_path, settings.vulns_path)
  return directory


def _fail_if_called(*args: Any, **kwargs: Any) -> NoReturn:
  """Stands in for a function that the code under test must not reach."""
  raise AssertionError(f"unexpected call with {args} {kwargs}")


def _error_line(stderr: str) -> str:
  """Returns the line a failed command wrote for the user.

  It is the last line of stderr, below any log lines, and the helper checks
  that no traceback came with it.
  """
  assert "Traceback" not in stderr
  line = stderr.splitlines()[-1]
  assert line.startswith("blast-radius: ")
  return line


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def test_ingest_builds_the_artifact_and_prints_the_report_as_json(
    exports_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
  artifact = tmp_path / "built" / "index.sqlite"

  status = cli.main(
      ["ingest", "--data-dir", str(exports_dir), "--artifact", str(artifact)]
  )

  assert status == 0
  report = json.loads(capsys.readouterr().out)
  assert report["hosts"] == 11
  assert report["explained_qids"] == 6
  assert report["trace_chunks"] == 2
  with store_lib.Store(artifact) as built:
    assert built.stats().hosts == 11


def test_ingest_missing_exports_exits_2_with_one_line(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
  empty = tmp_path / "empty"
  empty.mkdir()

  status = cli.main(["ingest", "--data-dir", str(empty)])

  assert status == 2
  captured = capsys.readouterr()
  assert captured.out == ""
  assert captured.err.count("\n") == 1
  assert "asset_data_scrubbed.json not found" in _error_line(captured.err)
  assert not (tmp_path / "artifacts").exists()


def test_ingest_malformed_export_exits_2_with_the_reason(
    exports_dir: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
  (exports_dir / "asset_data_scrubbed.json").write_text("{", encoding="utf-8")

  status = cli.main(["ingest", "--data-dir", str(exports_dir)])

  assert status == 2
  assert "asset_data_scrubbed.json" in _error_line(capsys.readouterr().err)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_json_prints_the_whole_response_on_stdout(
    artifact_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  monkeypatch.setenv("BLAST_ARTIFACT_PATH", str(artifact_path))

  status = cli.main([*_ASK, "--json"])

  assert status == 0
  response = models.AnalyzeResponse.model_validate_json(capsys.readouterr().out)
  assert response.status == "matched"
  assert [match.qid for match in response.matches] == [_QID_OPENSSH]
  assert len(response.hosts) == 8


def test_ask_prints_a_report_of_groups_evidence_and_notes(
    artifact_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  monkeypatch.setenv("BLAST_ARTIFACT_PATH", str(artifact_path))

  status = cli.main(_ASK)

  assert status == 0
  lines = capsys.readouterr().out.splitlines()
  assert lines[0].startswith(f"Matched QID {_QID_OPENSSH}")
  assert "priority  hosts  facing  group" in lines
  assert any(
      line.endswith("fx-bastion-sg  (fx-bastion)") for line in lines
  ), lines
  assert "  [e1] Affected Versions: OpenSSH up to version 9.6" in lines
  assert lines[-1].startswith("Note: No language model is configured")


def test_ask_no_llm_never_creates_the_configured_provider(
    artifact_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  monkeypatch.setenv("BLAST_ARTIFACT_PATH", str(artifact_path))
  monkeypatch.setenv("BLAST_LLM_PROVIDER", "anthropic")
  monkeypatch.setattr(anthropic_provider, "create", _fail_if_called)

  status = cli.main([*_ASK, "--no-llm", "--json"])

  assert status == 0
  response = json.loads(capsys.readouterr().out)
  assert response["meta"]["llm_provider"] is None


def test_ask_without_an_artifact_exits_2_and_points_at_ingest(
    capsys: pytest.CaptureFixture[str],
):
  status = cli.main(_ASK)

  assert status == 2
  captured = capsys.readouterr()
  assert captured.out == ""
  assert captured.err.count("\n") == 1
  line = _error_line(captured.err)
  assert "no index artifact at artifacts/index.sqlite" in line
  assert "blast-radius ingest" in line


def test_ask_artifact_built_with_another_embedder_exits_2(
    artifact_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  monkeypatch.setenv("BLAST_ARTIFACT_PATH", str(artifact_path))
  monkeypatch.setenv("BLAST_EMBEDDING_MODEL", "hashing-128")

  status = cli.main(_ASK)

  assert status == 2
  assert "BLAST_EMBEDDING_MODEL" in _error_line(capsys.readouterr().err)


def test_ask_blank_query_exits_2(
    artifact_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  monkeypatch.setenv("BLAST_ARTIFACT_PATH", str(artifact_path))

  status = cli.main(["ask", "   "])

  assert status == 2
  assert "must not be blank" in _error_line(capsys.readouterr().err)


# ---------------------------------------------------------------------------
# serve, eval, fetch-models and --version
# ---------------------------------------------------------------------------


def test_serve_runs_uvicorn_on_the_configured_address_without_its_logging(
    monkeypatch: pytest.MonkeyPatch,
):
  calls: list[tuple[Any, dict[str, Any]]] = []
  monkeypatch.setattr(
      uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs))
  )
  monkeypatch.setenv("BLAST_HOST", "0.0.0.0")
  monkeypatch.setenv("BLAST_PORT", "7000")

  status = cli.main(["serve", "--port", "9001"])

  assert status == 0
  assert len(calls) == 1
  app, kwargs = calls[0]
  assert isinstance(app, fastapi.FastAPI)
  assert kwargs["host"] == "0.0.0.0"
  # The flag wins over the environment.
  assert kwargs["port"] == 9001
  assert kwargs["log_config"] is None


def test_eval_missing_questions_file_exits_2(
    capsys: pytest.CaptureFixture[str],
):
  status = cli.main(["eval", "--questions", "nowhere/questions.jsonl"])

  assert status == 2
  line = _error_line(capsys.readouterr().err)
  assert line == "blast-radius: nowhere/questions.jsonl not found"


def test_eval_malformed_questions_file_exits_2_naming_the_line(
    artifact_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
  monkeypatch.setenv("BLAST_ARTIFACT_PATH", str(artifact_path))
  questions = tmp_path / "questions.jsonl"
  questions.write_text("not json\n", encoding="utf-8")

  status = cli.main(["eval", "--questions", str(questions)])

  assert status == 2
  assert f"{questions}:1:" in _error_line(capsys.readouterr().err)


def test_eval_hands_the_paths_to_the_runner_and_returns_its_status(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  questions = tmp_path / "questions.jsonl"
  questions.touch()
  results = tmp_path / "results.json"
  calls: list[tuple[pathlib.Path, pathlib.Path]] = []

  def run(
      settings: config.Settings,
      questions_path: pathlib.Path,
      results_path: pathlib.Path,
  ) -> int:
    del settings  # Unused: the paths are what the command line decides.
    calls.append((questions_path, results_path))
    return 1

  monkeypatch.setattr(runner, "run", run)

  status = cli.main(
      ["eval", "--questions", str(questions), "--out", str(results)]
  )

  assert status == 1
  assert calls == [(questions, results)]


def test_fetch_models_loads_both_models_and_prints_what_it_loaded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
  monkeypatch.setenv("BLAST_MODEL_CACHE_DIR", "model-cache")

  status = cli.main(["fetch-models"])

  assert status == 0
  assert json.loads(capsys.readouterr().out) == {
      "cache_dir": "model-cache",
      "embedding_model": "hashing-256",
      "embedding_dim": 256,
      "rerank_model": "lexical",
  }


def test_version_flag_prints_the_package_version(
    capsys: pytest.CaptureFixture[str],
):
  with pytest.raises(SystemExit) as exit_info:
    cli.main(["--version"])

  assert exit_info.value.code == 0
  assert capsys.readouterr().out.strip() == blast_radius.__version__


def test_python_dash_m_runs_the_command_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
  monkeypatch.setattr("sys.argv", ["blast-radius", "--version"])

  with pytest.raises(SystemExit) as exit_info:
    runpy.run_module("blast_radius", run_name="__main__")

  assert exit_info.value.code == 0
  assert capsys.readouterr().out.strip() == blast_radius.__version__


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def _response(**fields: Any) -> models.AnalyzeResponse:
  """Returns a matched response with ``fields`` filled in."""
  return models.AnalyzeResponse(
      query="QID 1",
      status="matched",
      parsed=models.ParsedQuery(raw="QID 1", qids=["1"]),
      summary="Matched QID 1.",
      meta=models.ResponseMeta(version=blast_radius.__version__),
      **fields,
  )


def _group(number: int) -> models.HostGroup:
  """Returns a group of two hosts whose names carry ``number``."""
  return models.HostGroup(
      key=f"sg:web-{number}",
      label=f"web-{number}",
      count=2,
      priority=1.5,
      internet_facing_count=1,
      example_hosts=[f"web-{number}a", f"web-{number}b"],
  )


def test_format_report_flags_the_unverified_claim_with_its_problems():
  answer = models.Answer(
      summary="Two hosts need the update.",
      claims=[
          models.Claim(
              text="The fix is in 9.7.",
              citations=[
                  models.Citation(source_id="c1", quote="9.7", verified=True)
              ],
              verified=True,
          ),
          models.Claim(
              text="It is exploited in the wild.",
              verified=False,
              problems=["cites no source"],
          ),
      ],
  )

  report = cli.format_report(_response(answer=answer, groups=[_group(1)]))

  assert report.splitlines() == [
      "Matched QID 1.",
      "",
      "Two hosts need the update.",
      "  [ok] The fix is in 9.7. (c1)",
      "  [UNVERIFIED] It is exploited in the wild. (no source)",
      "         ! cites no source",
      "",
      "priority  hosts  facing  group",
      "    1.50      2       1  web-1  (web-1a, web-1b)",
  ]
  assert report.endswith("\n")


def test_format_report_cuts_long_lists_and_says_how_much_is_hidden():
  evidence = [
      models.FixEvidence(
          id=f"e{number}",
          kind="patch_reference",
          doc_type="cve",
          doc_id="CVE-2099-0001",
          text=f"Patch {number}",
      )
      for number in range(1, 13)
  ]
  groups = [_group(number) for number in range(1, 15)]

  report = cli.format_report(
      _response(
          groups=groups,
          fix_evidence=evidence,
          caveats=["No CVE is attached."],
          notices=["No brief."],
      )
  )

  lines = report.splitlines()
  assert "          ... and 2 more groups" in lines
  assert not any("web-13" in line for line in lines)
  assert "  [e10] Patch 10" in lines
  assert "  [e11] Patch 11" not in lines
  assert "  ... and 2 more" in lines
  assert lines[-2:] == ["Caveat: No CVE is attached.", "Note: No brief."]
