"""The ``blast-radius`` command.

    blast-radius ingest        build the index artifact from the exports
    blast-radius serve         run the API and UI
    blast-radius ask QUERY     answer one query in the terminal
    blast-radius eval          print the evaluation table
    blast-radius fetch-models  download the retrieval models into the cache

Each subcommand is thin: read settings, call one library function, print the
result. Machine-readable output goes to stdout and logs to stderr, so
``blast-radius ask --json ... | jq`` works. Errors a user can act on (no
exports, no artifact, a model that does not fit the artifact, a malformed
question set) become one line on stderr and exit status 2 instead of a
traceback.

Flags override environment variables, which override the defaults in
``config.Settings``.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
import sys
from typing import Any

import pydantic

import blast_radius
from blast_radius import config
from blast_radius import embeddings
from blast_radius import ingest
from blast_radius import logging_config
from blast_radius import models
from blast_radius import services as services_lib
from blast_radius import store
from blast_radius.evaluation import questions as questions_lib
from blast_radius.retrieval import rerank
from blast_radius.retrieval import retriever as retriever_lib

_LOG = logging.getLogger(__name__)

_EXIT_OK = 0
_EXIT_USAGE = 2

_DEFAULT_QUESTIONS = pathlib.Path("eval/questions.jsonl")
_DEFAULT_RESULTS = pathlib.Path("eval/results.json")

# How many groups and evidence items the terminal report lists before it
# says how many more there are; the JSON output is never truncated.
_REPORT_GROUPS = 12
_REPORT_EVIDENCE = 10
_REPORT_HEADER = "priority  hosts  facing  group"


class _UserError(Exception):
  """A failure the user can fix, reported without a traceback."""


def _parser() -> argparse.ArgumentParser:
  """Returns the argument parser for every subcommand."""
  parser = argparse.ArgumentParser(
      prog="blast-radius",
      description=(
          "Which hosts does a vulnerability affect, which should be fixed"
          " first, and what does the data say to do?"
      ),
  )
  parser.add_argument(
      "--version", action="version", version=blast_radius.__version__
  )
  commands = parser.add_subparsers(dest="command", required=True)

  ingest_cmd = commands.add_parser(
      "ingest", help="build the index artifact from the scanner exports"
  )
  ingest_cmd.add_argument(
      "--data-dir", type=pathlib.Path, help="directory holding the two exports"
  )
  ingest_cmd.add_argument(
      "--artifact", type=pathlib.Path, help="where to write the artifact"
  )

  serve_cmd = commands.add_parser("serve", help="run the API and UI")
  serve_cmd.add_argument("--host", help="interface to bind")
  serve_cmd.add_argument("--port", type=int, help="port to bind")

  ask_cmd = commands.add_parser("ask", help="answer one query")
  ask_cmd.add_argument(
      "query", help='a CVE id, "QID 12345", or advisory text in quotes'
  )
  ask_cmd.add_argument(
      "--json", action="store_true", help="print the full response as JSON"
  )
  ask_cmd.add_argument(
      "--no-llm", action="store_true", help="skip the written brief"
  )

  evaluate_cmd = commands.add_parser("eval", help="print the evaluation table")
  evaluate_cmd.add_argument(
      "--questions", type=pathlib.Path, default=_DEFAULT_QUESTIONS
  )
  evaluate_cmd.add_argument(
      "--out", type=pathlib.Path, default=_DEFAULT_RESULTS
  )

  commands.add_parser(
      "fetch-models", help="download the retrieval models into the cache"
  )
  return parser


def _settings(**overrides: Any) -> config.Settings:
  """Returns settings with every flag the user actually passed applied.

  Args:
    **overrides: Setting names mapped to flag values; None means "not
      passed" and leaves the environment or the default in charge.
  """
  passed = {
      name: value for name, value in overrides.items() if value is not None
  }
  return config.Settings(**passed)


def _run_ingest(args: argparse.Namespace) -> int:
  """Builds the artifact and prints the ingest report as JSON.

  Args:
    args: The parsed command line.
  """
  settings = _settings(data_dir=args.data_dir, artifact_path=args.artifact)
  for path in (settings.assets_path, settings.vulns_path):
    if not path.is_file():
      raise _UserError(
          f"{path} not found; put both scanner exports in"
          f" {settings.data_dir}/ or pass --data-dir"
      )
  try:
    report = ingest.build_artifact(
        settings.assets_path,
        settings.vulns_path,
        settings.artifact_path,
        embeddings.create(settings),
        chunk_max_chars=settings.chunk_max_chars,
        chunk_overlap_sentences=settings.chunk_overlap_sentences,
    )
  except ingest.IngestError as error:
    raise _UserError(str(error)) from error
  json.dump(dataclasses.asdict(report), sys.stdout, indent=2)
  sys.stdout.write("\n")
  return _EXIT_OK


def _run_serve(args: argparse.Namespace) -> int:
  """Runs the API until interrupted.

  Args:
    args: The parsed command line.
  """
  # Imported here so that `ingest` and `ask` never pay for the web stack.
  import uvicorn  # pylint: disable=import-outside-toplevel

  from blast_radius import api  # pylint: disable=import-outside-toplevel

  settings = _settings(host=args.host, port=args.port)
  uvicorn.run(
      api.create_app(settings),
      host=settings.host,
      port=settings.port,
      # The application configures logging itself, as JSON lines.
      log_config=None,
      access_log=False,
  )
  return _EXIT_OK


def _build_services(
    settings: config.Settings, *, use_llm: bool
) -> services_lib.Services:
  """Returns the object graph, turning setup failures into user errors.

  Args:
    settings: The process's settings.
    use_llm: False to skip the language model whatever the settings say.
  """
  try:
    return services_lib.build(settings, use_llm=use_llm)
  except (store.StoreError, retriever_lib.RetrieverConfigError) as error:
    raise _UserError(str(error)) from error


def _run_ask(args: argparse.Namespace) -> int:
  """Answers one query and prints a report, or the raw response.

  Args:
    args: The parsed command line.
  """
  services = _build_services(_settings(), use_llm=not args.no_llm)
  try:
    response = services.pipeline.analyze(args.query)
  except ValueError as error:
    raise _UserError(str(error)) from error
  finally:
    services.close()
  if args.json:
    sys.stdout.write(response.model_dump_json(indent=2) + "\n")
  else:
    sys.stdout.write(format_report(response))
  return _EXIT_OK


def _run_evaluation(args: argparse.Namespace) -> int:
  """Runs the evaluation and prints its table.

  Args:
    args: The parsed command line.
  """
  # Imported here because the runner pulls in every retrieval model.
  # Lazy on purpose, see above. pylint: disable-next=import-outside-toplevel
  from blast_radius.evaluation import runner

  if not args.questions.is_file():
    raise _UserError(f"{args.questions} not found")
  try:
    return runner.run(_settings(), args.questions, args.out)
  except store.ArtifactNotFoundError as error:
    raise _UserError(f"{error}; run `blast-radius ingest` first") from error
  except (
      questions_lib.QuestionSetError,
      store.StoreError,
      retriever_lib.RetrieverConfigError,
  ) as error:
    raise _UserError(str(error)) from error


def _run_fetch_models(args: argparse.Namespace) -> int:
  """Loads each retrieval model once, which downloads it into the cache.

  Args:
    args: The parsed command line.
  """
  del args  # Unused: the command has no flags.
  settings = _settings()
  embedder = embeddings.create(settings)
  reranker = rerank.create(settings)
  json.dump(
      {
          "cache_dir": str(settings.model_cache_dir),
          "embedding_model": embedder.name,
          "embedding_dim": embedder.dim,
          "rerank_model": reranker.name,
      },
      sys.stdout,
      indent=2,
  )
  sys.stdout.write("\n")
  return _EXIT_OK


def format_report(response: models.AnalyzeResponse) -> str:
  """Returns a terminal-friendly rendering of a response.

  The report is for reading; ``--json`` is for everything else. It shows the
  written brief when there is one, flags claims that failed verification,
  and lists groups instead of hosts, as the UI does.

  Args:
    response: The pipeline's response.
  """
  lines = [response.summary, ""]
  if response.answer is not None:
    lines.append(response.answer.summary)
    for claim in response.answer.claims:
      mark = "ok" if claim.verified else "UNVERIFIED"
      sources = ", ".join(c.source_id for c in claim.citations) or "no source"
      lines.append(f"  [{mark}] {claim.text} ({sources})")
      lines.extend(f"         ! {problem}" for problem in claim.problems)
    lines.append("")

  if response.groups:
    lines.append(_REPORT_HEADER)
    for group in response.groups[:_REPORT_GROUPS]:
      examples = ", ".join(group.example_hosts)
      lines.append(
          f"{group.priority:8.2f}  {group.count:5d}"
          f"  {group.internet_facing_count:6d}  {group.label}  ({examples})"
      )
    hidden = len(response.groups) - _REPORT_GROUPS
    if hidden > 0:
      lines.append(f"          ... and {hidden} more groups")
    lines.append("")

  if response.fix_evidence:
    lines.append("Fix evidence:")
    for evidence in response.fix_evidence[:_REPORT_EVIDENCE]:
      lines.append(f"  [{evidence.id}] {evidence.text}")
    hidden = len(response.fix_evidence) - _REPORT_EVIDENCE
    if hidden > 0:
      lines.append(f"  ... and {hidden} more")
    lines.append("")

  lines.extend(f"Caveat: {caveat}" for caveat in response.caveats)
  lines.extend(f"Note: {notice}" for notice in response.notices)
  return "\n".join(lines).rstrip() + "\n"


_COMMANDS = {
    "ingest": _run_ingest,
    "serve": _run_serve,
    "ask": _run_ask,
    "eval": _run_evaluation,
    "fetch-models": _run_fetch_models,
}


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the command line and returns the process exit status.

  Args:
    argv: Arguments without the program name. Defaults to ``sys.argv[1:]``.
  """
  args = _parser().parse_args(argv)
  try:
    logging_config.configure(config.Settings().log_level)
    return _COMMANDS[args.command](args)
  except pydantic.ValidationError as error:
    # A bad BLAST_* value: name the settings, skip the traceback.
    fields = ", ".join(str(issue["loc"][0]) for issue in error.errors())
    sys.stderr.write(f"blast-radius: invalid settings: {fields}\n")
    return _EXIT_USAGE
  except (_UserError, ValueError) as error:
    sys.stderr.write(f"blast-radius: {error}\n")
    return _EXIT_USAGE


if __name__ == "__main__":
  sys.exit(main())
