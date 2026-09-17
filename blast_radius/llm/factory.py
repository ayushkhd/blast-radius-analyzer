"""Chooses the language-model provider from settings.

A missing package or credential is not an error for the service: search,
host resolution and ranking need no language model. The reason is logged
once, here, and the caller carries on in no-LLM mode.

Provider SDKs read their keys from the process environment. For local runs
the key usually sits in a ``.env`` file, so that file is loaded into the
environment before a provider is built. The key goes from the file to the
SDK and nowhere else: not into ``config.Settings``, not into a log line and
not into a response.
"""

import logging
import os
import pathlib

from blast_radius import config
from blast_radius.llm import anthropic_provider
from blast_radius.llm import base
from blast_radius.llm import openai_provider

_LOG = logging.getLogger(__name__)

DOTENV_FILE = pathlib.Path(".env")


def load_dotenv(path: pathlib.Path = DOTENV_FILE) -> None:
  """Loads ``KEY=value`` lines from ``path`` into the environment.

  Variables that are already set win, so a deployment's real environment is
  never overridden by a stray file. Blank lines and ``#`` comments are
  ignored, a leading ``export`` is accepted, and surrounding quotes are
  stripped. A missing file is not an error.

  Args:
    path: The dotenv file.
  """
  if not path.is_file():
    return
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip().removeprefix("export ").strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def create(settings: config.Settings) -> base.Provider | None:
  """Returns the configured provider, or None to run without one.

  Args:
    settings: Which provider to use, and how.

  Raises:
    ValueError: If ``settings.llm_effort`` is not an effort level of the
      chosen provider.
  """
  if settings.llm_provider == "none":
    return None
  load_dotenv()
  try:
    if settings.llm_provider == "openai":
      return openai_provider.OpenAIProvider(settings)
    return anthropic_provider.AnthropicProvider(settings)
  except base.ProviderUnavailableError as err:
    _LOG.warning("running without a language model: %s", err)
    return None
