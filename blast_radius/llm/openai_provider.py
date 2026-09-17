"""The OpenAI implementation of the provider protocol.

One ``complete`` call is one request to the Responses API whose reply is
constrained to a JSON schema:

* The schema travels in ``text.format`` with ``strict`` set, so the API
  itself enforces the output contract. The prompts' schemas already meet
  strict mode's rules: every object closes ``additionalProperties`` and
  lists all of its properties as required.
* The trusted instructions go in ``instructions`` and the task input, which
  may embed fenced third-party text, in ``input``, so the two never share a
  message.
* Depth is set with ``reasoning.effort``. Reasoning models take no sampling
  parameters, so reproducibility comes from frozen, hashed prompts instead
  (see ``blast_radius.llm.prompts``).
* ``store`` is off. The context pack describes a real environment, and
  nothing here needs the provider to keep a copy of it.

A refusal arrives as an ordinary HTTP 200 whose message carries a refusal
item instead of text, so the reply's content is inspected before its text is
read. The corpus is vulnerability text, which makes that case a real one.

Retries and the per-attempt timeout are the SDK's own, set on the client from
settings. The ``openai`` package is an optional dependency. It is imported
when a provider is built or used, never when this module is imported.
"""

import json
import logging
import os
import time
from typing import Any, Literal, TYPE_CHECKING

from blast_radius import config
from blast_radius.llm import base

if TYPE_CHECKING:
  import openai
  from openai.types import responses as responses_types

_LOG = logging.getLogger(__name__)

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-5.6-luna"
ENV_API_KEY = "OPENAI_API_KEY"

# The name the API files the schema under; it has no effect on the reply.
_SCHEMA_NAME = "blast_radius_output"

_Effort = Literal["minimal", "low", "medium", "high"]
_EFFORTS: tuple[_Effort, ...] = ("minimal", "low", "medium", "high")


def _effort(value: str) -> _Effort:
  """Returns ``value`` as a reasoning effort the API accepts.

  Args:
    value: ``settings.llm_effort``.

  Raises:
    ValueError: If ``value`` is not a known effort level.
  """
  for effort in _EFFORTS:
    if value == effort:
      return effort
  known = ", ".join(_EFFORTS)
  raise ValueError(f"llm_effort must be one of {known}; got {value!r}")


def _build_client(settings: config.Settings) -> "openai.OpenAI":
  """Returns an SDK client that times out and retries as ``settings`` say.

  The SDK reads ``OPENAI_API_KEY`` from the environment itself. This function
  only checks that the variable is set, so the secret never passes through
  this package's own code.

  Args:
    settings: The per-attempt timeout and the retry count.

  Raises:
    base.ProviderUnavailableError: If the package is not installed or no key
      is configured.
  """
  try:
    import openai  # Optional extra. pylint: disable=import-outside-toplevel
  except ImportError as err:
    raise base.ProviderUnavailableError(
        "the openai package is not installed; install the project's 'openai'"
        " extra"
    ) from err
  if not os.environ.get(ENV_API_KEY):
    raise base.ProviderUnavailableError(
        f"no OpenAI credential found; set {ENV_API_KEY}, in the environment"
        " or in .env, to get the written brief"
    )
  return openai.OpenAI(
      timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries
  )


def _failed(
    started: float, kind: base.ResponseKind, err: Exception, detail: str
) -> base.RawResponse:
  """Returns the failed response for an exception the SDK raised.

  Only the exception's class and ``detail`` reach the error text. The SDK's
  own message embeds the body of the API's reply, which can echo the prompt.

  Args:
    started: ``time.monotonic()`` when the call began.
    kind: The failure class.
    err: The exception.
    detail: What happened, in words that are safe to log and to show.
  """
  return base.RawResponse(duration_s=time.monotonic() - started).fail(
      kind, f"{type(err).__name__}: {detail}"
  )


def _status(err: "openai.APIStatusError") -> str:
  """Returns the HTTP status of a failed call, with its request id if any."""
  if err.request_id:
    return f"HTTP {err.status_code}, request {err.request_id}"
  return f"HTTP {err.status_code}"


def _refused(reply: "responses_types.Response") -> bool:
  """Returns whether the model declined instead of answering."""
  for item in reply.output:
    if item.type != "message":
      continue
    if any(part.type == "refusal" for part in item.content):
      return True
  return False


def _read_reply(
    reply: "responses_types.Response", response: base.RawResponse
) -> base.RawResponse:
  """Fills ``response`` from the API's reply and returns it.

  Args:
    reply: The reply to a request that set ``text.format``.
    response: The response so far: timing, usage and the answering model.
  """
  # A refusal is a successful HTTP call with no usable text, so the content
  # is looked at before anything is read.
  if _refused(reply):
    return response.fail(base.KIND_REFUSAL, "the model declined the request")
  text = reply.output_text
  if not text:
    return response.fail(
        base.KIND_PARSE, f"the reply has no text (status: {reply.status})"
    )
  response.text = text
  try:
    decoded = json.loads(text)
  except json.JSONDecodeError:
    # The usual cause is a reply cut short, which the status shows.
    return response.fail(
        base.KIND_PARSE,
        f"the reply is not valid JSON (status: {reply.status})",
    )
  if not isinstance(decoded, dict):
    return response.fail(
        base.KIND_PARSE,
        f"the reply is JSON but not an object ({type(decoded).__name__})",
    )
  response.parsed = decoded
  return response


class OpenAIProvider:
  """Answers prompts with schema-constrained JSON from the Responses API.

  Attributes:
    name: ``openai``.
    model: The model that is asked.
  """

  name = PROVIDER_NAME

  def __init__(
      self,
      settings: config.Settings,
      *,
      client: "openai.OpenAI | None" = None,
  ) -> None:
    """Initialises the provider.

    Args:
      settings: Model, effort, output cap, timeout and retry count.
      client: An SDK client to use instead of building one. Tests pass a
        fake with the same method shape; its timeout and retries are then
        the caller's business.

    Raises:
      base.ProviderUnavailableError: If the ``openai`` package is not
        installed or no key is configured.
      ValueError: If ``settings.llm_effort`` is not an effort level.
    """
    self.model = settings.llm_model or DEFAULT_MODEL
    self._effort = _effort(settings.llm_effort)
    self._max_tokens = settings.llm_max_tokens
    self._timeout_s = settings.llm_timeout_s
    self._client = client if client is not None else _build_client(settings)

  def complete(
      self, system: str, prompt: str, schema: dict[str, Any]
  ) -> base.RawResponse:
    """Runs one call, with the SDK's retries, and never raises.

    Args:
      system: Instructions that frame the task. Trusted text only.
      prompt: The task input, which may embed fenced untrusted text.
      schema: JSON Schema the reply must satisfy.
    """
    import openai  # Optional extra. pylint: disable=import-outside-toplevel

    started = time.monotonic()
    # The handlers below run most specific first: to the SDK a timeout is a
    # connection error, and 404 is a status error.
    try:
      reply = self._client.responses.create(
          model=self.model,
          instructions=system,
          input=prompt,
          reasoning={"effort": self._effort},
          max_output_tokens=self._max_tokens,
          store=False,
          text={
              "format": {
                  "type": "json_schema",
                  "name": _SCHEMA_NAME,
                  "strict": True,
                  "schema": schema,
              }
          },
      )
    except openai.APITimeoutError as err:
      detail = f"no reply within {self._timeout_s:g} s per attempt"
      return _failed(started, base.KIND_TIMEOUT, err, detail)
    except openai.APIConnectionError as err:
      return _failed(started, base.KIND_PROVIDER, err, "API unreachable")
    except openai.NotFoundError as err:
      detail = (
          f"model {self.model!r} is unknown or not allowed ({_status(err)})"
      )
      return _failed(started, base.KIND_PROVIDER, err, detail)
    except openai.APIStatusError as err:
      detail = f"the API answered with an error ({_status(err)})"
      return _failed(started, base.KIND_PROVIDER, err, detail)
    except openai.OpenAIError as err:
      # Raised before a request goes out, e.g. a key removed after start-up.
      detail = "the SDK could not make the call"
      return _failed(started, base.KIND_PROVIDER, err, detail)

    usage = reply.usage
    response = base.RawResponse(
        duration_s=time.monotonic() - started,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        model=reply.model,
    )
    return _read_reply(reply, response)
