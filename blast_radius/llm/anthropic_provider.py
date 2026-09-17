"""The Anthropic implementation of the provider protocol.

One ``complete`` call is one request to the Messages API whose reply is
constrained to a JSON schema. The request is shaped by what ``claude-opus-5``
accepts:

* No sampling parameters. The model answers ``temperature``, ``top_p`` or
  ``top_k`` with a 400, so none is sent. Reproducibility comes from frozen,
  hashed prompts instead (see ``blast_radius.llm.prompts``).
* No ``thinking`` parameter. Left out, thinking is adaptive, and its depth
  follows ``output_config.effort``.
* The schema travels in ``output_config.format``. There is no assistant
  prefill, which the model also rejects.
* ``fallbacks="default"``. The corpus is vulnerability text, and a safety
  classifier may decline it: an ordinary HTTP 200 whose ``stop_reason`` is
  ``"refusal"``. With the fallback, the API re-runs a declined request on the
  substitute model it recommends for that kind of refusal, inside the same
  call. The parameter is in beta, hence ``client.beta.messages`` and the
  ``betas`` header.

Retries and the per-attempt timeout are the SDK's own, set on the client from
settings. How many attempts a call took is known only when a reply arrived;
an exception carries no retry count, so a failed call leaves ``attempts`` at
zero.

The ``anthropic`` package is an optional dependency. It is imported when a
provider is built or used, never when this module is imported.
"""

import json
import logging
import time
from typing import Any, Literal, TYPE_CHECKING

from blast_radius import config
from blast_radius.llm import base

if TYPE_CHECKING:
  import anthropic
  from anthropic.types import beta as beta_types

_LOG = logging.getLogger(__name__)

PROVIDER_NAME = "anthropic"
DEFAULT_MODEL = "claude-opus-5"

_Effort = Literal["low", "medium", "high", "xhigh", "max"]
_EFFORTS: tuple[_Effort, ...] = ("low", "medium", "high", "xhigh", "max")

# Goes with the "default" form of ``fallbacks``. The older form, a list of
# models, takes a different header, and mixing the two is a 400.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _effort(value: str) -> _Effort:
  """Returns ``value`` as an effort level the API accepts.

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


def _build_client(settings: config.Settings) -> "anthropic.Anthropic":
  """Returns an SDK client that times out and retries as ``settings`` say.

  The SDK finds the credential itself: ``ANTHROPIC_API_KEY`` or
  ``ANTHROPIC_AUTH_TOKEN`` in the environment, or a profile written by
  ``ant auth login``. This function only asks the client whether it found
  one, so the secret never passes through this package's own code.

  Args:
    settings: The per-attempt timeout and the retry count.

  Raises:
    base.ProviderUnavailableError: If the package is not installed, or the
      SDK has no credential or cannot load the one it was pointed at.
  """
  try:
    import anthropic  # Optional extra. pylint: disable=import-outside-toplevel
  except ImportError as err:
    raise base.ProviderUnavailableError(
        "the anthropic package is not installed; install the project's"
        " 'anthropic' extra"
    ) from err
  try:
    client = anthropic.Anthropic(
        timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries
    )
  except anthropic.AnthropicError as err:
    # Raised for a credential profile that is named but missing or broken.
    # The SDK's message is left out: it quotes paths on the user's machine.
    raise base.ProviderUnavailableError(
        "the Anthropic SDK could not load its credentials"
        f" ({type(err).__name__}); check ANTHROPIC_PROFILE and the SDK's"
        " configuration directory"
    ) from err
  if (
      client.api_key is None
      and client.auth_token is None
      and client.credentials is None
  ):
    raise base.ProviderUnavailableError(
        "no Anthropic credential found; set ANTHROPIC_API_KEY to get the"
        " written brief"
    )
  return client


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


def _status(err: "anthropic.APIStatusError") -> str:
  """Returns the HTTP status of a failed call, with its request id if any."""
  if err.request_id:
    return f"HTTP {err.status_code}, request {err.request_id}"
  return f"HTTP {err.status_code}"


def _read_reply(
    message: "beta_types.BetaMessage", response: base.RawResponse
) -> base.RawResponse:
  """Fills ``response`` from the API's reply and returns it.

  Args:
    message: The reply to a request that set ``output_config.format``.
    response: The response so far: timing, usage and the answering model.
  """
  # A refusal is a successful HTTP call whose content is empty or partial,
  # so the stop reason is looked at before anything is read.
  if message.stop_reason == "refusal":
    details = message.stop_details
    category = details.category if details and details.category else "none"
    return response.fail(
        base.KIND_REFUSAL,
        f"the model declined the request (refusal category: {category})",
    )
  # Thinking and fallback blocks may come first; the JSON is the text block.
  text = next(
      (block.text for block in message.content if block.type == "text"), None
  )
  if text is None:
    return response.fail(
        base.KIND_PARSE,
        f"the reply has no text (stop reason: {message.stop_reason})",
    )
  response.text = text
  try:
    decoded = json.loads(text)
  except json.JSONDecodeError:
    # The usual cause is a reply cut short, which the stop reason shows.
    return response.fail(
        base.KIND_PARSE,
        f"the reply is not valid JSON (stop reason: {message.stop_reason})",
    )
  if not isinstance(decoded, dict):
    return response.fail(
        base.KIND_PARSE,
        f"the reply is JSON but not an object ({type(decoded).__name__})",
    )
  response.parsed = decoded
  return response


class AnthropicProvider:
  """Answers prompts with schema-constrained JSON from the Messages API.

  Attributes:
    name: ``anthropic``.
    model: The model that is asked. A fallback model may answer instead; the
      one that did is recorded on each ``RawResponse``.
  """

  name = PROVIDER_NAME

  def __init__(
      self,
      settings: config.Settings,
      *,
      client: "anthropic.Anthropic | None" = None,
  ) -> None:
    """Initialises the provider.

    Args:
      settings: Model, effort, output cap, timeout and retry count.
      client: An SDK client to use instead of building one. Tests pass a
        fake with the same method shape; its timeout and retries are then
        the caller's business.

    Raises:
      base.ProviderUnavailableError: If the ``anthropic`` package is not
        installed or the SDK finds no credential.
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

    Returns:
      The reply, or a failed response whose ``kind`` says what went wrong.
      ``error`` never quotes the prompt, the credential or the API's reply,
      which is what makes it safe to log here.
    """
    response = self._call(system, prompt, schema)
    if not response.ok:
      _LOG.warning(
          "language model call failed (%s): %s", response.kind, response.error
      )
    return response

  def _call(
      self, system: str, prompt: str, schema: dict[str, Any]
  ) -> base.RawResponse:
    """Sends the request and maps every outcome onto a ``RawResponse``.

    Args:
      system: Instructions that frame the task.
      prompt: The task input.
      schema: JSON Schema the reply must satisfy.
    """
    import anthropic  # Optional extra. pylint: disable=import-outside-toplevel

    started = time.monotonic()
    # The handlers below run most specific first: to the SDK a timeout is a
    # connection error, and 504 and 404 are status errors.
    try:
      raw = self._client.beta.messages.with_raw_response.create(
          model=self.model,
          max_tokens=self._max_tokens,
          system=system,
          messages=[{"role": "user", "content": prompt}],
          output_config={
              "effort": self._effort,
              "format": {"type": "json_schema", "schema": schema},
          },
          fallbacks="default",
          betas=[_FALLBACK_BETA],
      )
      message = raw.parse()
    except anthropic.APITimeoutError as err:
      detail = f"no reply within {self._timeout_s:g} s per attempt"
      return _failed(started, base.KIND_TIMEOUT, err, detail)
    except anthropic.APIConnectionError as err:
      return _failed(started, base.KIND_PROVIDER, err, "API unreachable")
    except anthropic.DeadlineExceededError as err:
      detail = f"the API gave up waiting for the model ({_status(err)})"
      return _failed(started, base.KIND_TIMEOUT, err, detail)
    except anthropic.NotFoundError as err:
      # On this endpoint a 404 is about the model. The API gives the same
      # answer for an id it does not know and for one the credential may not
      # use, so the message has to name both.
      detail = (
          f"model {self.model!r} is unknown or not allowed ({_status(err)})"
      )
      return _failed(started, base.KIND_PROVIDER, err, detail)
    except anthropic.APIStatusError as err:
      detail = f"the API answered with an error ({_status(err)})"
      return _failed(started, base.KIND_PROVIDER, err, detail)
    except anthropic.AnthropicError as err:
      # Raised before a request goes out, e.g. a token that cannot be renewed.
      detail = "the SDK could not make the call"
      return _failed(started, base.KIND_PROVIDER, err, detail)

    response = base.RawResponse(
        duration_s=time.monotonic() - started,
        attempts=raw.retries_taken + 1,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        model=message.model,
    )
    return _read_reply(message, response)


def create(settings: config.Settings) -> base.Provider | None:
  """Returns the configured provider, or None to run without one.

  A missing package or credential is not an error for the service: search,
  host resolution and ranking need no language model. The reason is logged
  once, here, and the caller carries on in no-LLM mode.

  Args:
    settings: Which provider to use, and how.

  Raises:
    ValueError: If ``settings.llm_effort`` is not an effort level.
  """
  if settings.llm_provider == "none":
    return None
  try:
    return AnthropicProvider(settings)
  except base.ProviderUnavailableError as err:
    _LOG.warning("running without a language model: %s", err)
    return None
