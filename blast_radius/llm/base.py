"""The language-model provider protocol.

A provider turns a prompt plus a JSON schema into a ``RawResponse`` and knows
nothing about hosts, QIDs or briefs. That is the whole boundary: the pipeline
on one side never trusts what comes back, and providers on the other never
see the inventory.

Providers do not raise for anything that can go wrong during a call. A
timeout, an API error, a refusal and unparseable output all come back as a
``RawResponse`` with ``kind`` set, so the pipeline can degrade to its no-LLM
response without a try/except around every call site.
"""

import dataclasses
from typing import Any, Literal, Protocol

# Why an attempt produced no usable output. ``refusal`` is separate from
# ``provider`` because it is not retryable: the corpus is vulnerability text,
# and a safety classifier may decline it however often it is asked.
ResponseKind = Literal["timeout", "provider", "parse", "refusal"]
KIND_TIMEOUT: ResponseKind = "timeout"
KIND_PROVIDER: ResponseKind = "provider"
KIND_PARSE: ResponseKind = "parse"
KIND_REFUSAL: ResponseKind = "refusal"


class ProviderError(Exception):
  """Base class for provider failures raised at construction time."""


class ProviderUnavailableError(ProviderError):
  """The provider cannot run here: a missing package or credential."""


@dataclasses.dataclass
class RawResponse:
  """What a provider returned for one prompt.

  Attributes:
    text: Final message text, expected to be JSON.
    parsed: ``text`` decoded as a JSON object, when it decoded.
    duration_s: Wall-clock seconds across all attempts.
    attempts: How many attempts were made.
    input_tokens: Input tokens the provider reported, if any.
    output_tokens: Output tokens the provider reported, if any.
    model: The model that actually answered, which differs from the one
      requested when a server-side fallback stepped in.
    error: Description of the failure, when the call failed.
    kind: Failure class; ``None`` on success.
  """

  text: str = ""
  parsed: dict[str, Any] | None = None
  duration_s: float = 0.0
  attempts: int = 0
  input_tokens: int | None = None
  output_tokens: int | None = None
  model: str | None = None
  error: str | None = None
  kind: ResponseKind | None = None

  @property
  def ok(self) -> bool:
    """Returns whether the call produced a usable JSON object."""
    return self.parsed is not None and self.error is None

  def fail(self, kind: ResponseKind, error: str) -> "RawResponse":
    """Marks this response as failed and returns it, for one-line returns."""
    self.kind = kind
    self.error = error
    self.parsed = None
    return self


class Provider(Protocol):
  """Anything that can answer a prompt with schema-constrained JSON.

  Attributes:
    name: Short provider name recorded in responses, e.g. ``anthropic``.
    model: Model identifier recorded in responses.
  """

  name: str
  model: str

  def complete(
      self, system: str, prompt: str, schema: dict[str, Any]
  ) -> RawResponse:
    """Runs one call, with the provider's own retries, and never raises.

    Args:
      system: Instructions that frame the task. Trusted text only.
      prompt: The task input, which may embed fenced untrusted text.
      schema: JSON Schema the reply must satisfy.
    """
