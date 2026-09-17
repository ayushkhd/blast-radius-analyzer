"""Test doubles shared by the unit, pipeline and API tests.

``ScriptedProvider`` stands in for a language model: a test says up front
what each call returns, and afterwards reads back what each call was given.
No test needs a network, a key or the ``anthropic`` package to exercise the
steps that talk to a model.
"""

from collections.abc import Sequence
import json
from typing import Any, NamedTuple

from blast_radius.llm import base

MODEL = "scripted-model"


class Call(NamedTuple):
  """One ``complete`` call as the provider received it."""

  system: str
  prompt: str
  schema: dict[str, Any]


def _failed(kind: base.ResponseKind, error: str) -> base.RawResponse:
  """Returns a response that failed with ``kind`` after one attempt."""
  return base.RawResponse(attempts=1, model=MODEL).fail(kind, error)


def timeout() -> base.RawResponse:
  """Returns the response of a call that timed out."""
  return _failed(base.KIND_TIMEOUT, "scripted timeout")


def provider_error() -> base.RawResponse:
  """Returns the response of a call the API or the network failed."""
  return _failed(base.KIND_PROVIDER, "scripted provider error")


def refusal() -> base.RawResponse:
  """Returns the response of a call the model declined to answer."""
  return _failed(base.KIND_REFUSAL, "scripted refusal")


def unparseable(text: str = "not json") -> base.RawResponse:
  """Returns the response of a call whose reply was not a JSON object.

  Args:
    text: The reply, kept on the response as a real provider would keep it.
  """
  response = _failed(base.KIND_PARSE, "scripted parse failure")
  response.text = text
  return response


class ScriptedProvider:
  """A provider that replays scripted responses, in order.

  Attributes:
    name: ``scripted``.
    model: ``scripted-model``.
    calls: Every call received so far, oldest first.
  """

  name = "scripted"
  model = MODEL

  def __init__(
      self, responses: Sequence[base.RawResponse | dict[str, Any]]
  ) -> None:
    """Initialises the provider.

    Args:
      responses: What each call returns, in order. A dict stands for a
        successful call whose reply decoded to that dict.
    """
    self._responses = list(responses)
    self.calls: list[Call] = []

  def complete(
      self, system: str, prompt: str, schema: dict[str, Any]
  ) -> base.RawResponse:
    """Records the call and returns the next scripted response.

    Args:
      system: The instructions the caller sent.
      prompt: The task input the caller sent.
      schema: The schema the caller sent.

    Raises:
      AssertionError: If every scripted response has been used already.
    """
    self.calls.append(Call(system, prompt, schema))
    if len(self.calls) > len(self._responses):
      raise AssertionError(
          f"ScriptedProvider was called {len(self.calls)} times but scripted"
          f" with {len(self._responses)} responses"
      )
    response = self._responses[len(self.calls) - 1]
    if isinstance(response, base.RawResponse):
      return response
    return base.RawResponse(
        text=json.dumps(response), parsed=response, attempts=1, model=MODEL
    )
