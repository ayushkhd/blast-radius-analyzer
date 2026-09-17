"""Tests for blast_radius.llm.openai_provider."""

import types
from typing import Any

import httpx2
import openai
import pytest

from blast_radius import config
from blast_radius.llm import base
from blast_radius.llm import openai_provider

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
_SYSTEM = "You write briefs."
_PROMPT = "SECRET-PROMPT-TEXT about adm-bastion"
_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/responses")


class _FakeResponses:
  """Records each ``create`` call and replays one reply or exception."""

  def __init__(self, outcome: Any) -> None:
    self.outcome = outcome
    self.calls: list[dict[str, Any]] = []

  def create(self, **kwargs: Any) -> Any:
    self.calls.append(kwargs)
    if isinstance(self.outcome, Exception):
      raise self.outcome
    return self.outcome


class _FakeClient:
  """The slice of ``openai.OpenAI`` that the provider uses."""

  def __init__(self, outcome: Any) -> None:
    self.responses = _FakeResponses(outcome)


def _reply(
    text: str = '{"summary": "ok"}',
    *,
    refusal: bool = False,
    status: str = "completed",
) -> types.SimpleNamespace:
  part_type = "refusal" if refusal else "output_text"
  message = types.SimpleNamespace(
      type="message", content=[types.SimpleNamespace(type=part_type)]
  )
  reasoning = types.SimpleNamespace(type="reasoning")
  return types.SimpleNamespace(
      output=[reasoning, message],
      output_text="" if refusal else text,
      status=status,
      model="gpt-5.6-luna-2026-08-01",
      usage=types.SimpleNamespace(input_tokens=120, output_tokens=30),
  )


def _settings(**overrides: Any) -> config.Settings:
  return config.Settings(_env_file=None, **overrides)


def _complete(
    outcome: Any, **overrides: Any
) -> tuple[base.RawResponse, _FakeResponses]:
  fake = _FakeClient(outcome)
  client: Any = fake
  provider = openai_provider.OpenAIProvider(
      _settings(**overrides), client=client
  )
  return provider.complete(_SYSTEM, _PROMPT, _SCHEMA), fake.responses


def _status_error(
    error_class: type[openai.APIStatusError], status: int
) -> openai.APIStatusError:
  response = httpx2.Response(
      status, request=_REQUEST, headers={"x-request-id": "req_123"}
  )
  return error_class(f"echo: {_PROMPT}", response=response, body=None)


def test_request_asks_luna_for_strict_json_and_keeps_instructions_apart():
  _, responses = _complete(_reply())

  call = responses.calls[0]
  assert call["model"] == "gpt-5.6-luna"
  assert call["instructions"] == _SYSTEM
  assert call["input"] == _PROMPT
  assert call["reasoning"] == {"effort": "low"}
  assert call["store"] is False
  assert call["max_output_tokens"] == 16000
  assert call["text"]["format"] == {
      "type": "json_schema",
      "name": "blast_radius_output",
      "strict": True,
      "schema": _SCHEMA,
  }
  assert "temperature" not in call


def test_settings_override_the_model_effort_and_output_cap():
  _, responses = _complete(
      _reply(), llm_model="gpt-test", llm_effort="high", llm_max_tokens=512
  )

  call = responses.calls[0]
  assert call["model"] == "gpt-test"
  assert call["reasoning"] == {"effort": "high"}
  assert call["max_output_tokens"] == 512


def test_successful_reply_is_parsed_with_usage_and_the_answering_model():
  response, _ = _complete(_reply())

  assert response.ok
  assert response.parsed == {"summary": "ok"}
  assert response.input_tokens == 120
  assert response.output_tokens == 30
  assert response.model == "gpt-5.6-luna-2026-08-01"
  assert response.duration_s >= 0


def test_refusal_is_reported_as_a_refusal_not_a_parse_failure():
  response, _ = _complete(_reply(refusal=True))

  assert not response.ok
  assert response.kind == base.KIND_REFUSAL


@pytest.mark.parametrize(
    "text", ["", "not json", '{"summary": "cut off', "[1, 2]", '"a string"']
)
def test_unusable_text_is_a_parse_failure(text: str):
  response, _ = _complete(_reply(text, status="incomplete"))

  assert response.kind == base.KIND_PARSE
  assert response.parsed is None
  assert "incomplete" in (response.error or "") or "JSON" in (
      response.error or ""
  )


@pytest.mark.parametrize(
    "error, kind",
    [
        (openai.APITimeoutError(request=_REQUEST), base.KIND_TIMEOUT),
        (openai.APIConnectionError(request=_REQUEST), base.KIND_PROVIDER),
        (_status_error(openai.NotFoundError, 404), base.KIND_PROVIDER),
        (_status_error(openai.RateLimitError, 429), base.KIND_PROVIDER),
        (_status_error(openai.AuthenticationError, 401), base.KIND_PROVIDER),
        (openai.OpenAIError("no key"), base.KIND_PROVIDER),
    ],
)
def test_sdk_errors_become_failed_responses_and_never_raise(
    error: Exception, kind: str
):
  response, _ = _complete(error)

  assert not response.ok
  assert response.kind == kind
  assert type(error).__name__ in (response.error or "")


def test_error_text_never_echoes_the_prompt_or_the_api_body():
  response, _ = _complete(_status_error(openai.BadRequestError, 400))

  assert _PROMPT not in (response.error or "")
  assert "HTTP 400, request req_123" in (response.error or "")


def test_unknown_model_error_names_the_model():
  response, _ = _complete(
      _status_error(openai.NotFoundError, 404), llm_model="gpt-typo"
  )

  assert "'gpt-typo'" in (response.error or "")


def test_unknown_effort_is_rejected_at_construction():
  client: Any = _FakeClient(_reply())

  with pytest.raises(ValueError, match="llm_effort"):
    openai_provider.OpenAIProvider(_settings(llm_effort="max"), client=client)


def test_provider_is_unavailable_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.delenv(openai_provider.ENV_API_KEY, raising=False)

  with pytest.raises(base.ProviderUnavailableError, match="OPENAI_API_KEY"):
    openai_provider.OpenAIProvider(_settings())


def test_provider_builds_its_own_client_when_a_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setenv(openai_provider.ENV_API_KEY, "sk-test-not-a-real-key")

  provider = openai_provider.OpenAIProvider(
      _settings(llm_timeout_s=12.0, llm_max_retries=1)
  )

  assert provider.name == "openai"
  assert provider.model == "gpt-5.6-luna"
