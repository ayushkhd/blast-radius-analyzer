"""Tests for blast_radius.llm.anthropic_provider.

No test here reaches the network. Most drive the provider through a fake
client with the method shape of the SDK's. The last few run the real SDK over
a mock transport, to pin down the HTTP request that the documented parameters
turn into and the exceptions that the SDK really raises.
"""

from collections.abc import Callable
import json
import logging
import sys
import types
from typing import Any

import anthropic
from anthropic.types import beta as beta_types
import httpx2
import pytest

from blast_radius import config
from blast_radius.llm import anthropic_provider
from blast_radius.llm import base
from blast_radius.llm import prompts

_SYSTEM = "Reply with the product."
_PROMPT = "SECRET-PROMPT-TEXT about ExampleD"
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"product": {"type": "string"}},
    "required": ["product"],
    "additionalProperties": False,
}
_REQUEST = httpx2.Request("POST", "https://api.anthropic.test/v1/messages")


def _settings(**overrides: Any) -> config.Settings:
  """Returns settings that do not depend on the machine's environment.

  Args:
    **overrides: Settings to replace.
  """
  values: dict[str, Any] = {
      "llm_provider": "anthropic",
      "llm_model": "claude-opus-5",
      "llm_effort": "low",
      "llm_max_tokens": 2048,
      "llm_timeout_s": 12.5,
      "llm_max_retries": 1,
  }
  return config.Settings(**{**values, **overrides})


def _message(
    text: str | None = '{"product": "ExampleD"}', **fields: Any
) -> beta_types.BetaMessage:
  """Returns an API reply whose only content is ``text``, if any.

  Args:
    text: The reply's text block, or None for a reply with no content.
    **fields: Fields of the reply to replace.
  """
  content = [] if text is None else [{"type": "text", "text": text}]
  reply: dict[str, Any] = {
      "id": "msg_test",
      "type": "message",
      "role": "assistant",
      "model": "claude-opus-5",
      "content": content,
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 120, "output_tokens": 30},
  }
  return beta_types.BetaMessage.model_validate({**reply, **fields})


def _status_error(
    error_class: type[anthropic.APIStatusError], status: int
) -> anthropic.APIStatusError:
  """Returns an SDK error whose message and body echo the prompt."""
  response = httpx2.Response(
      status, request=_REQUEST, headers={"request-id": "req_test"}
  )
  body = {"error": {"type": "invalid_request_error", "message": _PROMPT}}
  return error_class(
      f"Error code: {status} - {body}", response=response, body=body
  )


class _FakeRawResponse:
  """What ``with_raw_response.create`` returns: a reply and a retry count."""

  def __init__(self, message: beta_types.BetaMessage, retries_taken: int):
    self._message = message
    self.retries_taken = retries_taken

  def parse(self) -> beta_types.BetaMessage:
    """Returns the reply, as the SDK's raw response does."""
    return self._message


class _FakeClient:
  """A client with the SDK's method shape that replays one outcome.

  Attributes:
    requests: The keyword arguments of every ``create`` call.
    beta: Mirrors ``client.beta.messages.with_raw_response.create``.
  """

  def __init__(
      self, outcome: beta_types.BetaMessage | Exception, retries_taken: int = 0
  ):
    self._outcome = outcome
    self._retries_taken = retries_taken
    self.requests: list[dict[str, Any]] = []
    raw = types.SimpleNamespace(create=self._create)
    messages = types.SimpleNamespace(with_raw_response=raw)
    self.beta = types.SimpleNamespace(messages=messages)

  def _create(self, **request: Any) -> _FakeRawResponse:
    self.requests.append(request)
    if isinstance(self._outcome, Exception):
      raise self._outcome
    return _FakeRawResponse(self._outcome, self._retries_taken)


def _complete(
    outcome: beta_types.BetaMessage | Exception, **settings: Any
) -> tuple[base.RawResponse, _FakeClient]:
  """Runs one call against a fake client and returns both.

  Args:
    outcome: What the client returns or raises.
    **settings: Settings to replace.
  """
  client = _FakeClient(outcome)
  provider = anthropic_provider.AnthropicProvider(
      _settings(**settings), client=client
  )
  return provider.complete(_SYSTEM, _PROMPT, _SCHEMA), client


def test_complete_asks_the_configured_model_with_the_output_cap():
  _, client = _complete(_message(), llm_model="claude-test", llm_max_tokens=512)

  assert client.requests[0]["model"] == "claude-test"
  assert client.requests[0]["max_tokens"] == 512


def test_complete_sends_effort_and_schema_in_output_config():
  _, client = _complete(_message(), llm_effort="medium")

  assert client.requests[0]["output_config"] == {
      "effort": "medium",
      "format": {"type": "json_schema", "schema": _SCHEMA},
  }


def test_complete_sends_system_at_top_level_and_the_prompt_as_one_user_turn():
  _, client = _complete(_message())

  assert client.requests[0]["system"] == _SYSTEM
  assert client.requests[0]["messages"] == [
      {"role": "user", "content": _PROMPT}
  ]


def test_complete_opts_into_the_default_server_side_refusal_fallback():
  _, client = _complete(_message())

  assert client.requests[0]["fallbacks"] == "default"
  assert client.requests[0]["betas"] == ["server-side-fallback-2026-07-01"]


def test_complete_sends_nothing_that_the_model_rejects():
  _, client = _complete(_message())

  assert set(client.requests[0]) == {
      "model",
      "max_tokens",
      "system",
      "messages",
      "output_config",
      "fallbacks",
      "betas",
  }
  for rejected in ("temperature", "top_p", "top_k", "thinking"):
    assert rejected not in client.requests[0]


def test_complete_returns_the_decoded_object_with_usage_and_timing():
  response, _ = _complete(_message())

  assert response.ok
  assert response.parsed == {"product": "ExampleD"}
  assert response.text == '{"product": "ExampleD"}'
  assert response.kind is None
  assert response.error is None
  assert response.input_tokens == 120
  assert response.output_tokens == 30
  assert response.model == "claude-opus-5"
  assert response.duration_s >= 0


def test_complete_counts_the_attempts_the_sdk_reports():
  client = _FakeClient(_message(), retries_taken=2)
  provider = anthropic_provider.AnthropicProvider(_settings(), client=client)

  response = provider.complete(_SYSTEM, _PROMPT, _SCHEMA)

  assert response.attempts == 3


def test_complete_records_the_fallback_model_that_answered():
  response, _ = _complete(_message(model="claude-opus-4-8"))

  assert response.ok
  assert response.model == "claude-opus-4-8"


def test_complete_reads_the_text_block_behind_fallback_and_thinking_blocks():
  content = [
      {
          "type": "fallback",
          "from": {"model": "claude-opus-5"},
          "to": {"model": "claude-opus-4-8"},
          "trigger": {"type": "refusal", "category": "cyber"},
      },
      {"type": "thinking", "thinking": "", "signature": "sig"},
      {"type": "text", "text": '{"product": "ExampleD"}'},
  ]

  response, _ = _complete(_message(content=content))

  assert response.parsed == {"product": "ExampleD"}


def test_complete_refusal_is_reported_without_reading_the_partial_content():
  refused = _message(
      '{"product": "half an ans',
      stop_reason="refusal",
      stop_details={"type": "refusal", "category": "cyber"},
  )

  response, _ = _complete(refused)

  assert not response.ok
  assert response.kind == base.KIND_REFUSAL
  assert response.error == (
      "the model declined the request (refusal category: cyber)"
  )
  assert response.parsed is None
  assert response.text == ""
  assert response.attempts == 1
  assert response.model == "claude-opus-5"


def test_complete_refusal_without_details_is_still_a_refusal():
  response, _ = _complete(_message(None, stop_reason="refusal"))

  assert response.kind == base.KIND_REFUSAL
  assert response.error == (
      "the model declined the request (refusal category: none)"
  )


def _class_name(error: Exception) -> str:
  """Returns the test id of an SDK error: the name of its class."""
  return type(error).__name__


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(request=_REQUEST),
        _status_error(anthropic.DeadlineExceededError, 504),
    ],
    ids=_class_name,
)
def test_complete_sdk_timeout_is_reported_as_a_timeout(error: Exception):
  response, _ = _complete(error)

  assert not response.ok
  assert response.kind == base.KIND_TIMEOUT
  assert response.parsed is None
  assert response.duration_s >= 0


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APIConnectionError(request=_REQUEST),
        _status_error(anthropic.BadRequestError, 400),
        _status_error(anthropic.AuthenticationError, 401),
        _status_error(anthropic.PermissionDeniedError, 403),
        _status_error(anthropic.NotFoundError, 404),
        _status_error(anthropic.RateLimitError, 429),
        _status_error(anthropic.InternalServerError, 500),
        _status_error(anthropic.OverloadedError, 529),
        anthropic.CredentialsError(_PROMPT),
    ],
    ids=_class_name,
)
def test_complete_other_sdk_failure_is_reported_as_a_provider_failure(
    error: Exception,
):
  response, _ = _complete(error)

  assert not response.ok
  assert response.kind == base.KIND_PROVIDER
  assert response.parsed is None
  assert response.error is not None
  assert response.error.startswith(f"{type(error).__name__}: ")
  assert _PROMPT not in response.error


def test_complete_error_never_quotes_the_prompt_or_the_api_reply():
  error = _status_error(anthropic.BadRequestError, 400)

  response, _ = _complete(error)

  assert _PROMPT in str(error)
  assert response.error == (
      "BadRequestError: the API answered with an error (HTTP 400, request"
      " req_test)"
  )


def test_complete_timeout_error_names_the_configured_limit():
  response, _ = _complete(anthropic.APITimeoutError(request=_REQUEST))

  assert response.error == (
      "APITimeoutError: no reply within 12.5 s per attempt"
  )


def test_complete_unknown_model_error_names_the_model():
  error = _status_error(anthropic.NotFoundError, 404)

  response, _ = _complete(error, llm_model="claude-typo")

  assert response.error == (
      "NotFoundError: model 'claude-typo' is unknown or not allowed (HTTP 404,"
      " request req_test)"
  )


def test_complete_failure_leaves_attempts_unknown():
  response, _ = _complete(anthropic.APIConnectionError(request=_REQUEST))

  assert response.attempts == 0


@pytest.mark.parametrize(
    "text, error",
    [
        (
            "Here is the product: ExampleD",
            "the reply is not valid JSON (stop reason: end_turn)",
        ),
        ('["ExampleD"]', "the reply is JSON but not an object (list)"),
        ('"ExampleD"', "the reply is JSON but not an object (str)"),
    ],
    ids=["prose", "array", "string"],
)
def test_complete_reply_that_is_not_a_json_object_is_a_parse_failure(
    text: str, error: str
):
  response, _ = _complete(_message(text))

  assert not response.ok
  assert response.kind == base.KIND_PARSE
  assert response.error == error
  assert response.parsed is None
  assert response.text == text


def test_complete_reply_cut_off_at_the_output_cap_is_a_parse_failure():
  truncated = _message('{"product": "Exam', stop_reason="max_tokens")

  response, _ = _complete(truncated)

  assert response.kind == base.KIND_PARSE
  assert response.error == (
      "the reply is not valid JSON (stop reason: max_tokens)"
  )


def test_complete_reply_without_a_text_block_is_a_parse_failure():
  response, _ = _complete(_message(None))

  assert response.kind == base.KIND_PARSE
  assert response.error == "the reply has no text (stop reason: end_turn)"


def test_complete_logs_a_failure_without_the_prompt(
    caplog: pytest.LogCaptureFixture,
):
  with caplog.at_level(logging.WARNING):
    _complete(_status_error(anthropic.RateLimitError, 429))

  assert len(caplog.records) == 1
  assert "RateLimitError" in caplog.text
  assert "provider" in caplog.text
  assert _PROMPT not in caplog.text


def test_init_without_a_client_builds_one_with_the_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
):
  built: list[dict[str, Any]] = []

  class _SdkClient:
    api_key = "set"
    auth_token = None
    credentials = None

    def __init__(self, **options: Any):
      built.append(options)

  monkeypatch.setattr(anthropic, "Anthropic", _SdkClient)

  provider = anthropic_provider.AnthropicProvider(_settings())

  assert built == [{"timeout": 12.5, "max_retries": 1}]
  assert provider.name == "anthropic"
  assert provider.model == "claude-opus-5"


def test_init_without_the_package_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setitem(sys.modules, "anthropic", None)

  with pytest.raises(base.ProviderUnavailableError, match="not installed"):
    anthropic_provider.AnthropicProvider(_settings())


def test_init_without_a_credential_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
  class _SdkClient:
    api_key = None
    auth_token = None
    credentials = None

    def __init__(self, **options: Any):
      del options

  monkeypatch.setattr(anthropic, "Anthropic", _SdkClient)

  with pytest.raises(base.ProviderUnavailableError, match="ANTHROPIC_API_KEY"):
    anthropic_provider.AnthropicProvider(_settings())


def test_init_with_a_broken_credential_profile_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
  def _broken(**options: Any) -> None:
    del options
    raise anthropic.CredentialsError("cannot read /home/someone/secret.json")

  monkeypatch.setattr(anthropic, "Anthropic", _broken)

  with pytest.raises(base.ProviderUnavailableError) as raised:
    anthropic_provider.AnthropicProvider(_settings())

  assert "CredentialsError" in str(raised.value)
  assert "secret.json" not in str(raised.value)


def test_init_with_an_unknown_effort_is_rejected():
  with pytest.raises(ValueError, match="llm_effort"):
    anthropic_provider.AnthropicProvider(
        _settings(llm_effort="hgih"), client=_FakeClient(_message())
    )


def test_create_with_the_provider_set_to_none_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setitem(sys.modules, "anthropic", None)

  assert anthropic_provider.create(_settings(llm_provider="none")) is None


def test_create_when_unavailable_returns_none_and_logs_the_reason_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
  monkeypatch.setitem(sys.modules, "anthropic", None)

  with caplog.at_level(logging.WARNING):
    provider = anthropic_provider.create(_settings())

  assert provider is None
  assert len(caplog.records) == 1
  assert caplog.records[0].levelname == "WARNING"
  assert "not installed" in caplog.text


def test_create_with_a_credential_returns_the_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

  provider = anthropic_provider.create(_settings(llm_model="claude-test"))

  assert isinstance(provider, anthropic_provider.AnthropicProvider)
  assert provider.name == "anthropic"
  assert provider.model == "claude-test"


@pytest.mark.parametrize(
    "schema",
    [prompts.PARSE_SCHEMA, prompts.WRITE_SCHEMA],
    ids=["parse", "write"],
)
def test_prompt_schemas_pass_through_the_sdk_transform_unchanged(
    schema: dict[str, Any],
):
  # The SDK rewrites what structured outputs do not support; a schema it
  # leaves alone uses nothing unsupported.
  assert anthropic.transform_schema(schema) == schema


def _sdk_provider(
    respond: Callable[[httpx2.Request], httpx2.Response],
) -> anthropic_provider.AnthropicProvider:
  """Returns a provider on the real SDK, with ``respond`` as its network.

  Args:
    respond: Answers each HTTP request the SDK sends, or raises.
  """
  client = anthropic.Anthropic(
      api_key="test-key",
      base_url="https://api.anthropic.test",
      max_retries=0,
      http_client=anthropic.DefaultHttpxClient(
          transport=httpx2.MockTransport(respond)
      ),
  )
  return anthropic_provider.AnthropicProvider(_settings(), client=client)


def test_complete_through_the_real_sdk_sends_the_documented_request():
  seen: list[httpx2.Request] = []

  def _respond(request: httpx2.Request) -> httpx2.Response:
    seen.append(request)
    return httpx2.Response(200, json=_message().to_dict())

  response = _sdk_provider(_respond).complete(_SYSTEM, _PROMPT, _SCHEMA)

  assert response.ok
  assert response.parsed == {"product": "ExampleD"}
  assert response.attempts == 1
  assert len(seen) == 1
  assert seen[0].url.path == "/v1/messages"
  assert seen[0].headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
  assert json.loads(seen[0].content) == {
      "model": "claude-opus-5",
      "max_tokens": 2048,
      "system": _SYSTEM,
      "messages": [{"role": "user", "content": _PROMPT}],
      "output_config": {
          "effort": "low",
          "format": {"type": "json_schema", "schema": _SCHEMA},
      },
      "fallbacks": "default",
  }


def test_complete_through_the_real_sdk_reports_a_transport_timeout():
  def _time_out(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("no bytes in time", request=request)

  response = _sdk_provider(_time_out).complete(_SYSTEM, _PROMPT, _SCHEMA)

  assert response.kind == base.KIND_TIMEOUT
  assert response.error == (
      "APITimeoutError: no reply within 12.5 s per attempt"
  )


def test_complete_through_the_real_sdk_keeps_an_error_body_out_of_the_error():
  def _reject(request: httpx2.Request) -> httpx2.Response:
    del request
    body = {"type": "error", "error": {"type": "x", "message": _PROMPT}}
    return httpx2.Response(401, json=body, headers={"request-id": "req_wire"})

  response = _sdk_provider(_reject).complete(_SYSTEM, _PROMPT, _SCHEMA)

  assert response.kind == base.KIND_PROVIDER
  assert response.error == (
      "AuthenticationError: the API answered with an error (HTTP 401, request"
      " req_wire)"
  )
