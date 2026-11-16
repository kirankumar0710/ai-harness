"""The provider abstraction: one narrow doorway to any model.

A harness talks to a model through a single call: send the conversation,
get back text and/or requests to call tools. This module defines that
contract once, independent of any vendor:

* ``ProviderConfig`` - the settings every vendor needs (API key, base
  URL, model) plus shared tuning knobs. Each vendor subclasses it; its
  default base URL, model and key variable names come from that
  vendor's section in ``providers.ini``.
* ``Provider`` - an abstract base class. ``complete()`` is a template
  method: subclasses only translate the neutral request into their wire
  format and parse the reply back. Authentication headers, retries and
  error handling are shared.
* Neutral message types (``Message``, ``ToolCall``, ``Completion``) -
  plain dicts, so the loop and the session log stay JSON-serialisable.

Design rules:

* Vendor quirks live in the vendor subclass and nowhere else.
* Every tool call carries an ``id``; every tool result quotes it back.
* Sampling parameters are omitted unless explicitly configured, so each
  model runs on its vendor's recommended defaults.
* Standard library only.
"""

from __future__ import annotations

import configparser
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict

# --------------------------------------------------------------------------
# Neutral message format
# --------------------------------------------------------------------------

GENERATED_ID_PREFIX = "odysseus-"


class _ToolCallRequired(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


class ToolCall(_ToolCallRequired, total=False):
    """A request from the model to run one tool.

    ``signature`` is an opaque reasoning token some vendors (Gemini 3)
    require to be echoed back with the call on later turns. ``error`` is
    set when the vendor returned arguments that could not be parsed.
    """

    signature: str | None
    error: str


class Message(TypedDict, total=False):
    """One entry in the conversation.

    user      -> {"role", "text"}
    assistant -> {"role", "text", "tool_calls"}
    tool      -> {"role", "tool_call_id", "name", "text", "is_error"}
    """

    role: Literal["user", "assistant", "tool"]
    text: str
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str
    is_error: bool


class Usage(TypedDict):
    input: int
    output: int


class _CompletionRequired(TypedDict):
    text: str
    tool_calls: list[ToolCall]
    usage: Usage


class Completion(_CompletionRequired, total=False):
    """What every provider returns, whatever its wire format.

    ``truncated`` is True when the reply stopped because it hit the
    output-token limit: its text, or a tool call's arguments, may be
    cut off mid-way. Every vendor reports this; every provider sets it.
    """

    truncated: bool


# A tool schema is a JSON Schema function declaration:
# {"name": str, "description": str, "parameters": {JSON Schema object}}
ToolSchema = Mapping[str, Any]


def new_call_id() -> str:
    """Return a locally generated tool-call id.

    Used when a vendor does not issue ids itself. The prefix lets a
    provider recognise ids it must not send back to the vendor.
    """
    return f"{GENERATED_ID_PREFIX}{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ConfigError(RuntimeError):
    """Raised when a provider cannot be configured (e.g. no API key)."""


class ProviderError(RuntimeError):
    """Raised when a model call fails after all retries."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


DEFAULTS_FILE = Path(__file__).with_name("providers.ini")
CONFIG_PATH_ENV_VAR = "ODYSSEUS_CONFIG"


@dataclass(frozen=True)
class VendorDefaults:
    """One vendor's section of the ini files."""

    base_url: str
    model: str
    api_key_env: tuple[str, ...]


def load_vendor_defaults(provider_name: str) -> VendorDefaults:
    """Read a vendor's defaults from the ini files.

    The packaged ``providers.ini`` is read first; the file named by
    ``ODYSSEUS_CONFIG``, if set, is layered on top and overrides it key
    by key. The files are re-read on every call, so edits take effect
    without restarting.
    """
    parser = configparser.ConfigParser()
    if not parser.read(DEFAULTS_FILE, encoding="utf-8"):
        raise ConfigError(f"Missing provider defaults file: {DEFAULTS_FILE}")
    user_file = os.environ.get(CONFIG_PATH_ENV_VAR)
    if user_file and not parser.read(user_file, encoding="utf-8"):
        raise ConfigError(
            f"{CONFIG_PATH_ENV_VAR} points to an unreadable file: "
            f"{user_file}")
    if not parser.has_section(provider_name):
        raise ConfigError(
            f"No [{provider_name}] section in the provider config files")
    section = parser[provider_name]
    return VendorDefaults(
        base_url=section.get("base_url", "").strip(),
        model=section.get("model", "").strip(),
        api_key_env=tuple(
            name.strip() for name in section.get("api_key_env", "").split(",")
            if name.strip()),
    )


@dataclass(frozen=True)
class ProviderConfig:
    """Settings shared by every vendor.

    Subclasses set the class-level defaults below and may add fields of
    their own. Build one directly, or with ``from_env()``.
    """

    api_key: str
    model: str
    base_url: str
    timeout: float = 600.0
    max_attempts: int = 5
    temperature: float | None = None
    max_output_tokens: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    # The vendor's section name in providers.ini; everything else about
    # the vendor's defaults lives in that file, not in code.
    PROVIDER_NAME: ClassVar[str] = ""

    def __post_init__(self) -> None:
        # Normalise once so subclasses can join paths with a single "/".
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.api_key:
            raise ConfigError(f"{self.PROVIDER_NAME}: api_key is empty")
        if not self.model:
            raise ConfigError(f"{self.PROVIDER_NAME}: model is empty")
        if self.max_attempts < 1:
            raise ConfigError("max_attempts must be at least 1")

    @classmethod
    def from_env(cls, **overrides: Any) -> ProviderConfig:
        """Build a config from environment variables plus overrides.

        Resolution order, highest first:
          1. keyword overrides passed to this method (``None`` is ignored)
          2. the vendor's own key variables, e.g. ``GEMINI_API_KEY``
          3. ``ODYSSEUS_API_KEY`` / ``ODYSSEUS_MODEL`` /
             ``ODYSSEUS_BASE_URL``
          4. the vendor's section in the ini files (see
             ``load_vendor_defaults``)
        """
        defaults = load_vendor_defaults(cls.PROVIDER_NAME)
        api_key = next(
            (os.environ[name] for name in defaults.api_key_env
             if os.environ.get(name)),
            os.environ.get("ODYSSEUS_API_KEY", ""),
        )
        values: dict[str, Any] = {
            "api_key": api_key,
            "model": os.environ.get("ODYSSEUS_MODEL") or defaults.model,
            "base_url": (os.environ.get("ODYSSEUS_BASE_URL")
                         or defaults.base_url),
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        if not values["api_key"]:
            names = ", ".join((*defaults.api_key_env, "ODYSSEUS_API_KEY"))
            raise ConfigError(
                f"No API key for {cls.PROVIDER_NAME}: set one of {names}")
        return cls(**values)


# --------------------------------------------------------------------------
# HTTP transport (shared by every vendor)
# --------------------------------------------------------------------------

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 60.0

Transport = Callable[..., dict[str, Any]]


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, so parallel agents do not retry
    in lockstep."""
    base = min(MAX_BACKOFF_SECONDS, 2.0 ** attempt)
    return base * random.uniform(0.5, 1.0)


def _retry_after(error: urllib.error.HTTPError) -> float | None:
    """Honour a numeric ``Retry-After`` header when the server sends one."""
    value = error.headers.get("Retry-After") if error.headers else None
    try:
        return min(MAX_BACKOFF_SECONDS, float(value)) if value else None
    except ValueError:
        return None


def post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
    max_attempts: int,
) -> dict[str, Any]:
    """POST a JSON body and return the decoded JSON reply.

    Retries rate limits, server errors and network failures. Any other
    HTTP error is raised immediately as ``ProviderError``.
    """
    payload = json.dumps(body).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            url, data=payload, headers=all_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            if error.code not in RETRYABLE_STATUS or attempt == max_attempts:
                raise ProviderError(
                    f"HTTP {error.code}: {detail}", status=error.code
                ) from error
            delay = _retry_after(error) or _backoff_delay(attempt)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == max_attempts:
                raise ProviderError(f"Network error: {error}") from error
            delay = _backoff_delay(attempt)
        time.sleep(delay)
    raise ProviderError("Model call failed after retries")  # unreachable


# --------------------------------------------------------------------------
# The provider interface
# --------------------------------------------------------------------------


class Provider(ABC):
    """Abstract model provider.

    ``complete()`` is the only method the rest of the harness calls.
    Subclasses implement the two vendor-specific steps:

    * ``_build_request`` - neutral request -> (url, headers, body)
    * ``_parse_response`` - vendor JSON -> ``Completion``
    """

    config_class: ClassVar[type[ProviderConfig]] = ProviderConfig

    def __init__(self, config: ProviderConfig,
                 transport: Transport = post_json) -> None:
        if not isinstance(config, self.config_class):
            raise ConfigError(
                f"{type(self).__name__} needs a "
                f"{self.config_class.__name__}, got {type(config).__name__}")
        self.config = config
        self._transport = transport

    @property
    def model(self) -> str:
        return self.config.model

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] = (),
        *,
        allow_tool_calls: bool = True,
    ) -> Completion:
        """Make one model call.

        ``allow_tool_calls=False`` keeps the tool definitions in the
        request (some vendors reject a history containing tool calls
        without them) but forbids the model from calling any.
        """
        url, headers, body = self._build_request(
            system, messages, tools, allow_tool_calls)
        raw = self._transport(
            url, body, {**headers, **self.config.extra_headers},
            self.config.timeout, self.config.max_attempts)
        return self._parse_response(raw)

    @abstractmethod
    def _build_request(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        allow_tool_calls: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Translate a neutral request into (url, headers, json_body)."""

    @abstractmethod
    def _parse_response(self, raw: Mapping[str, Any]) -> Completion:
        """Translate the vendor's JSON reply into a ``Completion``."""
