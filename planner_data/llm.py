# Copyright 2026 The LLaMA Factory Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: RUF001, RUF002, RUF003
"""Thin DashScope Qwen text-generation helper shared across datagen.

Centralises the ``qwen-plus`` calls used to generate raw cases, rewrite queries
into an implicit phrasing, inject coherent filler words, and judge generated
cases in the eval module. Keeps prompt plumbing (system/user messages, robust
JSON extraction, temperature/seed) in one place so callers focus on prompts.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Any


DEFAULT_MODEL = "qwen-max"

# --- DashScope call timeout / retry policy --------------------------------- #
# A single ``qwen`` HTTP call uses a long read timeout so slow generations have
# time to complete. The reconnect/retry mechanism is disabled by default
# (``_LLM_MAX_ATTEMPTS == 1``): a failure raises immediately and the caller
# (raw / eval) skips just that round/tool without crashing the whole job. Both
# are overridable via env for anyone who wants retries back.
_LLM_REQUEST_TIMEOUT_S = int(os.environ.get("BENCHMARK_LLM_TIMEOUT_S", "300"))
_LLM_MAX_ATTEMPTS = max(1, int(os.environ.get("BENCHMARK_LLM_MAX_ATTEMPTS", "1")))
_LLM_BACKOFF_BASE_S = 2.0
_LLM_BACKOFF_CAP_S = 30.0
# HTTP statuses worth retrying (timeout / rate-limit / transient server errors).
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

# DashScope hosts whose GTM DNS chain currently ships a *bogus* DNSSEC signature,
# so every validating resolver (corporate, 223.5.5.5, 8.8.8.8, …) rejects the A
# lookup with SERVFAIL — surfacing as ``[Errno -3] Temporary failure in name
# resolution``. We resolve these via DoH with checking-disabled (cd=1) as a
# fallback. See :func:`_install_dashscope_doh_fallback`.
_DOH_HOSTS = ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com")
# IP-literal DoH endpoints (no bootstrap DNS needed). ``cd=1`` disables DNSSEC
# validation so the bogus GTM signature no longer causes SERVFAIL.
_DOH_ENDPOINTS = ("https://223.5.5.5/resolve", "https://8.8.8.8/resolve")
_DOH_TTL_S = 300.0
_DOH_TIMEOUT_S = 5.0
_doh_cache: dict[str, tuple[list[str], float]] = {}
_doh_lock = threading.Lock()


def _is_doh_host(host: Any) -> bool:
    h = str(host or "").rstrip(".").lower()
    return h in _DOH_HOSTS


def _doh_lookup(host: str) -> list[str]:
    """Resolve ``host`` A records via DoH (cd=1), cached for ``_DOH_TTL_S``.

    Fail-open: returns ``[]`` when every endpoint is unreachable so the caller
    can re-raise the original resolution error.
    """
    now = time.monotonic()
    with _doh_lock:
        hit = _doh_cache.get(host)
        if hit and now < hit[1]:
            return hit[0]

    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # endpoint is an IP; we only need the A list
    for base in _DOH_ENDPOINTS:
        try:
            url = f"{base}?name={host}&type=A&cd=1"
            req = urllib.request.Request(
                url, headers={"accept": "application/dns-json"}
            )
            with urllib.request.urlopen(
                req, timeout=_DOH_TIMEOUT_S, context=ctx
            ) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - any endpoint failure just tries the next
            continue
        ips = [
            str(a.get("data"))
            for a in (doc.get("Answer") or [])
            if a.get("type") == 1 and a.get("data")  # type 1 == A record
        ]
        if ips:
            with _doh_lock:
                _doh_cache[host] = (ips, now + _DOH_TTL_S)
            return ips
    return []


def _install_dashscope_doh_fallback() -> None:
    """Make ``socket.getaddrinfo`` fall back to DoH for DashScope on SERVFAIL.

    Only triggers when the normal resolver raises ``gaierror`` for a DashScope
    host; every other lookup (and every host with working DNS) is untouched.
    """
    import socket

    original = socket.getaddrinfo

    def _patched(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        try:
            return original(host, port, family, type, proto, flags)
        except socket.gaierror:
            if not _is_doh_host(host):
                raise
            ips = _doh_lookup(str(host).rstrip("."))
            if not ips:
                raise
            try:
                p = int(port)
            except (TypeError, ValueError):
                p = 443
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (ip, p),
                )
                for ip in ips
            ]

    socket.getaddrinfo = _patched


def _prefer_ipv4_when_no_ipv6_route() -> None:
    """Pin outbound HTTP to IPv4 when the host has no usable IPv6 route.

    DashScope resolves via Aliyun's GTM smart DNS, which intermittently hands
    back AAAA (IPv6) records. On hosts without an IPv6 default route the HTTP
    stack (requests/urllib3) then fails with ``[Errno 101] Network is
    unreachable`` because—unlike curl—it does not fall back to IPv4. We probe
    IPv6 routability once and, if absent, force ``getaddrinfo`` selection to
    IPv4 for this process. IPv6-capable hosts are left untouched.
    """
    import socket

    def _has_ipv6_route() -> bool:
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        except OSError:
            return False
        try:
            # UDP connect sends no packet; it only checks that a route exists.
            s.connect(("2400:3200::1", 53))  # Aliyun public IPv6 DNS
            return True
        except OSError:
            return False
        finally:
            s.close()

    if _has_ipv6_route():
        return
    try:
        import urllib3.util.connection as _u

        _u.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    except Exception:
        pass


_prefer_ipv4_when_no_ipv6_route()
_install_dashscope_doh_fallback()


def _is_transient_exc(exc: BaseException) -> bool:
    """True for network blips worth retrying (timeouts / conn resets / DNS)."""
    try:
        import requests

        if isinstance(
            exc,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ),
        ):
            return True
    except Exception:  # noqa: BLE001 - requests always present, be defensive
        pass
    # Low-level socket / OS errors (incl. name-resolution) are transient too.
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _sleep_backoff(attempt: int) -> None:
    delay = min(_LLM_BACKOFF_BASE_S * (2**attempt), _LLM_BACKOFF_CAP_S)
    time.sleep(delay + random.uniform(0.0, 1.0))


def _call_with_retry(*, api_key: str | None, **kwargs: Any) -> Any:
    """Call ``dashscope.Generation.call`` with a short timeout + retries.

    Reconnect semantics: a stuck call fails fast (``request_timeout``) and any
    transient failure (timeout / connection / rate-limit / 5xx) is retried with
    bounded exponential backoff. Non-transient errors (auth / bad params) raise
    immediately.
    """
    import dashscope

    if api_key:
        dashscope.api_key = api_key
    last_err: Exception | None = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        try:
            resp = dashscope.Generation.call(
                request_timeout=_LLM_REQUEST_TIMEOUT_S, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - classify then retry/re-raise
            if not _is_transient_exc(exc) or attempt + 1 >= _LLM_MAX_ATTEMPTS:
                raise
            last_err = exc
            _sleep_backoff(attempt)
            continue
        status = getattr(resp, "status_code", 200)
        if status == 200:
            return resp
        msg = getattr(resp, "message", resp)
        if status in _RETRYABLE_STATUS and attempt + 1 < _LLM_MAX_ATTEMPTS:
            last_err = RuntimeError(f"LLM call failed (status {status}): {msg}")
            _sleep_backoff(attempt)
            continue
        raise RuntimeError(f"LLM call failed (status {status}): {msg}")
    raise last_err or RuntimeError("LLM call failed after retries")


def chat(
    prompt: str,
    *,
    api_key: str | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.9,
    seed: int | None = None,
) -> str:
    """Single-turn Qwen completion returning the raw text content."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "result_format": "message",
        "temperature": max(0.0, min(2.0, float(temperature))),
    }
    if seed is not None:
        kwargs["seed"] = int(seed) & 0x7FFFFFFF
    resp = _call_with_retry(api_key=api_key, **kwargs)
    return str(resp["output"]["choices"][0]["message"]["content"]).strip()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _first_json_span(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json(text: str) -> Any:
    """Extract the first JSON value (object or array) from an LLM reply."""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to scanning for a balanced object/array span.
    for opener, closer in (("[", "]"), ("{", "}")):
        span = _first_json_span(cleaned, opener, closer)
        if span:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON found in LLM reply")


def chat_json(
    prompt: str,
    *,
    api_key: str | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.9,
    seed: int | None = None,
) -> Any:
    """Like :func:`chat` but parse the reply as JSON (object or array)."""
    text = chat(
        prompt,
        api_key=api_key,
        system=system,
        model=model,
        temperature=temperature,
        seed=seed,
    )
    return parse_json(text)
