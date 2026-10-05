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

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, Self


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _get_field(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _format_exception(error: BaseException) -> str:
    return f"{type(error).__name__}: request failed"


def _validate_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _load_openai_client() -> type[Any]:
    from openai import OpenAI

    return OpenAI


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    system: str
    user: str


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str | None
    latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    error: str | None


class CompletionBackend(Protocol):
    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...

    def close(self) -> None: ...


class OpenAICompatibleBackend:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        concurrency: int = 8,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        _validate_positive_int(concurrency, field_name="concurrency")
        _validate_positive_int(max_tokens, field_name="max_tokens")
        if not model:
            raise ValueError("model is required")
        if not base_url:
            raise ValueError("base_url is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._model = model
        self._concurrency = concurrency
        self._max_tokens = max_tokens
        self._client = client or _load_openai_client()(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
        )
        self._closed = False

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        started_ms = _now_ms()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                temperature=0,
                max_tokens=self._max_tokens,
            )
            choices = _get_field(response, "choices")
            if not choices:
                raise ValueError("missing completion choice")

            message = _get_field(choices[0], "message")
            content = _get_field(message, "content")
            if not isinstance(content, str) or not content:
                raise ValueError("empty completion content")

            usage = _get_field(response, "usage")
            return GenerationResult(
                request_id=request.request_id,
                text=content,
                latency_ms=_now_ms() - started_ms,
                prompt_tokens=_get_field(usage, "prompt_tokens") if usage is not None else None,
                output_tokens=_get_field(usage, "completion_tokens") if usage is not None else None,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            return GenerationResult(
                request_id=request.request_id,
                text=None,
                latency_ms=_now_ms() - started_ms,
                prompt_tokens=None,
                output_tokens=None,
                error=_format_exception(exc),
            )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []

        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            return list(executor.map(self._generate_one, requests))

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True


class SshTunnel:
    def __init__(
        self,
        host: str,
        user: str = "root",
        ssh_port: int = 1024,
        local_port: int = 8001,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8001,
    ) -> None:
        for field_name, value in (("ssh_port", ssh_port), ("local_port", local_port), ("remote_port", remote_port)):
            _validate_positive_int(value, field_name=field_name)
        if not host:
            raise ValueError("host is required")
        if not user:
            raise ValueError("user is required")
        if not remote_host:
            raise ValueError("remote_host is required")

        self._host = host
        self._user = user
        self._ssh_port = ssh_port
        self._local_port = local_port
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._process: subprocess.Popen[Any] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def command(self) -> list[str]:
        return [
            "ssh",
            "-N",
            "-p",
            str(self._ssh_port),
            "-L",
            f"{self._local_port}:{self._remote_host}:{self._remote_port}",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            f"{self._user}@{self._host}",
        ]

    def start(self) -> None:
        if self.is_running:
            return
        self._process = subprocess.Popen(self.command)

    def ensure_running(self) -> None:
        if not self.is_running:
            self.start()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
