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
from types import SimpleNamespace

import pytest

from planner_val.backends import GenerationRequest, OpenAICompatibleBackend, SshTunnel


class _FakeCompletions:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][1]["content"]
        outcome = self._outcomes[user]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeOpenAIClient:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self.completions = _FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float] = []

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminate_calls += 1

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        return 0

    def kill(self) -> None:
        self.kill_calls += 1


def _response(text: str, prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_openai_backend_uses_one_compatible_chat_api_and_preserves_order() -> None:
    client = _FakeOpenAIClient(
        {
            "first": _response("one", 7, 2),
            "second": _response("two", 9, 3),
        }
    )
    backend = OpenAICompatibleBackend(
        model="served-model",
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        concurrency=2,
        max_tokens=55,
        client=client,
    )

    results = backend.generate(
        [
            GenerationRequest(request_id="r1", system="system-1", user="first"),
            GenerationRequest(request_id="r2", system="system-2", user="second"),
        ]
    )

    assert [result.request_id for result in results] == ["r1", "r2"]
    assert [result.text for result in results] == ["one", "two"]
    assert [result.prompt_tokens for result in results] == [7, 9]
    assert [result.output_tokens for result in results] == [2, 3]
    calls_by_user = {call["messages"][1]["content"]: call for call in client.completions.calls}
    assert calls_by_user == {
        "first": {
            "model": "served-model",
            "messages": [
                {"role": "system", "content": "system-1"},
                {"role": "user", "content": "first"},
            ],
            "temperature": 0,
            "max_tokens": 55,
        },
        "second": {
            "model": "served-model",
            "messages": [
                {"role": "system", "content": "system-2"},
                {"role": "user", "content": "second"},
            ],
            "temperature": 0,
            "max_tokens": 55,
        },
    }


def test_openai_backend_returns_one_sanitized_result_for_each_failure() -> None:
    client = _FakeOpenAIClient(
        {
            "broken": RuntimeError("api_key=secret provider payload"),
            "empty": _response("", 3, 0),
        }
    )
    backend = OpenAICompatibleBackend(
        model="qwen-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="secret",
        client=client,
    )

    results = backend.generate(
        [
            GenerationRequest(request_id="r1", system="system", user="broken"),
            GenerationRequest(request_id="r2", system="system", user="empty"),
        ]
    )

    assert [result.request_id for result in results] == ["r1", "r2"]
    assert all(result.text is None for result in results)
    assert all(result.error for result in results)
    assert all("secret" not in result.error for result in results if result.error)

    backend.close()
    backend.close()
    assert client.close_calls == 1


def test_ssh_tunnel_uses_requested_forward_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()
    popen_calls: list[list[str]] = []

    def fake_popen(command: list[str]):
        popen_calls.append(command)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    tunnel = SshTunnel(host="8.130.18.173", user="root", ssh_port=1024, local_port=8001, remote_port=8001)

    with tunnel:
        assert tunnel.is_running is True

    assert popen_calls == [
        [
            "ssh",
            "-N",
            "-p",
            "1024",
            "-L",
            "8001:127.0.0.1:8001",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "root@8.130.18.173",
        ]
    ]
    assert process.terminate_calls == 1
    assert process.wait_calls == [5.0]


def test_ssh_tunnel_restarts_a_dead_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _FakeProcess(return_code=255)
    second = _FakeProcess()
    processes = iter([first, second])
    monkeypatch.setattr(subprocess, "Popen", lambda command: next(processes))
    tunnel = SshTunnel(host="8.130.18.173")

    tunnel.start()
    tunnel.ensure_running()

    assert tunnel.is_running is True
    tunnel.close()
    assert second.terminate_calls == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"concurrency": 0}, "concurrency"),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_openai_backend_rejects_invalid_limits(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleBackend(model="model", base_url="http://localhost/v1", client=object(), **kwargs)
