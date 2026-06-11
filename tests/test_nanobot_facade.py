"""Tests for the Nanobot programmatic facade."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.nanobot import Nanobot, RunResult, RunStream, SessionInfo, SessionSnapshot, StreamEvent


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    data = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {"defaults": {"model": "openai/gpt-4.1"}},
    }
    if overrides:
        data.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    return config_path


def test_from_config_missing_file():
    with pytest.raises(FileNotFoundError):
        Nanobot.from_config("/nonexistent/config.json")


def test_from_config_creates_instance(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    assert bot._loop is not None
    assert bot._loop.workspace == tmp_path


def test_from_config_default_path():
    from nanobot.config.schema import Config

    with patch("nanobot.config.loader.load_config") as mock_load, \
         patch("nanobot.providers.factory.make_provider") as mock_prov:
        mock_load.return_value = Config()
        mock_prov.return_value = MagicMock()
        mock_prov.return_value.get_default_model.return_value = "test"
        mock_prov.return_value.generation.max_tokens = 4096
        Nanobot.from_config()
        mock_load.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_run_returns_result(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    from nanobot.bus.events import OutboundMessage

    mock_response = OutboundMessage(
        channel="cli", chat_id="direct", content="Hello back!"
    )
    bot._loop.process_direct = AsyncMock(return_value=mock_response)

    result = await bot.run("hi")

    assert isinstance(result, RunResult)
    assert result.content == "Hello back!"
    bot._loop.process_direct.assert_awaited_once_with("hi", session_key="sdk:default")


@pytest.mark.asyncio
async def test_run_with_hooks(tmp_path):
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    class TestHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            pass

    mock_response = OutboundMessage(
        channel="cli", chat_id="direct", content="done"
    )
    bot._loop.process_direct = AsyncMock(return_value=mock_response)

    result = await bot.run("hi", hooks=[TestHook()])

    assert result.content == "done"
    assert bot._loop._extra_hooks == []


@pytest.mark.asyncio
async def test_run_hooks_restored_on_error(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    from nanobot.agent.hook import AgentHook

    bot._loop.process_direct = AsyncMock(side_effect=RuntimeError("boom"))
    original_hooks = bot._loop._extra_hooks

    with pytest.raises(RuntimeError):
        await bot.run("hi", hooks=[AgentHook()])

    assert bot._loop._extra_hooks is original_hooks


@pytest.mark.asyncio
async def test_run_none_response(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock(return_value=None)

    result = await bot.run("hi")
    assert result.content == ""


def test_workspace_override(tmp_path):
    config_path = _write_config(tmp_path)
    custom_ws = tmp_path / "custom_workspace"
    custom_ws.mkdir()

    bot = Nanobot.from_config(config_path, workspace=custom_ws)
    assert bot._loop.workspace == custom_ws


def test_sdk_make_provider_uses_github_copilot_backend():
    from nanobot.config.schema import Config
    from nanobot.providers.factory import make_provider

    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "github-copilot",
                    "model": "github-copilot/gpt-4.1",
                }
            }
        }
    )

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = make_provider(config)

    assert provider.__class__.__name__ == "GitHubCopilotProvider"


@pytest.mark.asyncio
async def test_run_custom_session_key(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    mock_response = OutboundMessage(
        channel="cli", chat_id="direct", content="ok"
    )
    bot._loop.process_direct = AsyncMock(return_value=mock_response)

    await bot.run("hi", session_key="user-alice")
    bot._loop.process_direct.assert_awaited_once_with("hi", session_key="user-alice")


def test_import_from_top_level():
    import nanobot

    assert nanobot.Nanobot is Nanobot
    assert nanobot.RunResult is RunResult
    assert nanobot.RunStream is RunStream
    assert nanobot.SessionInfo is SessionInfo
    assert nanobot.SessionSnapshot is SessionSnapshot
    assert nanobot.StreamEvent is StreamEvent


# ---------------------------------------------------------------------------
# RunResult.tools_used / messages — populated from the agent iterations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_populates_tools_used_across_iterations(tmp_path):
    """tools_used collects every tool name fired across all iterations, in order."""
    from nanobot.agent.hook import AgentHookContext
    from nanobot.bus.events import OutboundMessage
    from nanobot.providers.base import ToolCallRequest

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key):
        # Whatever hooks the SDK installed are now on the loop.
        extras = bot._loop._extra_hooks
        messages = [{"role": "user", "content": message}]
        ctx1 = AgentHookContext(iteration=0, messages=messages)
        ctx1.tool_calls = [
            ToolCallRequest(id="c1", name="read_file", arguments={}),
            ToolCallRequest(id="c2", name="grep", arguments={}),
        ]
        for h in extras:
            await h.after_iteration(ctx1)
        messages.append({"role": "assistant", "content": "ok"})
        ctx2 = AgentHookContext(iteration=1, messages=messages)
        ctx2.tool_calls = [ToolCallRequest(id="c3", name="web_fetch", arguments={})]
        for h in extras:
            await h.after_iteration(ctx2)
        return OutboundMessage(channel="cli", chat_id="direct", content="final")

    bot._loop.process_direct = fake_process_direct
    result = await bot.run("do stuff")
    assert result.content == "final"
    assert result.tools_used == ["read_file", "grep", "web_fetch"]


@pytest.mark.asyncio
async def test_run_populates_final_messages(tmp_path):
    """messages reflects the agent's message list at the last iteration."""
    from nanobot.agent.hook import AgentHookContext
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key):
        extras = bot._loop._extra_hooks
        messages = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "hi there"},
        ]
        ctx = AgentHookContext(iteration=0, messages=messages)
        for h in extras:
            await h.after_iteration(ctx)
        return OutboundMessage(channel="cli", chat_id="direct", content="hi there")

    bot._loop.process_direct = fake_process_direct
    result = await bot.run("hello")
    assert result.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


@pytest.mark.asyncio
async def test_run_no_iterations_leaves_defaults_empty(tmp_path):
    """If process_direct never triggers after_iteration, tools_used/messages stay []."""
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="direct", content="noop"),
    )
    result = await bot.run("hi")
    assert result.tools_used == []
    assert result.messages == []
    assert result.usage == {}
    assert result.stop_reason is None
    assert result.error is None


@pytest.mark.asyncio
async def test_run_populates_observability_fields(tmp_path):
    from nanobot.agent.hook import AgentRunHookContext
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key):
        ctx = AgentRunHookContext(
            messages=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": "done"},
            ],
            final_content="done",
            tools_used=["read_file"],
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            stop_reason="completed",
            error=None,
            tool_events=[{"tool": "read_file", "status": "ok"}],
        )
        for h in bot._loop._extra_hooks:
            await h.after_run(ctx)
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="done",
            metadata={"latency_ms": 42},
        )

    bot._loop.process_direct = fake_process_direct
    result = await bot.run("work")

    assert result.content == "done"
    assert result.tools_used == ["read_file"]
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert result.stop_reason == "completed"
    assert result.error is None
    assert result.metadata == {"latency_ms": 42}


@pytest.mark.asyncio
async def test_run_forwards_non_default_runtime_options(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock(
        return_value=OutboundMessage(channel="sdk", chat_id="chat-a", content="ok"),
    )

    await bot.run(
        "hi",
        session_key="sdk:chat-a",
        channel="sdk",
        chat_id="chat-a",
        sender_id="alice",
        media=["/tmp/image.png"],
        ephemeral=True,
    )

    bot._loop.process_direct.assert_awaited_once_with(
        "hi",
        session_key="sdk:chat-a",
        channel="sdk",
        chat_id="chat-a",
        sender_id="alice",
        media=["/tmp/image.png"],
        ephemeral=True,
    )


@pytest.mark.asyncio
async def test_run_user_hooks_still_fire_alongside_capture(tmp_path):
    """Capture hook must not displace user-provided hooks."""
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    seen_iterations: list[int] = []

    class UserHook(AgentHook):
        async def after_iteration(self, context: AgentHookContext) -> None:
            seen_iterations.append(context.iteration)

    async def fake_process_direct(message, *, session_key):
        extras = bot._loop._extra_hooks
        assert len(extras) == 2, f"expected capture + user hook, got {len(extras)}"
        ctx = AgentHookContext(iteration=7, messages=[])
        for h in extras:
            await h.after_iteration(ctx)
        return OutboundMessage(channel="cli", chat_id="direct", content="ok")

    bot._loop.process_direct = fake_process_direct
    await bot.run("x", hooks=[UserHook()])
    assert seen_iterations == [7]


@pytest.mark.asyncio
async def test_run_restores_extra_hooks_even_on_populated_iterations(tmp_path):
    """Previously-installed _extra_hooks must be restored regardless of capture state."""
    from nanobot.agent.hook import AgentHook, AgentHookContext
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    sentinel_hook = AgentHook()
    bot._loop._extra_hooks = [sentinel_hook]

    async def fake_process_direct(message, *, session_key):
        ctx = AgentHookContext(iteration=0, messages=[])
        for h in bot._loop._extra_hooks:
            await h.after_iteration(ctx)
        return OutboundMessage(channel="cli", chat_id="direct", content="done")

    bot._loop.process_direct = fake_process_direct
    await bot.run("hello")
    assert bot._loop._extra_hooks == [sentinel_hook]


@pytest.mark.asyncio
async def test_stream_yields_text_events_in_order(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key, on_stream, on_stream_end):
        assert message == "hi"
        assert session_key == "sdk:default"
        await on_stream("Hel")
        await on_stream("lo")
        await on_stream_end(resuming=False)
        return OutboundMessage(channel="cli", chat_id="direct", content="Hello")

    bot._loop.process_direct = fake_process_direct

    events = [event async for event in bot.stream("hi")]

    assert [event.type for event in events] == [
        "run.started",
        "text.delta",
        "text.delta",
        "text.completed",
        "run.completed",
    ]
    assert events[1].delta == "Hel"
    assert events[2].delta == "lo"
    assert events[3].content == "Hello"
    assert events[4].result is not None
    assert events[4].result.content == "Hello"


@pytest.mark.asyncio
async def test_run_streamed_wait_returns_full_result_without_consuming_events(tmp_path):
    from nanobot.agent.hook import AgentRunHookContext
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key, on_stream, on_stream_end):
        await on_stream("done")
        await on_stream_end(resuming=False)
        ctx = AgentRunHookContext(
            messages=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": "done"},
            ],
            final_content="done",
            tools_used=["read_file"],
            usage={"total_tokens": 9},
            stop_reason="completed",
        )
        for hook in bot._loop._extra_hooks:
            await hook.after_run(ctx)
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="done",
            metadata={"latency_ms": 5},
        )

    bot._loop.process_direct = fake_process_direct

    run = await bot.run_streamed("work")
    assert isinstance(run, RunStream)
    result = await run.wait()

    assert result.content == "done"
    assert result.tools_used == ["read_file"]
    assert result.usage == {"total_tokens": 9}
    assert result.stop_reason == "completed"
    assert result.metadata == {"latency_ms": 5}


@pytest.mark.asyncio
async def test_run_streamed_text_returns_final_content(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="direct", content="plain text"),
    )

    run = await bot.run_streamed("hi")

    assert await run.text() == "plain text"


@pytest.mark.asyncio
async def test_run_streamed_forwards_runtime_options(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock(
        return_value=OutboundMessage(channel="sdk", chat_id="chat-a", content="ok"),
    )

    run = await bot.run_streamed(
        "hi",
        session_key="sdk:chat-a",
        channel="sdk",
        chat_id="chat-a",
        sender_id="alice",
        media=["/tmp/image.png"],
        ephemeral=True,
    )
    await run.wait()

    bot._loop.process_direct.assert_awaited_once()
    args, kwargs = bot._loop.process_direct.call_args
    assert args == ("hi",)
    assert kwargs["session_key"] == "sdk:chat-a"
    assert kwargs["channel"] == "sdk"
    assert kwargs["chat_id"] == "chat-a"
    assert kwargs["sender_id"] == "alice"
    assert kwargs["media"] == ["/tmp/image.png"]
    assert kwargs["ephemeral"] is True
    assert callable(kwargs["on_stream"])
    assert callable(kwargs["on_stream_end"])


@pytest.mark.asyncio
async def test_run_streamed_emits_tool_events(tmp_path):
    from nanobot.agent.hook import AgentHookContext
    from nanobot.bus.events import OutboundMessage
    from nanobot.providers.base import ToolCallRequest

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key, on_stream, on_stream_end):
        calls = [
            ToolCallRequest(id="call_ok", name="read_file", arguments={"path": "README.md"}),
            ToolCallRequest(id="call_bad", name="exec", arguments={"cmd": "false"}),
        ]
        ctx = AgentHookContext(iteration=2, messages=[{"role": "user", "content": message}])
        ctx.tool_calls = calls
        for hook in bot._loop._extra_hooks:
            await hook.before_execute_tools(ctx)
        ctx.tool_events = [
            {"name": "read_file", "status": "ok", "detail": "README.md"},
            {"name": "exec", "status": "error", "detail": "exit 1"},
        ]
        for hook in bot._loop._extra_hooks:
            await hook.after_iteration(ctx)
        return OutboundMessage(channel="cli", chat_id="direct", content="done")

    bot._loop.process_direct = fake_process_direct

    run = await bot.run_streamed("inspect")
    events = [event async for event in run.stream_events()]
    await run.wait()

    assert [event.type for event in events] == [
        "run.started",
        "tool.started",
        "tool.started",
        "tool.completed",
        "tool.failed",
        "run.completed",
    ]
    assert events[1].name == "read_file"
    assert events[1].tool_call_id == "call_ok"
    assert events[1].arguments == {"path": "README.md"}
    assert events[3].metadata["status"] == "ok"
    assert events[4].name == "exec"
    assert events[4].error == "exit 1"


@pytest.mark.asyncio
async def test_run_streamed_emits_reasoning_events(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    async def fake_process_direct(message, *, session_key, on_stream, on_stream_end):
        for hook in bot._loop._extra_hooks:
            await hook.emit_reasoning("thinking")
            await hook.emit_reasoning_end()
        return OutboundMessage(channel="cli", chat_id="direct", content="done")

    bot._loop.process_direct = fake_process_direct

    events = [event async for event in bot.stream("think")]

    assert [event.type for event in events] == [
        "run.started",
        "reasoning.delta",
        "reasoning.completed",
        "run.completed",
    ]
    assert events[1].delta == "thinking"


@pytest.mark.asyncio
async def test_run_streamed_restores_hooks_and_reports_failure(tmp_path):
    from nanobot.agent.hook import AgentHook

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    sentinel_hook = AgentHook()
    bot._loop._extra_hooks = [sentinel_hook]

    async def fake_process_direct(message, **kwargs):
        raise RuntimeError("boom")

    bot._loop.process_direct = fake_process_direct

    run = await bot.run_streamed("fail")
    events = [event async for event in run.stream_events()]

    assert [event.type for event in events] == ["run.started", "run.failed"]
    assert events[1].error == "boom"
    with pytest.raises(RuntimeError, match="boom"):
        await run.wait()
    assert bot._loop._extra_hooks == [sentinel_hook]


@pytest.mark.asyncio
async def test_run_streamed_stream_events_is_single_consumer(tmp_path):
    from nanobot.bus.events import OutboundMessage

    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="direct", content="done"),
    )

    run = await bot.run_streamed("hi")
    events = [event async for event in run.stream_events()]
    assert [event.type for event in events] == ["run.started", "run.completed"]
    await run.wait()

    with pytest.raises(RuntimeError, match="only be consumed once"):
        _ = [event async for event in run.stream_events()]


@pytest.mark.asyncio
async def test_sdk_capture_prefers_run_level_snapshot():
    from nanobot.agent.hook import AgentHookContext, AgentRunHookContext, SDKCaptureHook
    from nanobot.providers.base import ToolCallRequest

    hook = SDKCaptureHook()
    iter_messages = [{"role": "user", "content": "work"}]
    iter_context = AgentHookContext(iteration=0, messages=iter_messages)
    iter_context.tool_calls = [
        ToolCallRequest(id="call_1", name="read_file", arguments={}),
        ToolCallRequest(id="call_2", name="grep", arguments={}),
    ]
    await hook.after_iteration(iter_context)

    final_messages = [
        {"role": "user", "content": "work"},
        {"role": "assistant", "content": "done"},
    ]
    await hook.after_run(AgentRunHookContext(
        messages=final_messages,
        tools_used=["read_file"],
        usage={"total_tokens": 3},
        stop_reason="completed",
    ))

    assert hook.tools_used == ["read_file"]
    assert hook.messages == final_messages
    assert hook.usage == {"total_tokens": 3}
    assert hook.stop_reason == "completed"


@pytest.mark.asyncio
async def test_sessions_ingest_imports_transcript_without_running_model(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.process_direct = AsyncMock()
    bot._loop.consolidator.maybe_consolidate_by_tokens = AsyncMock()

    snapshot = await bot.sessions.ingest(
        "sdk:history",
        [
            {
                "role": "user",
                "content": "I graduated with a Business Administration degree.",
                "timestamp": "2023/05/30 (Tue) 17:27",
                "source_session_id": "answer_1",
            },
            {
                "role": "assistant",
                "content": "Congratulations on your degree.",
                "timestamp": "2023/05/30 (Tue) 17:27",
            },
        ],
        metadata={"title": "LongMemEval case"},
        source="longmemeval",
    )

    assert isinstance(snapshot, SessionSnapshot)
    assert snapshot.key == "sdk:history"
    assert snapshot.metadata["title"] == "LongMemEval case"
    assert snapshot.messages[0]["role"] == "user"
    assert snapshot.messages[0]["timestamp"] == "2023/05/30 (Tue) 17:27"
    assert snapshot.messages[0]["source_session_id"] == "answer_1"
    assert snapshot.messages[0]["source"] == "longmemeval"
    assert snapshot.messages[1]["source"] == "longmemeval"
    bot._loop.process_direct.assert_not_called()
    bot._loop.consolidator.maybe_consolidate_by_tokens.assert_not_called()

    reloaded = bot.sessions.get("sdk:history")
    assert reloaded is not None
    assert reloaded.messages == snapshot.messages


@pytest.mark.asyncio
async def test_sessions_ingest_validates_message_shape(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    with pytest.raises(ValueError, match="role"):
        await bot.sessions.ingest("sdk:bad", [{"content": "missing role"}])

    with pytest.raises(ValueError, match="unsupported message role"):
        await bot.sessions.ingest("sdk:bad", [{"role": "developer", "content": "nope"}])


@pytest.mark.asyncio
async def test_session_helpers_get_list_export_clear_delete_flush(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    await bot.sessions.ingest("sdk:first", [{"role": "user", "content": "hello"}])

    listed = bot.sessions.list()
    assert listed
    assert isinstance(listed[0], SessionInfo)
    assert {row.key for row in listed} == {"sdk:first"}

    exported = bot.sessions.export("sdk:first")
    assert exported is not None
    exported.messages[0]["content"] = "mutated copy"
    assert bot.sessions.get("sdk:first").messages[0]["content"] == "hello"

    cleared = bot.sessions.clear("sdk:first")
    assert cleared.messages == []
    assert bot.sessions.flush() >= 1
    assert bot.sessions.delete("sdk:first") is True
    assert bot.sessions.get("sdk:first") is None


def test_memory_helpers_read_write_append_and_filter_history(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    assert bot.memory.read() == ""
    bot.memory.write("# Memory\n- User likes concise APIs.")
    assert "concise APIs" in bot.memory.read()

    c1 = bot.memory.append_history("general event")
    c2 = bot.memory.append_history("session event", session_key="sdk:history")

    all_entries = bot.memory.read_history()
    assert [entry["cursor"] for entry in all_entries] == [c1, c2]

    session_entries = bot.memory.read_history(session_key="sdk:history")
    assert len(session_entries) == 1
    assert session_entries[0]["content"] == "session event"


@pytest.mark.asyncio
async def test_runtime_helpers_expose_model_workspace_and_compact(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    await bot.sessions.ingest("sdk:history", [{"role": "user", "content": "hello"}])

    bot._loop.consolidator.maybe_consolidate_by_tokens = AsyncMock()
    snapshot = await bot.runtime.compact_session("sdk:history")
    assert snapshot.key == "sdk:history"
    bot._loop.consolidator.maybe_consolidate_by_tokens.assert_awaited_once()
    assert bot.runtime.model == bot._loop.model
    assert bot.runtime.workspace == tmp_path

    bot._loop.consolidator.compact_idle_session = AsyncMock(return_value="Summary.")
    summary = await bot.runtime.compact_idle_session("sdk:history", max_suffix=4)
    assert summary == "Summary."
    bot._loop.consolidator.compact_idle_session.assert_awaited_once_with(
        "sdk:history",
        max_suffix=4,
    )


@pytest.mark.asyncio
async def test_aclose_delegates_to_loop_close_mcp(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.close_mcp = AsyncMock()

    await bot.aclose()

    bot._loop.close_mcp.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_manager_calls_aclose_on_exit(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.close_mcp = AsyncMock()

    async with bot as b:
        assert b is bot

    bot._loop.close_mcp.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_manager_does_not_swallow_exceptions(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._loop.close_mcp = AsyncMock()

    with pytest.raises(ValueError):
        async with bot as b:
            assert b is bot
            raise ValueError("boom")

    bot._loop.close_mcp.assert_awaited_once()
