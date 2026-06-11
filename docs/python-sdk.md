# Python SDK

Use nanobot as a library — no CLI, no gateway, just Python. The SDK exposes
the same agent runtime used by the CLI, plus stable helpers for sessions,
memory, and compaction.

Before debugging SDK code, prove the same config works from the CLI:

```bash
nanobot agent -m "Hello!"
```

`Nanobot.from_config()` reuses your normal `~/.nanobot/config.json`, so provider, model, tools, and workspace behavior match the CLI unless you override them.

## Quick Start

```python
import asyncio

from nanobot import Nanobot


async def main() -> None:
    async with Nanobot.from_config() as bot:
        result = await bot.run("What time is it in Tokyo?")
    print(result.content)


asyncio.run(main())
```

Use `async with` when possible so MCP connections and background cleanup work are closed before the event loop exits. If you manage the instance manually, call `await bot.aclose()` in a `finally` block.

## Common Patterns

### Use a specific config or workspace

```python
from nanobot import Nanobot

bot = Nanobot.from_config(
    config_path="~/.nanobot/config.json",
    workspace="/my/project",
)
```

### Choose a default or per-run model

Set the SDK instance default model when you create the bot:

```python
bot = Nanobot.from_config(model="openai/gpt-4.1")
```

Override the model for one run without changing the instance default:

```python
result = await bot.run("Summarize this file", model="openai/gpt-4.1-mini")
```

Model presets from `config.json` work the same way:

```python
bot = Nanobot.from_config(model_preset="fast")

result = await bot.run("Think deeply about this bug", model_preset="reasoning")
```

`model` and `model_preset` are mutually exclusive.

### Isolate conversations with `session_key`

Different session keys keep independent conversation history:

```python
await bot.run("hi", session_key="user-alice")
await bot.run("hi", session_key="task-42")
```

### Stream long-running output

Use `bot.stream()` when you want Cursor/OpenAI-style live events instead of
waiting for the final `RunResult`:

```python
from nanobot import (
    STREAM_EVENT_RUN_COMPLETED,
    STREAM_EVENT_TEXT_DELTA,
    STREAM_EVENT_TOOL_STARTED,
)

async for event in bot.stream("Review this repository"):
    if event.type == STREAM_EVENT_TEXT_DELTA:
        print(event.delta, end="", flush=True)
    elif event.type == STREAM_EVENT_TOOL_STARTED:
        print(f"\nusing {event.name}")
    elif event.type == STREAM_EVENT_RUN_COMPLETED:
        print("\nfinal:", event.result.content)
```

Use `run_streamed()` when you also want a handle you can wait on:

```python
from nanobot import STREAM_EVENT_TEXT_DELTA

run = await bot.run_streamed("Write a detailed migration plan")

async for event in run.stream_events():
    if event.type == STREAM_EVENT_TEXT_DELTA:
        print(event.delta, end="", flush=True)

result = await run.wait()
```

### Import an existing transcript

Use `bot.sessions.ingest()` when you already have a transcript and want it to
become nanobot session history. Ingesting a transcript does not call the model,
execute tools, trigger Dream, or compact automatically.

```python
await bot.sessions.ingest(
    "eval:case-1",
    [
        {
            "role": "user",
            "content": "I graduated with a degree in Business Administration.",
            "timestamp": "2023/05/30 (Tue) 17:27",
            "source_session_id": "answer_280352e9",
        },
        {
            "role": "assistant",
            "content": "Congratulations on your degree.",
            "timestamp": "2023/05/30 (Tue) 17:27",
        },
    ],
    source="longmemeval",
)

await bot.runtime.compact_session("eval:case-1")

result = await bot.run(
    "Current Date: 2023/05/30 (Tue) 23:40\n"
    "Question: What degree did I graduate with?",
    session_key="eval:case-1",
)
print(result.content)
```

### Attach hooks for observability

Hooks let you inspect tool calls, streaming, and iteration state without modifying nanobot internals:

```python
from nanobot.agent import AgentHook, AgentHookContext


class AuditHook(AgentHook):
    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tc in context.tool_calls:
            print(f"[tool] {tc.name}")


result = await bot.run("Review this change", hooks=[AuditHook()])
```

## API Reference

### `Nanobot.from_config(config_path=None, *, workspace=None)`

Create a `Nanobot` instance from a config file.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `config_path` | `str \| Path \| None` | `None` | Path to `config.json`. Defaults to `~/.nanobot/config.json`. |
| `workspace` | `str \| Path \| None` | `None` | Override the workspace directory from config. |
| `model` | `str \| None` | `None` | Override the instance default model. |
| `model_preset` | `str \| None` | `None` | Override the instance default model preset from `config.json`. |

Raises `FileNotFoundError` if an explicit config path does not exist.
Raises `ValueError` if both `model` and `model_preset` are provided.

### `await bot.run(...)`

Run the agent once and return a `RunResult`.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `str` | *(required)* | The user message to process. |
| `session_key` | `str` | `"sdk:default"` | Session identifier for conversation isolation. Different keys get independent history. |
| `channel` | `str` | `"cli"` | Logical channel label used in runtime context. |
| `chat_id` | `str` | `"direct"` | Logical chat identifier used in runtime context. |
| `sender_id` | `str` | `"user"` | Logical sender identifier used in runtime context. |
| `media` | `list[str] \| None` | `None` | Optional local media paths attached to the message. |
| `ephemeral` | `bool` | `False` | Run without persisting the turn or compacting session history. |
| `hooks` | `list[AgentHook] \| None` | `None` | Lifecycle hooks for this run only. |
| `model` | `str \| None` | `None` | Override the model for this run only. |
| `model_preset` | `str \| None` | `None` | Override the model preset for this run only. |

`model` and `model_preset` are per-run overrides and do not change
`bot.runtime.model` after the run completes. They are mutually exclusive.

### `await bot.run_streamed(...)`

Start a streamed agent turn and return a `RunStream`. It accepts the same
parameters as `bot.run(...)`.

```python
run = await bot.run_streamed("Generate a long answer")

async for event in run.stream_events():
    ...

result = await run.wait()
```

### `bot.stream(...)`

Convenience wrapper around `run_streamed()` for direct event iteration. It
accepts the same parameters as `bot.run(...)`.

```python
async for event in bot.stream("Generate a long answer"):
    ...
```

### `RunStream`

| Method | Description |
|--------|-------------|
| `stream_events()` | Single-consumer async iterator of `StreamEvent` objects. |
| `await wait()` | Wait for the run to finish and return `RunResult`. |
| `await text()` | Wait for the run to finish and return `RunResult.content`. |

### `StreamEvent`

| Field | Type | Description |
|-------|------|-------------|
| `type` | `StreamEventType` | Event type, such as `text.delta` or `run.completed`. |
| `delta` | `str` | Incremental text or reasoning chunk. |
| `content` | `str` | Completed text segment or final content. |
| `result` | `RunResult \| None` | Present on `run.completed`. |
| `name` | `str \| None` | Tool name for tool events. |
| `tool_call_id` | `str \| None` | Provider tool call id when available. |
| `arguments` | `dict \| None` | Tool arguments when available. |
| `iteration` | `int \| None` | Agent loop iteration when available. |
| `resuming` | `bool \| None` | Whether a text segment ended before more tool work. |
| `usage` | `dict[str, int]` | Token usage on completion events. |
| `error` | `str \| None` | Error text on failed events. |
| `metadata` | `dict` | Additional event metadata. |

Use the exported constants instead of hard-coded strings when possible:

| Constant | Value |
|----------|-------|
| `STREAM_EVENT_RUN_STARTED` | `run.started` |
| `STREAM_EVENT_TEXT_DELTA` | `text.delta` |
| `STREAM_EVENT_TEXT_COMPLETED` | `text.completed` |
| `STREAM_EVENT_REASONING_DELTA` | `reasoning.delta` |
| `STREAM_EVENT_REASONING_COMPLETED` | `reasoning.completed` |
| `STREAM_EVENT_TOOL_STARTED` | `tool.started` |
| `STREAM_EVENT_TOOL_COMPLETED` | `tool.completed` |
| `STREAM_EVENT_TOOL_FAILED` | `tool.failed` |
| `STREAM_EVENT_RUN_COMPLETED` | `run.completed` |
| `STREAM_EVENT_RUN_FAILED` | `run.failed` |

`STREAM_EVENT_TYPES` contains all stable v1 event values.

### `await bot.aclose()`

Release resources held by the SDK instance, including MCP connections. The async context manager calls this automatically:

```python
async with Nanobot.from_config() as bot:
    result = await bot.run("Summarize this repo")
```

### `RunResult`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The agent's final text response. |
| `tools_used` | `list[str]` | Tool names used during the run. |
| `messages` | `list[dict]` | Final message list from the run. |
| `usage` | `dict[str, int]` | Token usage reported or estimated by the runtime. |
| `stop_reason` | `str \| None` | Why the run stopped, such as `"completed"` or `"max_iterations"`. |
| `error` | `str \| None` | Error text when the run failed inside the agent runtime. |
| `metadata` | `dict` | Outbound metadata such as latency. |

## Session, Memory, And Runtime Helpers

### `bot.sessions`

| Method | Description |
|--------|-------------|
| `await ingest(session_key, messages, metadata=None, source=None, save=True)` | Import existing transcript messages without running the model. |
| `get(session_key)` | Return a `SessionSnapshot`, or `None` if missing. |
| `list()` | Return compact `SessionInfo` rows. |
| `export(session_key)` | Return a full `SessionSnapshot` suitable for JSON serialization. |
| `clear(session_key)` | Clear and persist one session. |
| `delete(session_key)` | Delete one session from disk and cache. |
| `flush()` | Flush cached sessions to durable storage. |

Ingested messages must include `role` and `content`. Roles may be `user`,
`assistant`, `tool`, or `system`. Other fields, such as `timestamp`,
`source_session_id`, or `source_date`, are persisted as message metadata.

### `bot.memory`

| Method | Description |
|--------|-------------|
| `read()` | Read `memory/MEMORY.md`. |
| `write(text)` | Overwrite `memory/MEMORY.md`. |
| `append_history(text, session_key=None)` | Append one `memory/history.jsonl` entry and return its cursor. |
| `read_history(session_key=None)` | Read memory history entries, optionally filtered by session key. |

### `bot.runtime`

| Method / Property | Description |
|-------------------|-------------|
| `model` | Current runtime model name. |
| `workspace` | Current runtime workspace path. |
| `await compact_session(session_key)` | Run token/replay-window consolidation for a session. |
| `await compact_idle_session(session_key, max_suffix=8)` | Run idle-session compaction and return its summary. |

## Hooks

Hooks let you observe or customize the agent loop. Subclass `AgentHook` and override the methods you need.

### Hook lifecycle

| Method | When |
|--------|------|
| `wants_streaming()` | Return `True` if you want token-by-token `on_stream()` callbacks |
| `before_iteration(context)` | Before each LLM call |
| `on_stream(context, delta)` | On each streamed token when streaming is enabled |
| `on_stream_end(context, *, resuming)` | When streaming finishes |
| `before_execute_tools(context)` | Before tool execution |
| `after_iteration(context)` | After each iteration |
| `finalize_content(context, content)` | Transform final output text |

Useful fields on `AgentHookContext` include:

- `iteration`
- `messages`
- `response`
- `usage`
- `tool_calls`
- `tool_results`
- `tool_events`
- `final_content`
- `stop_reason`
- `error`

### Example: audit tool calls

```python
from nanobot.agent import AgentHook, AgentHookContext


class AuditHook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tc in context.tool_calls:
            self.calls.append(tc.name)
            print(f"[audit] {tc.name}({tc.arguments})")
```

```python
hook = AuditHook()
result = await bot.run("List files in /tmp", hooks=[hook])
print(result.content)
print(f"Tools observed: {hook.calls}")
```

### Example: receive streaming tokens

```python
from nanobot.agent import AgentHook, AgentHookContext


class StreamingHook(AgentHook):
    def wants_streaming(self) -> bool:
        return True

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        print(delta, end="", flush=True)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        print()
```

### Compose multiple hooks

Pass multiple hooks when you want to combine behaviors:

```python
result = await bot.run("hi", hooks=[AuditHook(), MetricsHook()])
```

Async hook methods are fan-out with error isolation. `finalize_content` is a pipeline: each hook receives the previous hook's output.

### Example: post-process final content

```python
from nanobot.agent import AgentHook


class Censor(AgentHook):
    def finalize_content(self, context, content):
        return content.replace("secret", "***") if content else content
```

## Full Example

```python
import asyncio
import time

from nanobot import Nanobot
from nanobot.agent import AgentHook, AgentHookContext


class TimingHook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self._started_at = 0.0

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._started_at = time.perf_counter()

    async def after_iteration(self, context: AgentHookContext) -> None:
        elapsed_ms = (time.perf_counter() - self._started_at) * 1000
        print(f"[timing] iteration {context.iteration} took {elapsed_ms:.1f}ms")


async def main() -> None:
    bot = Nanobot.from_config(workspace="/my/project")
    result = await bot.run(
        "Explain the main function",
        session_key="sdk:demo",
        hooks=[TimingHook()],
    )
    print(result.content)


asyncio.run(main())
```
