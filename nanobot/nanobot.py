"""High-level programmatic interface to nanobot."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from nanobot.agent.hook import AgentHook, AgentHookContext, SDKCaptureHook
from nanobot.agent.loop import AgentLoop
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.factory import ProviderSnapshot, build_provider_snapshot
from nanobot.providers.image_generation import image_gen_provider_configs

StreamEventType: TypeAlias = Literal[
    "run.started",
    "text.delta",
    "text.completed",
    "reasoning.delta",
    "reasoning.completed",
    "tool.started",
    "tool.completed",
    "tool.failed",
    "run.completed",
    "run.failed",
]

STREAM_EVENT_RUN_STARTED: StreamEventType = "run.started"
STREAM_EVENT_TEXT_DELTA: StreamEventType = "text.delta"
STREAM_EVENT_TEXT_COMPLETED: StreamEventType = "text.completed"
STREAM_EVENT_REASONING_DELTA: StreamEventType = "reasoning.delta"
STREAM_EVENT_REASONING_COMPLETED: StreamEventType = "reasoning.completed"
STREAM_EVENT_TOOL_STARTED: StreamEventType = "tool.started"
STREAM_EVENT_TOOL_COMPLETED: StreamEventType = "tool.completed"
STREAM_EVENT_TOOL_FAILED: StreamEventType = "tool.failed"
STREAM_EVENT_RUN_COMPLETED: StreamEventType = "run.completed"
STREAM_EVENT_RUN_FAILED: StreamEventType = "run.failed"

STREAM_EVENT_TYPES: tuple[StreamEventType, ...] = (
    STREAM_EVENT_RUN_STARTED,
    STREAM_EVENT_TEXT_DELTA,
    STREAM_EVENT_TEXT_COMPLETED,
    STREAM_EVENT_REASONING_DELTA,
    STREAM_EVENT_REASONING_COMPLETED,
    STREAM_EVENT_TOOL_STARTED,
    STREAM_EVENT_TOOL_COMPLETED,
    STREAM_EVENT_TOOL_FAILED,
    STREAM_EVENT_RUN_COMPLETED,
    STREAM_EVENT_RUN_FAILED,
)


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamEvent:
    """A typed event emitted by ``Nanobot.stream()`` and ``RunStream``."""

    type: StreamEventType
    delta: str = ""
    content: str = ""
    result: RunResult | None = None
    name: str | None = None
    tool_call_id: str | None = None
    arguments: dict[str, Any] | None = None
    iteration: int | None = None
    resuming: bool | None = None
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_STREAM_SENTINEL = object()


class RunStream:
    """A running SDK turn with Cursor/OpenAI-style event streaming."""

    def __init__(
        self,
        task: asyncio.Task[RunResult],
        queue: asyncio.Queue[StreamEvent | object],
    ) -> None:
        self._task = task
        self._queue = queue
        self._events_started = False
        self._events_done = False
        self._stream_active = False

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Yield streaming events for this run.

        The event stream is single-consumer: call this method only once.
        """
        if self._events_started:
            raise RuntimeError("RunStream.stream_events() can only be consumed once")
        self._events_started = True
        self._stream_active = True
        try:
            while True:
                item = await self._queue.get()
                if item is _STREAM_SENTINEL:
                    self._events_done = True
                    break
                yield item
        finally:
            self._stream_active = False

    async def wait(self) -> RunResult:
        """Wait for the run to finish and return its final result."""
        if not self._events_done and not self._stream_active:
            if not self._events_started:
                self._events_started = True
            await self._drain_events()
        return await self._task

    async def text(self) -> str:
        """Wait for the run to finish and return the final text."""
        return (await self.wait()).content

    async def _drain_events(self) -> None:
        while not self._events_done:
            item = await self._queue.get()
            if item is _STREAM_SENTINEL:
                self._events_done = True
                break


@dataclass(slots=True)
class SessionSnapshot:
    """A durable snapshot of one nanobot session."""

    key: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the snapshot."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": deepcopy(self.metadata),
            "messages": deepcopy(self.messages),
        }


@dataclass(slots=True)
class SessionInfo:
    """Compact session metadata for listings."""

    key: str
    created_at: str | None = None
    updated_at: str | None = None
    title: str = ""
    preview: str = ""
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the listing row."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "preview": self.preview,
            "path": self.path,
        }


def _snapshot_from_session(session: Any) -> SessionSnapshot:
    return SessionSnapshot(
        key=session.key,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        metadata=deepcopy(session.metadata),
        messages=deepcopy(session.messages),
    )


def _snapshot_from_payload(payload: Mapping[str, Any]) -> SessionSnapshot:
    return SessionSnapshot(
        key=str(payload.get("key") or ""),
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        metadata=deepcopy(dict(payload.get("metadata") or {})),
        messages=deepcopy(list(payload.get("messages") or [])),
    )


def _result_from_response(response: Any, capture: SDKCaptureHook) -> RunResult:
    content = (response.content if response else None) or ""
    metadata = dict(response.metadata) if response and response.metadata else {}
    return RunResult(
        content=content,
        tools_used=capture.tools_used,
        messages=capture.messages,
        usage=capture.usage,
        stop_reason=capture.stop_reason,
        error=capture.error,
        metadata=metadata,
    )


class _SDKStreamEmitter:
    """Serialize SDK streaming events onto a bounded async queue."""

    def __init__(self, queue: asyncio.Queue[StreamEvent | object]) -> None:
        self._queue = queue
        self._text_parts: list[str] = []

    async def emit(self, event: StreamEvent) -> None:
        await self._queue.put(event)

    async def text_delta(self, delta: str, *, iteration: int | None = None) -> None:
        if not delta:
            return
        self._text_parts.append(delta)
        await self.emit(StreamEvent(
            type=STREAM_EVENT_TEXT_DELTA,
            delta=delta,
            iteration=iteration,
        ))

    async def text_completed(
        self,
        *,
        resuming: bool = False,
        iteration: int | None = None,
        force: bool = True,
    ) -> None:
        content = "".join(self._text_parts)
        if not content and (resuming or not force):
            return
        self._text_parts = []
        await self.emit(StreamEvent(
            type=STREAM_EVENT_TEXT_COMPLETED,
            content=content,
            iteration=iteration,
            resuming=resuming,
        ))

    async def close(self) -> None:
        await self._queue.put(_STREAM_SENTINEL)


class _SDKStreamingHook(AgentHook):
    """Convert agent lifecycle hooks into public SDK stream events."""

    def __init__(self, emitter: _SDKStreamEmitter) -> None:
        super().__init__()
        self._emitter = emitter
        self._reasoning_open = False

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for call in context.tool_calls:
            await self._emitter.emit(StreamEvent(
                type=STREAM_EVENT_TOOL_STARTED,
                name=call.name,
                tool_call_id=call.id,
                arguments=deepcopy(call.arguments),
                iteration=context.iteration,
            ))

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        if not reasoning_content:
            return
        self._reasoning_open = True
        await self._emitter.emit(StreamEvent(
            type=STREAM_EVENT_REASONING_DELTA,
            delta=reasoning_content,
        ))

    async def emit_reasoning_end(self) -> None:
        if not self._reasoning_open:
            return
        self._reasoning_open = False
        await self._emitter.emit(StreamEvent(type=STREAM_EVENT_REASONING_COMPLETED))

    async def after_iteration(self, context: AgentHookContext) -> None:
        if not context.tool_events:
            return
        for index, raw_event in enumerate(context.tool_events):
            call = context.tool_calls[index] if index < len(context.tool_calls) else None
            event = dict(raw_event)
            status = event.get("status")
            name = str(event.get("name") or (call.name if call else ""))
            event_type = (
                STREAM_EVENT_TOOL_COMPLETED if status == "ok" else STREAM_EVENT_TOOL_FAILED
            )
            await self._emitter.emit(StreamEvent(
                type=event_type,
                name=name or None,
                tool_call_id=call.id if call else None,
                arguments=deepcopy(call.arguments) if call else None,
                iteration=context.iteration,
                error=None if status == "ok" else str(event.get("detail") or ""),
                metadata=event,
            ))


class SessionClient:
    """Session management helpers exposed through ``bot.sessions``."""

    _RESERVED_MESSAGE_KEYS = {"role", "content"}
    _VALID_ROLES = {"user", "assistant", "tool", "system"}

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    async def ingest(
        self,
        session_key: str,
        messages: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        source: str | None = None,
        save: bool = True,
    ) -> SessionSnapshot:
        """Import an existing transcript without running the model."""
        session = self._loop.sessions.get_or_create(session_key)
        if metadata:
            session.metadata.update(deepcopy(dict(metadata)))

        for raw in messages:
            if "role" not in raw:
                raise ValueError("ingested messages must include a role")
            if "content" not in raw:
                raise ValueError("ingested messages must include content")
            role = str(raw["role"]).strip()
            if role not in self._VALID_ROLES:
                raise ValueError(f"unsupported message role: {role!r}")
            extra = {
                key: deepcopy(value)
                for key, value in raw.items()
                if key not in self._RESERVED_MESSAGE_KEYS
            }
            if source is not None and "source" not in extra:
                extra["source"] = source
            session.add_message(role, deepcopy(raw["content"]), **extra)

        if save:
            self._loop.sessions.save(session)
        return _snapshot_from_session(session)

    def get(self, session_key: str) -> SessionSnapshot | None:
        """Return a session snapshot without creating a new session on disk."""
        cached = self._loop.sessions._cache.get(session_key)
        if cached is not None:
            return _snapshot_from_session(cached)
        payload = self._loop.sessions.read_session_file(session_key)
        if payload is None:
            return None
        return _snapshot_from_payload(payload)

    def list(self) -> list[SessionInfo]:
        """List persisted sessions."""
        return [
            SessionInfo(
                key=str(row.get("key") or ""),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
                title=str(row.get("title") or ""),
                preview=str(row.get("preview") or ""),
                path=row.get("path"),
            )
            for row in self._loop.sessions.list_sessions()
        ]

    def export(self, session_key: str) -> SessionSnapshot | None:
        """Return a full session snapshot suitable for JSON serialization."""
        return self.get(session_key)

    def clear(self, session_key: str) -> SessionSnapshot:
        """Clear one session and persist the empty session."""
        session = self._loop.sessions.get_or_create(session_key)
        session.clear()
        self._loop.sessions.save(session)
        return _snapshot_from_session(session)

    def delete(self, session_key: str) -> bool:
        """Delete one session from disk and cache."""
        return self._loop.sessions.delete_session(session_key)

    def flush(self) -> int:
        """Flush cached sessions to durable storage."""
        return self._loop.sessions.flush_all()


class MemoryClient:
    """Long-term memory helpers exposed through ``bot.memory``."""

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    def read(self) -> str:
        """Read ``memory/MEMORY.md``."""
        return self._loop.context.memory.read_memory()

    def write(self, text: str) -> None:
        """Overwrite ``memory/MEMORY.md``."""
        self._loop.context.memory.write_memory(text)

    def append_history(self, text: str, *, session_key: str | None = None) -> int:
        """Append one entry to ``memory/history.jsonl`` and return its cursor."""
        return self._loop.context.memory.append_history(text, session_key=session_key)

    def read_history(self, *, session_key: str | None = None) -> list[dict[str, Any]]:
        """Read memory history entries, optionally filtered by session."""
        entries = self._loop.context.memory.read_unprocessed_history(since_cursor=0)
        if session_key is not None:
            entries = [entry for entry in entries if entry.get("session_key") == session_key]
        return deepcopy(entries)


class RuntimeClient:
    """Runtime control helpers exposed through ``bot.runtime``."""

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop

    @property
    def model(self) -> str:
        """Current runtime model name."""
        return self._loop.model

    @property
    def workspace(self) -> Path:
        """Current runtime workspace."""
        return self._loop.workspace

    async def compact_session(self, session_key: str) -> SessionSnapshot:
        """Run token/replay-window consolidation for one session."""
        session = self._loop.sessions.get_or_create(session_key)
        await self._loop.consolidator.maybe_consolidate_by_tokens(
            session,
            replay_max_messages=self._loop._max_messages,
        )
        return _snapshot_from_session(self._loop.sessions.get_or_create(session_key))

    async def compact_idle_session(self, session_key: str, *, max_suffix: int = 8) -> str | None:
        """Run idle-session compaction for one session and return the summary."""
        return await self._loop.consolidator.compact_idle_session(
            session_key,
            max_suffix=max_suffix,
        )


class Nanobot:
    """Programmatic facade for running the nanobot agent.

    Usage::

        bot = Nanobot.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)
    """

    def __init__(self, loop: AgentLoop, *, config: Config | None = None) -> None:
        self._loop = loop
        self._config = config
        self._sdk_run_lock = asyncio.Lock()
        self.sessions = SessionClient(loop)
        self.memory = MemoryClient(loop)
        self.runtime = RuntimeClient(loop)

    @staticmethod
    def _process_direct_kwargs(
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        sender_id: str,
        media: list[str] | None,
        ephemeral: bool,
        on_stream: Any | None = None,
        on_stream_end: Any | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"session_key": session_key}
        if channel != "cli":
            kwargs["channel"] = channel
        if chat_id != "direct":
            kwargs["chat_id"] = chat_id
        if sender_id != "user":
            kwargs["sender_id"] = sender_id
        if media is not None:
            kwargs["media"] = media
        if ephemeral:
            kwargs["ephemeral"] = True
        if on_stream is not None:
            kwargs["on_stream"] = on_stream
        if on_stream_end is not None:
            kwargs["on_stream_end"] = on_stream_end
        return kwargs

    @staticmethod
    def _ensure_single_model_selector(
        *,
        model: str | None,
        model_preset: str | None,
    ) -> None:
        if model is not None and model_preset is not None:
            raise ValueError("model and model_preset are mutually exclusive")

    def _model_override_snapshot(
        self,
        *,
        model: str | None,
        model_preset: str | None,
    ) -> ProviderSnapshot | None:
        self._ensure_single_model_selector(model=model, model_preset=model_preset)
        if model_preset is not None:
            return self._loop._build_model_preset_snapshot(model_preset)
        if model is None:
            return None

        if self._config is not None:
            base = self._config.resolve_preset(self._loop.model_preset)
            preset = base.model_copy(update={"model": model, "provider": "auto"})
            return build_provider_snapshot(self._config, preset=preset)

        generation = getattr(self._loop.provider, "generation", None)
        preset = ModelPresetConfig(
            model=model,
            provider="auto",
            max_tokens=getattr(generation, "max_tokens", 8192),
            context_window_tokens=self._loop.context_window_tokens,
            temperature=getattr(generation, "temperature", 0.1),
            reasoning_effort=getattr(generation, "reasoning_effort", None),
        )
        from nanobot.agent.model_presets import build_static_preset_snapshot

        return build_static_preset_snapshot(self._loop.provider, "sdk:override", preset)

    def _current_runtime_snapshot(self) -> ProviderSnapshot:
        signature = self._loop._provider_signature
        if signature is None:
            signature = ("sdk:runtime", id(self._loop.provider), self._loop.model)
        return ProviderSnapshot(
            provider=self._loop.provider,
            model=self._loop.model,
            context_window_tokens=self._loop.context_window_tokens,
            signature=signature,
        )

    def _restore_runtime_snapshot(
        self,
        snapshot: ProviderSnapshot,
        *,
        provider_signature: tuple[object, ...] | None,
    ) -> None:
        self._loop._apply_provider_snapshot(snapshot, publish_update=False)
        self._loop._provider_signature = provider_signature

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> Nanobot:
        """Create a Nanobot instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``~/.nanobot/config.json``.
            workspace: Override the workspace directory from config.
            model: Override the instance default model.
            model_preset: Override the instance default model preset.
        """
        from nanobot.config.loader import load_config, resolve_config_env_vars

        cls._ensure_single_model_selector(model=model, model_preset=model_preset)
        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(load_config(resolved))
        if workspace is not None:
            config.agents.defaults.workspace = str(
                Path(workspace).expanduser().resolve()
            )
        if model is not None:
            config.agents.defaults.model_preset = None
            config.agents.defaults.model = model
            config.agents.defaults.provider = "auto"
        elif model_preset is not None:
            config.agents.defaults.model_preset = model_preset

        loop = AgentLoop.from_config(
            config,
            image_generation_provider_configs=image_gen_provider_configs(config),
        )
        return cls(loop, config=config)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            channel: Logical channel label for runtime context.
            chat_id: Logical chat identifier for runtime context.
            sender_id: Logical sender identifier for runtime context.
            media: Optional local media paths attached to the message.
            ephemeral: If true, do not persist the turn or compact session history.
            hooks: Optional lifecycle hooks for this run.
            model: Override the model for this run only.
            model_preset: Override the model preset for this run only.
        """
        capture = SDKCaptureHook()
        prev = self._loop._extra_hooks
        base_hooks = list(hooks) if hooks is not None else list(prev or [])
        override = self._model_override_snapshot(model=model, model_preset=model_preset)
        async with self._sdk_run_lock:
            restore = self._current_runtime_snapshot() if override is not None else None
            restore_signature = self._loop._provider_signature
            if override is not None:
                self._loop._apply_provider_snapshot(
                    override,
                    publish_update=False,
                    model_preset=model_preset,
                )
            self._loop._extra_hooks = [capture, *base_hooks]
            kwargs = self._process_direct_kwargs(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                sender_id=sender_id,
                media=media,
                ephemeral=ephemeral,
            )
            try:
                response = await self._loop.process_direct(
                    message,
                    **kwargs,
                )
            finally:
                self._loop._extra_hooks = prev
                if restore is not None:
                    self._restore_runtime_snapshot(
                        restore,
                        provider_signature=restore_signature,
                    )

        return _result_from_response(response, capture)

    async def run_streamed(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> RunStream:
        """Start a streamed run and return a handle for events and final result."""
        queue: asyncio.Queue[StreamEvent | object] = asyncio.Queue(maxsize=256)
        emitter = _SDKStreamEmitter(queue)
        stream_hook = _SDKStreamingHook(emitter)
        capture = SDKCaptureHook()
        override = self._model_override_snapshot(model=model, model_preset=model_preset)

        async def _on_stream(delta: str) -> None:
            await emitter.text_delta(delta)

        async def _on_stream_end(*_args: Any, resuming: bool = False, **_kwargs: Any) -> None:
            await emitter.text_completed(resuming=resuming)

        async def _run() -> RunResult:
            async with self._sdk_run_lock:
                prev = self._loop._extra_hooks
                base_hooks = list(hooks) if hooks is not None else list(prev or [])
                restore = self._current_runtime_snapshot() if override is not None else None
                restore_signature = self._loop._provider_signature
                if override is not None:
                    self._loop._apply_provider_snapshot(
                        override,
                        publish_update=False,
                        model_preset=model_preset,
                    )
                self._loop._extra_hooks = [capture, stream_hook, *base_hooks]
                kwargs = self._process_direct_kwargs(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                    sender_id=sender_id,
                    media=media,
                    ephemeral=ephemeral,
                    on_stream=_on_stream,
                    on_stream_end=_on_stream_end,
                )
                await emitter.emit(StreamEvent(
                    type=STREAM_EVENT_RUN_STARTED,
                    metadata={
                        "session_key": session_key,
                        "channel": channel,
                        "chat_id": chat_id,
                        "sender_id": sender_id,
                        "model": self._loop.model,
                        "model_preset": (
                            model_preset if model_preset is not None else self._loop.model_preset
                        ),
                    },
                ))
                try:
                    response = await self._loop.process_direct(message, **kwargs)
                    await emitter.text_completed(resuming=False, force=False)
                    result = _result_from_response(response, capture)
                    await emitter.emit(StreamEvent(
                        type=STREAM_EVENT_RUN_COMPLETED,
                        content=result.content,
                        result=result,
                        usage=dict(result.usage),
                        metadata=dict(result.metadata),
                    ))
                    return result
                except Exception as exc:
                    await emitter.emit(StreamEvent(
                        type=STREAM_EVENT_RUN_FAILED,
                        error=str(exc),
                        metadata={"exception_type": type(exc).__name__},
                    ))
                    raise
                finally:
                    self._loop._extra_hooks = prev
                    if restore is not None:
                        self._restore_runtime_snapshot(
                            restore,
                            provider_signature=restore_signature,
                        )
                    await emitter.close()

        task = asyncio.create_task(_run())
        return RunStream(task, queue)

    async def stream(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events for one agent turn."""
        run = await self.run_streamed(
            message,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            media=media,
            ephemeral=ephemeral,
            hooks=hooks,
            model=model,
            model_preset=model_preset,
        )
        async for event in run.stream_events():
            yield event
        await run.wait()

    async def aclose(self) -> None:
        """Release resources held by this instance (MCP connections, etc.)."""
        await self._loop.close_mcp()

    async def __aenter__(self) -> Nanobot:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
