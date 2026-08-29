"""Tool-call prompting, parsing, and OpenAI response normalization.

Many local chat models emit their native tool syntax as text.  API clients cannot
execute that text; they need a structured ``tool_calls`` response.  This module
bridges common local-model formats to the OpenAI wire representation while still
preferring native structured calls from the inference backend.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOOL_RECOVERY_PROMPT = (
    "You said you would inspect or check something, but you did not emit a tool "
    "call. Continue now by calling the appropriate available tool. Do not merely "
    "describe the action."
)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a complete string.

    Thinking models (Qwen3, DeepSeek-R1, etc.) prefix responses with an
    internal scratchpad wrapped in <think> tags.  OpenAI-format clients have
    no structured field for reasoning content, so we strip it before sending.
    """
    if not text or "<think" not in text.lower():
        return text
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.lstrip("\n").strip()


class ThinkTagFilter:
    """Stateful streaming filter: suppress <think>...</think> blocks.

    Designed to be fed one raw chunk at a time.  Only the small tag-boundary
    window is buffered — the (potentially thousands-of-tokens) think content
    itself is discarded without being held in memory.

    Usage::

        flt = ThinkTagFilter()
        for raw_chunk in model_stream:
            visible = flt.feed(raw_chunk)
            if visible:
                send_to_client(visible)
        send_to_client(flt.finish())
    """

    _OPEN = "<think"
    _CLOSE = "</think>"
    _OPEN_KEEP = len("<think") - 1   # 5 — chars to keep to catch split opening tags
    _CLOSE_KEEP = len("</think>") - 1  # 7 — chars to keep to catch split closing tags

    def __init__(self) -> None:
        self._pending = ""
        self._in_think = False
        self._done = False  # True once the first </think> has been seen

    def feed(self, chunk: str) -> str:
        """Return the portion of *chunk* that should be sent to the client."""
        if not isinstance(chunk, str):
            return ""
        if self._done:
            return chunk  # Fast path after think block is closed

        self._pending += chunk
        output: List[str] = []

        while self._pending:
            if self._in_think:
                lo = self._pending.lower()
                pos = lo.find("</think>")
                if pos >= 0:
                    # Closing tag found — discard everything up to and
                    # including it, strip the trailing newline, then resume.
                    after = self._pending[pos + len("</think>"):]
                    self._in_think = False
                    self._done = True
                    self._pending = ""
                    output.append(after.lstrip("\n"))
                    break
                # Still inside think block — discard, keep only boundary.
                keep = min(self._CLOSE_KEEP, len(self._pending))
                self._pending = self._pending[-keep:]
                break
            else:
                lo = self._pending.lower()
                pos = lo.find("<think")
                if pos >= 0:
                    # Emit everything before <think.
                    output.append(self._pending[:pos])
                    rest = self._pending[pos:]
                    # Wait until we have the full opening tag (up to ">").
                    end = rest.find(">")
                    if end >= 0:
                        self._pending = rest[end + 1:]
                        self._in_think = True
                        # Loop: handle </think> in the same pending buffer.
                    else:
                        # Tag is split across chunks — buffer until next chunk.
                        self._pending = rest
                        break
                else:
                    # No opening tag yet; release safe prefix.
                    keep = min(self._OPEN_KEEP, len(self._pending))
                    if len(self._pending) > keep:
                        output.append(self._pending[:-keep])
                        self._pending = self._pending[-keep:]
                    break

        return "".join(output)

    def finish(self) -> str:
        """Flush remaining buffered content after the stream ends."""
        result = "" if self._in_think else self._pending
        self._pending = ""
        return result


class ToolTextStreamBuffer:
    """Release ordinary text while retaining possible native tool markup.

    Marker prefixes are kept across chunk boundaries, so a model emitting
    ``<tool_`` and ``call>`` in separate chunks never leaks the protocol markup
    to the user.  Once a marker begins, the remainder is retained for final
    structured parsing.
    """

    _MARKERS = (
        "<tool_call",
        "<function",
        "<parameter",
        "[tool_calls]",
        "<tool_use",
        "<tool>",
    )

    def __init__(self, hold_all: bool = False):
        self.hold_all = hold_all
        self.pending = ""
        self.raw_parts: List[str] = []
        self.tool_started = False
        self._keep = max(len(marker) for marker in self._MARKERS) - 1

    @property
    def raw_text(self) -> str:
        return "".join(self.raw_parts)

    def feed(self, text: Any) -> str:
        if not isinstance(text, str) or not text:
            return ""
        self.raw_parts.append(text)
        self.pending += text
        if self.tool_started or self.hold_all:
            return ""

        lowered = self.pending.lower()
        positions = [lowered.find(marker) for marker in self._MARKERS]
        positions = [position for position in positions if position >= 0]
        if positions:
            start = min(positions)
            safe = self.pending[:start]
            self.pending = self.pending[start:]
            self.tool_started = True
            return safe

        if len(self.pending) <= self._keep:
            return ""
        safe = self.pending[:-self._keep]
        self.pending = self.pending[-self._keep:]
        return safe

    def begin_native_tool(self) -> str:
        """Flush the safe prefix before backend-native structured deltas begin."""

        if self.tool_started:
            return ""
        self.tool_started = True
        safe = "" if self.hold_all else self.pending
        self.pending = ""
        return safe

    def finish(self) -> str:
        if self.tool_started:
            return ""
        safe = self.pending
        self.pending = ""
        return safe


def _tool_name_map(tools: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if isinstance(name, str) and name:
            names[name.lower()] = name
    return names


def tools_for_transformers(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Return tool schemas with object arguments in prior assistant calls.

    Transformers chat templates use the same outer shape as OpenAI tool schemas,
    but expect historical ``function.arguments`` values to be objects rather than
    JSON strings.  Tool definitions themselves can be copied unchanged.
    """

    return deepcopy(tools or [])


def messages_for_transformers(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    function["arguments"] = {"raw": arguments}
    return normalized


def build_tool_system_prompt(
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Any = "auto",
) -> str:
    """Build a fallback tool prompt for tokenizers without a tool-aware template."""

    schemas = []
    for tool in tools or []:
        function = tool.get("function", {})
        if function.get("name"):
            schemas.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {"type": "object"}),
                }
            )

    choice_instruction = "Use a tool when it is needed to complete the request."
    if tool_choice == "required":
        choice_instruction = "You must call one or more of the available tools."
    elif isinstance(tool_choice, dict):
        forced = tool_choice.get("function", {}).get("name")
        if forced:
            choice_instruction = f"You must call the {forced} tool."

    return (
        "You have access to external tools. "
        f"{choice_instruction} Never claim that you will inspect, check, search, "
        "read, or run something without emitting the tool call in the same turn. "
        "Emit each call exactly as <tool_call> followed by a JSON object with "
        'keys "name" and "arguments", followed by </tool_call>. The arguments '
        "must be a JSON object matching the selected tool schema. Available tools:\n"
        + json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    )


def inject_tool_prompt(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Any = "auto",
) -> List[Dict[str, Any]]:
    """Add fallback tool instructions without replacing the caller's system prompt."""

    if not tools or tool_choice == "none":
        return deepcopy(messages)
    prompt = build_tool_system_prompt(tools, tool_choice)
    result = deepcopy(messages)
    for message in result:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            message["content"] = f"{message['content']}\n\n{prompt}"
            return result
    return [{"role": "system", "content": prompt}] + result


def tokenizer_supports_tools(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        return "tool_use" in template or any("tools" in str(value) for value in template.values())
    return bool(template and "tools" in str(template))


def _coerce_value(value: str) -> Any:
    value = html.unescape(value.strip())
    if not value:
        return ""
    if value[0] in "[{\"" or value in ("true", "false", "null"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if re.fullmatch(r"-?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value):
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            pass
    return value


def _canonical_name(name: Any, names: Dict[str, str]) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if not names:
        return name
    if name.lower() in names:
        return names[name.lower()]
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    for k, v in names.items():
        if k.replace("-", "_") == normalized:
            return v
    return name


def _call_from_object(value: Any, names: Dict[str, str]) -> List[Tuple[str, Dict[str, Any]]]:
    if isinstance(value, list):
        calls: List[Tuple[str, Dict[str, Any]]] = []
        for item in value:
            calls.extend(_call_from_object(item, names))
        return calls
    if not isinstance(value, dict):
        return []

    function = value.get("function") if isinstance(value.get("function"), dict) else value
    name = _canonical_name(function.get("name"), names)
    if not name:
        return []
    arguments = function.get("arguments", function.get("parameters", function.get("input", {})))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return [(name, arguments)]


def _parse_jsonish(text: str, names: Dict[str, str]) -> List[Tuple[str, Dict[str, Any]]]:
    candidates = [text.strip()]
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        candidates.append(text[min(starts):].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        calls = _call_from_object(parsed, names)
        if calls:
            return calls

    # Some templates emit: function_name\n{"arg": "value"}
    match = re.match(r"^\s*([A-Za-z0-9_.:-]+)\s*\n\s*(\{.*\})\s*$", text, re.DOTALL)
    if match:
        name = _canonical_name(match.group(1), names)
        if name:
            try:
                arguments = json.loads(match.group(2))
                if isinstance(arguments, dict):
                    return [(name, arguments)]
            except json.JSONDecodeError:
                pass
    return []


def _parse_xml_function(text: str, names: Dict[str, str]) -> List[Tuple[str, Dict[str, Any]]]:
    # Match <function=Bash>, <function name="Bash">, <function:Bash>, etc.
    # Accepts closing tag or unclosed up to next <function, </tool_call>, or end of string.
    function_re = re.compile(
        r"<function(?:\s*=\s*|\s+name\s*=\s*[\"']?|:\s*|\s+)([A-Za-z0-9_.:-]+)[\"']?(?:\s*>|\s*\n)(.*?)(?:</function>|(?=<function|</tool_call|\Z))",
        re.IGNORECASE | re.DOTALL,
    )
    parameter_re = re.compile(
        r"<parameter(?:\s*=\s*|\s+name\s*=\s*[\"']?|:\s*|\s+)([A-Za-z0-9_.:-]+)[\"']?(?:\s*>|\s*\n)(.*?)(?:</parameter>|(?=<parameter|</function|</tool_call|\Z))",
        re.IGNORECASE | re.DOTALL,
    )
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for match in function_re.finditer(text):
        raw_name = match.group(1)
        name = _canonical_name(raw_name, names)
        if not name:
            continue
        body = match.group(2)
        arguments: Dict[str, Any] = {}
        for parameter in parameter_re.finditer(body):
            pname = parameter.group(1).strip()
            arguments[pname] = _coerce_value(parameter.group(2).strip())
        if not arguments:
            tag_pattern = re.compile(r"<([A-Za-z0-9_-]+)>(.*?)(?:</\1>|(?=<\w+|\Z))", re.DOTALL)
            for tmatch in tag_pattern.finditer(body):
                tname = tmatch.group(1).strip()
                if tname.lower() not in ("function", "tool_call", "tool_calls", "parameter"):
                    arguments[tname] = _coerce_value(tmatch.group(2).strip())
        if not arguments and body.strip():
            try:
                jargs = json.loads(body.strip())
                if isinstance(jargs, dict):
                    arguments = jargs
            except Exception:
                pass
        calls.append((name, arguments))
    return calls


def _structured_call(name: str, arguments: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def parse_tool_calls(
    text: Optional[str],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Parse common local-model tool syntaxes and remove them from visible text."""

    if not text:
        return text, []
    names = _tool_name_map(tools)
    raw_calls: List[Tuple[str, Dict[str, Any]]] = []
    spans: List[Tuple[int, int]] = []

    block_re = re.compile(r"<tool_call\b[^>]*>(.*?)(?:</tool_call>|\Z)", re.IGNORECASE | re.DOTALL)
    for match in block_re.finditer(text):
        calls = _parse_jsonish(match.group(1), names) or _parse_xml_function(match.group(1), names)
        if calls:
            raw_calls.extend(calls)
            spans.append(match.span())

    # Some clients/models omit the outer <tool_call> wrapper.
    if not raw_calls:
        xml_calls = _parse_xml_function(text, names)
        if xml_calls:
            raw_calls.extend(xml_calls)
            for match in re.finditer(
                r"<function(?:\s*=\s*|\s+name\s*=).*?(?:</function>|(?=<function|</tool_call|\Z))",
                text,
                re.IGNORECASE | re.DOTALL,
            ):
                spans.append(match.span())

    # Mistral-style marker followed by a JSON array/object.
    if not raw_calls:
        marker = re.search(r"\[TOOL_CALLS\]\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if marker:
            calls = _parse_jsonish(marker.group(1), names)
            if calls:
                raw_calls.extend(calls)
                spans.append((marker.start(), len(text)))

    # A forced-call retry may produce only the JSON object, without wrapper tags.
    if not raw_calls and text.strip().startswith(("{", "[")):
        calls = _parse_jsonish(text, names)
        if calls:
            raw_calls.extend(calls)
            spans.append((0, len(text)))

    if not raw_calls:
        return text, []

    clean = text
    for start, end in sorted(spans, reverse=True):
        clean = clean[:start] + clean[end:]
    clean = re.sub(r"</?tool_call\b[^>]*>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"</?function\b[^>]*>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"</?parameter\b[^>]*>", "", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    structured = [_structured_call(name, arguments, i) for i, (name, arguments) in enumerate(raw_calls)]
    return clean or None, structured


def normalize_openai_message(
    message: Optional[Dict[str, Any]],
    finish_reason: Optional[str],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Normalize native or textual calls into an OpenAI assistant message."""

    normalized = deepcopy(message or {"role": "assistant", "content": ""})
    normalized.setdefault("role", "assistant")
    native_calls = normalized.get("tool_calls") or []
    if native_calls:
        for index, call in enumerate(native_calls):
            call.setdefault("id", f"call_{uuid.uuid4().hex[:24]}")
            call.setdefault("type", "function")
            function = call.setdefault("function", {})
            arguments = function.get("arguments", {})
            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                )
        normalized["content"] = normalized.get("content") or None
        return normalized, "tool_calls"

    content = normalized.get("content")
    if isinstance(content, str):
        clean, parsed_calls = parse_tool_calls(content, tools)
        normalized["content"] = clean
        if parsed_calls:
            normalized["tool_calls"] = parsed_calls
            return normalized, "tool_calls"
    return normalized, finish_reason or "stop"


def tool_choice_requires_call(tool_choice: Any) -> bool:
    return tool_choice == "required" or isinstance(tool_choice, dict)


def looks_like_abandoned_tool_intent(text: Any) -> bool:
    """Detect a short promise to act that contains no actual result or call."""

    if not isinstance(text, str) or not text.strip() or len(text) > 400:
        return False
    pattern = re.compile(
        r"\b(?:i(?:'ll|\s+will|\s+am\s+going\s+to)|let\s+me|i\s+can)\s+"
        r"(?:now\s+)?(?:check|inspect|search|look|read|run|use|verify|find|open|examine)\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))
