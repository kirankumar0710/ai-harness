"""Tools: the hands of the agent.

A tool is three things: a name, a schema (the contract the model sees)
and a function. The six core tools below - read, write, edit, bash, ls,
grep - are enough to build real software. Add your own with ``@tool``.

Design rules:

* Schemas are plain JSON Schema function declarations
  (``{"name", "description", "parameters"}``). Each provider translates
  them into its own wire format, so tools never know which vendor runs
  them.
* Parameter types come from the function's annotations (``str``,
  ``int``, ``float``, ``bool``); anything else is declared as a string.
  Arguments are coerced to those types before the call, because models
  sometimes send ``"120"`` where ``120`` was asked for.
* Every file path is jailed to the working directory by ``resolve()``.
  Behaviour (what may run at all) is policed in ``security.py``.
* Tools return text. Expected failures are returned as ``ERROR: ...``
  strings the model can act on; anything unexpected is raised and the
  loop turns it into an error result.
"""

from __future__ import annotations

import fnmatch
import inspect
import os
import re
import subprocess
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .providers import ToolSchema

# Names every vendor accepts (Gemini is the strictest: 64 chars max).
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")

JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

SKIPPED_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv"})

MAX_READ_LINES = 4000
MAX_OUTPUT_CHARS = 12000
MAX_LIST_ENTRIES = 500
MAX_GREP_HITS = 200
MAX_GREP_LINE_CHARS = 200
DEFAULT_BASH_TIMEOUT = 120


@dataclass(frozen=True)
class Tool:
    """A callable the model may request by name.

    Satisfies ``loop.Tool``: ``spec["schema"]`` is the neutral JSON
    Schema declaration sent to the provider.
    """

    name: str
    spec: Mapping[str, ToolSchema]
    func: Callable[..., Any] = field(repr=False)
    param_types: Mapping[str, type] = field(default_factory=dict,
                                            repr=False)

    @property
    def schema(self) -> ToolSchema:
        return self.spec["schema"]

    def run(self, **kwargs: Any) -> Any:
        """Coerce the model's arguments to declared types, then call."""
        unknown = sorted(set(kwargs) - set(self.param_types))
        if unknown:
            raise TypeError(f"unexpected argument(s): {', '.join(unknown)}")
        args = {name: _coerce(value, self.param_types[name])
                for name, value in kwargs.items()}
        return self.func(**args)


# --------------------------------------------------------------------------
# The @tool decorator
# --------------------------------------------------------------------------


def tool(description: str, **params: str) -> Callable[[Callable], Tool]:
    """Turn a plain function into a ``Tool``.

        @tool("Read a file", path="Path to the file")
        def read_file(path: str) -> str: ...

    ``params`` maps each parameter name to the description the model
    sees. Every parameter must be described: an undocumented parameter
    is one the model will guess at. Parameters with defaults are
    optional; the rest are required.
    """
    def wrap(func: Callable) -> Tool:
        name = func.__name__
        if not TOOL_NAME_PATTERN.match(name):
            raise ValueError(f"invalid tool name {name!r}")

        signature = inspect.signature(func)
        hints = typing.get_type_hints(func)
        arg_names = list(signature.parameters)

        missing = [a for a in arg_names if a not in params]
        extra = [p for p in params if p not in signature.parameters]
        if missing or extra:
            raise ValueError(
                f"{name}: descriptions must match parameters "
                f"(missing: {missing or 'none'}, unknown: {extra or 'none'})")

        param_types = {a: _json_type_of(hints.get(a, str))
                       for a in arg_names}
        properties = {
            a: {"type": JSON_TYPES.get(param_types[a], "string"),
                "description": params[a]}
            for a in arg_names
        }
        required = [a for a, p in signature.parameters.items()
                    if p.default is inspect.Parameter.empty]
        schema: dict[str, Any] = {
            "name": name,
            "description": description,
            "parameters": {"type": "object",
                           "properties": properties,
                           "required": required},
        }
        return Tool(name=name, spec={"schema": schema}, func=func,
                    param_types=param_types)
    return wrap


def tool_map(tools: Iterable[Tool]) -> dict[str, Tool]:
    """Index tools by name, as ``run_loop`` expects. Names must be unique."""
    mapping: dict[str, Tool] = {}
    for item in tools:
        if item.name in mapping:
            raise ValueError(f"duplicate tool name {item.name!r}")
        mapping[item.name] = item
    return mapping


def _json_type_of(hint: Any) -> type:
    """Reduce an annotation to one of the ``JSON_TYPES`` keys.

    ``Optional[int]`` / ``int | None`` become ``int``; unknown or
    compound annotations fall back to ``str``.
    """
    if hint in JSON_TYPES:
        return hint
    options = [a for a in typing.get_args(hint) if a is not type(None)]
    if len(options) == 1 and options[0] in JSON_TYPES:
        return options[0]
    return str


def _coerce(value: Any, target: type) -> Any:
    """Convert a model-supplied argument to the declared type."""
    if value is None or (isinstance(value, target) and not (
            target is int and isinstance(value, bool))):
        return value
    if target is bool:
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if target is int:
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"expected an integer, got {value!r}")
        return int(number)
    return target(value)


def _glob_match(rel: str, name: str, pattern: str) -> bool:
    """Match a file against a glob, where ``**/`` may match no folders.

    ``fnmatch`` treats ``**/*`` as "something, a slash, something", so on
    its own it never matches files at the top of the tree. Trying the
    pattern with each ``**/`` removed as well fixes that.
    """
    variants = {pattern, pattern.replace("/**/", "/")}
    variants |= {v[3:] for v in variants if v.startswith("**/")}
    return any(fnmatch.fnmatch(rel, v) or fnmatch.fnmatch(name, v)
               for v in variants)


# --------------------------------------------------------------------------
# The six core tools
# --------------------------------------------------------------------------


def core_tools(workdir: str | os.PathLike[str]) -> list[Tool]:
    """Return the six core tools, jailed to ``workdir``."""
    root = os.path.realpath(workdir)

    def resolve(path: str) -> str:
        """Map a relative path into ``root``; refuse anything outside it.

        ``realpath`` follows symlinks first, so a link pointing out of
        the jail is caught as well as ``../`` tricks.
        """
        full = os.path.realpath(os.path.join(root, path))
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"{path!r} escapes the working directory")
        return full

    def walk() -> Iterator[tuple[str, str]]:
        """Yield (relative_path, file_name) for every file under root."""
        for base, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRS)
            for name in sorted(files):
                yield os.path.relpath(os.path.join(base, name), root), name

    @tool("Read a file. Returns its contents with line numbers.",
          path="Path relative to the working directory")
    def read_file(path: str) -> str:
        with open(resolve(path), encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        numbered = "".join(f"{i}\t{line}"
                           for i, line in enumerate(lines[:MAX_READ_LINES],
                                                    start=1))
        if total > MAX_READ_LINES:
            numbered += f"... truncated, {total} lines total\n"
        return numbered or "(empty file)"

    @tool("Create or overwrite a file with the given content.",
          path="Path relative to the working directory",
          content="Full file content to write")
    def write_file(path: str, content: str) -> str:
        full = resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

    @tool("Replace an exact text snippet in a file. "
          "The snippet must appear exactly once.",
          path="Path relative to the working directory",
          old="Exact text to find (must be unique in the file)",
          new="Replacement text")
    def edit_file(path: str, old: str, new: str) -> str:
        full = resolve(path)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        count = text.count(old) if old else 0
        if count == 0:
            return ("ERROR: snippet not found - read the file and copy "
                    "it exactly")
        if count > 1:
            return (f"ERROR: snippet appears {count} times - include more "
                    "context to make it unique")
        with open(full, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        return f"Edited {path}"

    @tool("Run a shell command in the working directory. "
          "Returns stdout and stderr.",
          command="The shell command to run",
          timeout=f"Seconds before the command is killed "
                  f"(default {DEFAULT_BASH_TIMEOUT})")
    def bash(command: str, timeout: int = DEFAULT_BASH_TIMEOUT) -> str:
        try:
            done = subprocess.run(
                command, shell=True, cwd=root, capture_output=True,
                text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"
        output = (done.stdout + done.stderr).strip()
        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = f"{output[:half]}\n... truncated ...\n{output[-half:]}"
        if done.returncode != 0:
            output = f"{output}\n(exit {done.returncode})".lstrip()
        return output or "(exit 0, no output)"

    @tool("List files matching a glob pattern, recursively.",
          pattern="Glob such as '**/*.py' or 'src/*' (default: everything)")
    def list_files(pattern: str = "**/*") -> str:
        matches = sorted(rel for rel, name in walk()
                         if _glob_match(rel, name, pattern))
        listing = "\n".join(matches[:MAX_LIST_ENTRIES])
        if len(matches) > MAX_LIST_ENTRIES:
            listing += f"\n... and {len(matches) - MAX_LIST_ENTRIES} more"
        return listing or "(no matches)"

    @tool("Search file contents with a regular expression. "
          "Returns matching lines as path:line: text.",
          regex="Regular expression to search for",
          pattern="Optional filename glob to limit the search, e.g. '*.py'")
    def grep(regex: str, pattern: str = "*") -> str:
        try:
            matcher = re.compile(regex)
        except re.error as error:
            return f"ERROR: invalid regular expression: {error}"
        hits: list[str] = []
        for rel, name in walk():
            if not fnmatch.fnmatch(name, pattern):
                continue
            try:
                with open(os.path.join(root, rel), encoding="utf-8",
                          errors="ignore") as f:
                    for number, line in enumerate(f, start=1):
                        if matcher.search(line):
                            text = line.rstrip()[:MAX_GREP_LINE_CHARS]
                            hits.append(f"{rel}:{number}: {text}")
                            if len(hits) >= MAX_GREP_HITS:
                                return "\n".join(hits) + "\n... truncated"
            except OSError:
                continue
        return "\n".join(hits) or "(no matches)"

    return [read_file, write_file, edit_file, bash, list_files, grep]
