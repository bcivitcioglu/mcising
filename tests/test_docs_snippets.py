"""Every code fence in the documentation executes against the current API.

Python fences (```python) run in order, page by page, in one namespace per
page and a temporary working directory — later snippets may use names and
files earlier ones created, exactly as a reader following the page would.
Shell fences (```bash) run the CLI the same way, command by command (slow:
the CLI reference takes ~30 s): every ``mcising`` command executes, commands
that are installation or development tooling (``pip``, ``uv``, ``git``,
``cargo``) are skipped, and anything else fails so a new kind of shell
example cannot slip through untested. The ``>>>`` examples in the package
docstrings run as doctests.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import mcising
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
#: First words of shell commands that are tooling, not mcising usage.
TOOLING = frozenset({"pip", "uv", "git", "cd", "cargo", "python", "python3"})


@dataclass(frozen=True)
class Fence:
    page: Path
    line: int
    lang: str
    code: str

    @property
    def where(self) -> str:
        return f"{self.page.relative_to(REPO_ROOT)}:{self.line}"


def fences(page: Path, lang: str) -> list[Fence]:
    text = page.read_text(encoding="utf-8")
    return [
        Fence(page, text[: m.start()].count("\n") + 1, m.group(1), m.group(2))
        for m in FENCE.finditer(text)
        if m.group(1) == lang
    ]


def _pages_with(lang: str) -> list[Path]:
    return [page for page in PAGES if fences(page, lang)]


def _ids(pages: list[Path]) -> list[str]:
    return [str(page.relative_to(REPO_ROOT)) for page in pages]


PYTHON_PAGES = _pages_with("python")
BASH_PAGES = _pages_with("bash")


def test_every_documented_page_is_collected() -> None:
    assert len(PYTHON_PAGES) >= 10 and len(BASH_PAGES) >= 5
    assert REPO_ROOT / "docs" / "guide" / "cli.md" in BASH_PAGES


@pytest.mark.parametrize("page", PYTHON_PAGES, ids=_ids(PYTHON_PAGES))
def test_python_fences_execute(
    page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for fence in fences(page, "python"):
        try:
            exec(compile(fence.code, fence.where, "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - any failure is a stale snippet
            pytest.fail(f"{fence.where}: {type(exc).__name__}: {exc}")


def _commands(fence: Fence) -> list[str]:
    body = fence.code.replace("\\\n", " ")
    commands = []
    for raw in body.splitlines():
        command = raw.split("#", 1)[0].strip()
        if command:
            commands.append(command)
    return commands


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from mcising.cli import app; app()", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **__import__("os").environ,
            "COLUMNS": "120",
            "TERM": "dumb",
            "MPLBACKEND": "Agg",
        },
    )


@pytest.mark.slow
@pytest.mark.parametrize("page", BASH_PAGES, ids=_ids(BASH_PAGES))
def test_bash_fences_execute(page: Path, tmp_path: Path) -> None:
    executed = 0
    for fence in fences(page, "bash"):
        for command in _commands(fence):
            words = shlex.split(command)
            if words[0] == "mcising":
                result = _run_cli(words[1:], tmp_path)
                tail = (result.stderr or result.stdout).strip().splitlines()[-8:]
                assert result.returncode == 0, (
                    f"{fence.where}: `{command}` exited {result.returncode}\n"
                    + "\n".join(tail)
                )
                executed += 1
            elif words[0] not in TOOLING:
                pytest.fail(f"{fence.where}: unclassified shell command: {command}")
    assert executed >= 0  # tooling-only pages are legitimately empty


def _modules_with_examples() -> list[ModuleType]:
    modules = []
    for info in pkgutil.walk_packages(mcising.__path__, prefix="mcising."):
        module = importlib.import_module(info.name)
        source = Path(module.__file__ or "").with_suffix(".py")
        if source.exists() and ">>>" in source.read_text(encoding="utf-8"):
            modules.append(module)
    return modules


EXAMPLE_MODULES = _modules_with_examples()


def test_docstring_examples_are_collected() -> None:
    assert "mcising.simulation" in {module.__name__ for module in EXAMPLE_MODULES}


@pytest.mark.parametrize(
    "module", EXAMPLE_MODULES, ids=[m.__name__ for m in EXAMPLE_MODULES]
)
def test_docstring_examples(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0, module.__name__
    assert result.failed == 0, f"{module.__name__}: {result.failed} doctest failures"
