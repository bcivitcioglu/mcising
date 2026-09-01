"""Every public symbol of the Rust extension's stub carries a docstring.

``python/mcising/_core.pyi`` is what mkdocstrings renders for
``IsingSimulation`` and what editors show, so an undocumented stub is an
undocumented public API.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

STUB = Path(__file__).resolve().parents[1] / "python" / "mcising" / "_core.pyi"


def _public_definitions() -> list[tuple[str, ast.AST]]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            found.append((node.name, node))
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (
                    not item.name.startswith("_") or item.name == "__new__"
                ):
                    found.append((f"{node.name}.{item.name}", item))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            found.append((node.name, node))
    return found


DEFINITIONS = _public_definitions()


def test_stub_declares_the_public_surface() -> None:
    names = {name for name, _ in DEFINITIONS}
    assert {
        "IsingSimulation",
        "IsingSimulation.sweep",
        "IsingSimulation.production_sweeps",
        "run_independent_temperatures",
        "run_parallel_tempering",
    } <= names
    assert len(DEFINITIONS) >= 25


@pytest.mark.parametrize("name,node", DEFINITIONS, ids=[n for n, _ in DEFINITIONS])
def test_public_symbol_has_docstring(name: str, node: ast.AST) -> None:
    docstring = ast.get_docstring(node)  # type: ignore[arg-type]
    assert docstring and len(docstring.split()) >= 3, f"{name} lacks a docstring"
