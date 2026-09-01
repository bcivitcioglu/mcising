"""Every published performance number comes from ``benchmarks/run_all.py``.

Layers: the two budgets, the pure renderers, the marker-block plumbing that
writes the docs, the committed ``benchmarks/results.json`` against the
committed pages, a scan of every page for quantitative speed claims outside
a generated block, and a quick-budget end-to-end run of the script itself
(one timing child process, like ``tests/test_lazy_imports.py``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "run_all.py"
#: Archived pre-1.0 narrative; its estimates describe code that no longer
#: exists and are framed as history, not as claims about the current core.
EXCLUDED_PAGES = frozenset({REPO_ROOT / "docs" / "advanced" / "history.md"})


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_all", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_all"] = module
    spec.loader.exec_module(module)
    return module


run_all = _load_script()


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    document: dict[str, Any] = run_all.load_document(run_all.RESULTS_PATH)
    return document


@pytest.fixture(scope="module")
def quick_document() -> dict[str, Any]:
    context = run_all.Context(log=lambda _: None, skip_peapods=True)
    document: dict[str, Any] = run_all.run_all(run_all.QUICK_BUDGET, context)
    return document


def _peapods_row(lattice: str, delta_percent: float) -> dict[str, Any]:
    dim = 3 if lattice == "cubic" else 2
    return {
        "label": lattice.capitalize(),
        "lattice": lattice,
        "geometry": None,
        "dim": dim,
        "size": 8,
        "temperature": 2.0,
        "mcising_energy": -1.4,
        "mcising_energy_error": 0.001,
        "peapods_energy": -1.4 * (1 + delta_percent / 100),
        "peapods_energy_error": None,
        "delta_percent": delta_percent,
        "agreement": delta_percent <= run_all.AGREEMENT_LIMIT_PERCENT,
        "mcising_median_seconds": 1.0,
        "peapods_median_seconds": 3.0,
    }


def _with_peapods(
    document: dict[str, Any], status: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    out = copy.deepcopy(document)
    out["sections"]["peapods"].update({"status": status, "rows": rows})
    out["peapods_version"] = "0.2.0"
    return out


class TestBudget:
    def test_quick_never_exceeds_full(self) -> None:
        quick, full = asdict(run_all.QUICK_BUDGET), asdict(run_all.FULL_BUDGET)
        assert quick["quick"] is True and full["quick"] is False

        def walk(q: Any, f: Any, path: str) -> None:
            if isinstance(q, dict):
                assert q.keys() == f.keys(), path
                for key in q:
                    walk(q[key], f[key], f"{path}.{key}")
            elif isinstance(q, bool):
                return
            elif isinstance(q, (int, float)):
                assert q <= f, path
            elif isinstance(q, (list, tuple)):
                assert max(_flatten(q)) <= max(_flatten(f)), path

        walk(quick, full, "budget")

    def test_section_registry_is_complete(self) -> None:
        assert set(run_all.SECTION_NAMES) <= set(run_all.RENDERERS)
        assert {"headline", "index-card"} <= set(run_all.RENDERERS)
        assert {section for _, section in run_all.DOC_BLOCKS} <= set(run_all.RENDERERS)


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in _flatten(item)]
    return [float(value)]


class TestRendering:
    def test_every_section_renders_a_table(
        self, quick_document: dict[str, Any]
    ) -> None:
        for name in run_all.SECTION_NAMES:
            if name == "peapods":
                continue
            text = run_all.render_section(name, quick_document)
            lines = text.split("\n")
            assert lines[0].startswith("| ") and lines[1].startswith("|---"), name
            width = lines[0].count("|")
            assert all(
                line.count("|") == width for line in lines if line.startswith("|")
            ), name

    def test_headline_and_card(self, quick_document: dict[str, Any]) -> None:
        headline = run_all.render_headline(quick_document)
        assert "spin updates per second" in headline
        assert "peapods" not in headline.split("\n")[0]
        assert run_all.REGENERATE_COMMAND in headline
        card = run_all.render_index_card(quick_document)
        assert "\n" not in card and "spin updates per second" in card

    def test_render_is_deterministic(self, quick_document: dict[str, Any]) -> None:
        for name, renderer in run_all.RENDERERS.items():
            assert renderer(quick_document) == renderer(quick_document), name

    def test_unknown_block_is_rejected(self, quick_document: dict[str, Any]) -> None:
        with pytest.raises(run_all.BenchmarkError, match="no renderer"):
            run_all.render_section("nope", quick_document)

    def test_peapods_unavailable(self, quick_document: dict[str, Any]) -> None:
        text = run_all.render_peapods(quick_document)
        assert "was not run" in text and "|" not in text

    def test_peapods_matched_partial_deferred(
        self, quick_document: dict[str, Any]
    ) -> None:
        matched = _with_peapods(
            quick_document, "matched", [_peapods_row("square", 0.1)]
        )
        text = run_all.render_peapods(matched)
        assert "| Square 8×8 |" in text and "Not published" not in text
        assert "3.0×" in text
        assert "peapods" in run_all.render_headline(matched).split("\n")[0]

        partial = _with_peapods(
            quick_document,
            "partial",
            [_peapods_row("square", 0.1), _peapods_row("cubic", 4.0)],
        )
        text = run_all.render_peapods(partial)
        assert (
            "| Square 8×8 |" in text
            and "Cubic 8×8×8" not in text.split("Not published")[0]
        )
        assert "Not published" in text and "4.00 %" in text

        deferred = _with_peapods(
            quick_document, "deferred", [_peapods_row("square", 2.0)]
        )
        text = run_all.render_peapods(deferred)
        assert "|" not in text and "Not published" in text
        assert "peapods" not in run_all.render_headline(deferred).split("\n")[0]
        assert "Wolff and Swendsen-Wang are not compared" in text


class TestDocsBlocks:
    def test_roundtrip_unindented(self, tmp_path: Path) -> None:
        begin, end = run_all.markers("demo")
        page = tmp_path / "page.md"
        page.write_text(f"intro\n{begin}\nold\n{end}\noutro\n", encoding="utf-8")
        run_all.write_block(page, "demo", "| a |\n|---|\n| 1 |")
        assert page.read_text(encoding="utf-8") == (
            f"intro\n{begin}\n| a |\n|---|\n| 1 |\n{end}\noutro\n"
        )
        assert run_all.read_block(page, "demo") == "| a |\n|---|\n| 1 |"

    def test_roundtrip_indented_keeps_blank_lines_empty(self, tmp_path: Path) -> None:
        begin, end = run_all.markers("card")
        page = tmp_path / "index.md"
        page.write_text(
            f"-   item\n\n    {begin}\n    old\n    {end}\n\n    link\n",
            encoding="utf-8",
        )
        run_all.write_block(page, "card", "first\n\nsecond")
        text = page.read_text(encoding="utf-8")
        assert text == (
            f"-   item\n\n    {begin}\n    first\n\n    second\n    {end}\n\n    link\n"
        )
        assert not any(line != line.rstrip() for line in text.split("\n"))
        assert run_all.read_block(page, "card") == "first\n\nsecond"

    def test_requires_exactly_one_pair(self, tmp_path: Path) -> None:
        begin, end = run_all.markers("x")
        page = tmp_path / "page.md"
        page.write_text("no markers\n", encoding="utf-8")
        with pytest.raises(run_all.BenchmarkError, match="exactly one"):
            run_all.write_block(page, "x", "block")
        page.write_text(f"{begin}\n{end}\n{begin}\n{end}\n", encoding="utf-8")
        with pytest.raises(run_all.BenchmarkError, match="exactly one"):
            run_all.read_block(page, "x")
        page.write_text(f"{end}\n{begin}\n", encoding="utf-8")
        with pytest.raises(run_all.BenchmarkError, match="in order"):
            run_all.read_block(page, "x")

    def test_marker_must_stand_alone(self, tmp_path: Path) -> None:
        begin, end = run_all.markers("x")
        page = tmp_path / "page.md"
        page.write_text(f"text {begin}\nbody\n{end}\n", encoding="utf-8")
        with pytest.raises(run_all.BenchmarkError, match="exactly one"):
            run_all.read_block(page, "x")

    def test_check_docs_reports_stale_block(
        self, tmp_path: Path, quick_document: dict[str, Any]
    ) -> None:
        begin, end = run_all.markers("scaling")
        page = tmp_path / "perf.md"
        page.write_text(f"# t\n\n{begin}\nstale\n{end}\n", encoding="utf-8")
        blocks = [(page, "scaling")]
        stale = run_all.check_docs(quick_document, blocks)
        assert len(stale) == 1 and stale[0].endswith(":scaling")
        run_all.write_block(page, "scaling", run_all.render_scaling(quick_document))
        assert run_all.check_docs(quick_document, blocks) == []

    def test_write_docs_refuses_quick_document(
        self, tmp_path: Path, quick_document: dict[str, Any]
    ) -> None:
        begin, end = run_all.markers("scaling")
        page = tmp_path / "perf.md"
        page.write_text(f"{begin}\nx\n{end}\n", encoding="utf-8")
        with pytest.raises(run_all.BenchmarkError, match="quick"):
            run_all.write_docs(quick_document, [(page, "scaling")])
        assert run_all.read_block(page, "scaling") == "x"


class TestCommittedResults:
    """The committed run behind README.md and the docs."""

    def test_schema_and_provenance(self, committed: dict[str, Any]) -> None:
        assert committed["schema_version"] == run_all.SCHEMA_VERSION
        assert set(committed) >= {
            "generated_utc",
            "mcising_version",
            "git_commit",
            "python",
            "platform",
            "machine",
            "peapods_version",
            "budget",
            "sections",
            "elapsed_seconds",
        }
        assert committed["machine"]["cpu"] and committed["machine"]["cpu_count"]
        assert committed["peapods_version"] is not None

    def test_used_the_full_budget(self, committed: dict[str, Any]) -> None:
        expected = json.loads(json.dumps(asdict(run_all.FULL_BUDGET)))
        assert committed["budget"] == expected

    def test_every_section_present(self, committed: dict[str, Any]) -> None:
        assert set(committed["sections"]) == set(run_all.SECTION_NAMES)
        for name, section in committed["sections"].items():
            assert section["generated_utc"] and section["elapsed_seconds"] > 0, name

    def test_committed_run_was_a_release_build(self, committed: dict[str, Any]) -> None:
        square = committed["sections"]["lattices"]["rows"][0]
        assert square["lattice"] == "square"
        # A debug build is ~10x slower; committing one would poison every ratio.
        assert square["attempted_updates"] / square["median_seconds"] >= 1e8

    def test_peapods_rows_are_consistent(self, committed: dict[str, Any]) -> None:
        peapods = committed["sections"]["peapods"]
        assert peapods["status"] in {"matched", "partial", "deferred"}
        assert peapods["rows"], "the committed run must have exercised peapods"
        for row in peapods["rows"]:
            expected = row["delta_percent"] <= run_all.AGREEMENT_LIMIT_PERCENT
            assert row["agreement"] is expected, row["label"]

    @pytest.mark.parametrize(
        ("path", "section"),
        run_all.DOC_BLOCKS,
        ids=[f"{p.relative_to(REPO_ROOT)}:{s}" for p, s in run_all.DOC_BLOCKS],
    )
    def test_docs_block_matches_the_json(
        self, committed: dict[str, Any], path: Path, section: str
    ) -> None:
        assert run_all.read_block(path, section) == run_all.render_section(
            section, committed
        ), (
            f"{path.relative_to(REPO_ROOT)}:{section} is stale: run "
            "`uv run python benchmarks/run_all.py --from-json "
            "benchmarks/results.json --write-docs`"
        )


SPEED_WORD = re.compile(r"\b(faster|slower|speed-?ups?)\b", re.IGNORECASE)
THROUGHPUT = re.compile(
    r"updates\s*/\s*s(ec)?\b|\d\s?M\s+(spin\s+)?updates", re.IGNORECASE
)
TIMING = re.compile(r"\d(\.\d+)?\s?(ms|µs|μs)\b")
MARKER = re.compile(r"\s*<!-- benchmarks:[\w-]+:(begin|end) -->\s*")
DIGIT = re.compile(r"\d")


def _pages() -> list[Path]:
    pages = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    return [page for page in pages if page not in EXCLUDED_PAGES]


def test_every_quantitative_speed_claim_is_generated() -> None:
    """A line with a digit and a speed word, throughput or timing must sit
    inside a ``benchmarks:*`` block — i.e. come from the script."""
    offenders = []
    for page in _pages():
        inside = False
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            marker = MARKER.fullmatch(line)
            if marker:
                inside = marker.group(1) == "begin"
                continue
            if inside or not DIGIT.search(line):
                continue
            if (
                SPEED_WORD.search(line)
                or THROUGHPUT.search(line)
                or TIMING.search(line)
            ):
                offenders.append(
                    f"{page.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
                )
    assert not offenders, "hand-typed performance numbers:\n" + "\n".join(offenders)


class TestQuickRun:
    def test_quick_run_end_to_end(self, quick_document: dict[str, Any]) -> None:
        assert set(quick_document["sections"]) == set(run_all.SECTION_NAMES)
        json.dumps(quick_document, allow_nan=False)
        peapods = quick_document["sections"]["peapods"]
        assert peapods["status"] == "unavailable" and peapods["reason"] == "skipped"
        modes = [row["mode"] for row in quick_document["sections"]["parallel"]["rows"]]
        assert modes == ["cooldown", "independent"]

    def test_parallel_section_with_fake_timer(self) -> None:
        calls: list[tuple[int, int]] = []

        def fake_timer(
            config: dict[str, Any], repeats: int, threads: int
        ) -> list[float]:
            calls.append((repeats, threads))
            assert config["mode"] in {"independent", "parallel_tempering"}
            return [0.8 / threads] * repeats

        budget = replace(
            run_all.QUICK_BUDGET,
            parallel=replace(
                run_all.QUICK_BUDGET.parallel,
                thread_counts=(1, 2, 4),
                include_parallel_tempering=True,
                repeats=2,
            ),
        )
        context = run_all.Context(log=lambda _: None, timer=fake_timer)
        section = run_all.run_parallel(budget, context)
        assert [c[0] for c in calls] == [2, 2, 2, 2]
        assert [c[1] for c in calls][:3] == [1, 2, 4]
        rows = section["rows"]
        assert [r["mode"] for r in rows] == [
            "cooldown",
            "independent",
            "independent",
            "independent",
            "parallel_tempering",
        ]
        assert rows[2]["median_seconds"] == pytest.approx(0.4)
        document = run_all.provenance(budget)
        document["sections"]["parallel"] = section
        assert "| Parallel tempering |" in run_all.render_parallel(document)

    def test_merge_sections(self, quick_document: dict[str, Any]) -> None:
        fresh = copy.deepcopy(quick_document)
        fresh["generated_utc"] = "later"
        fresh["sections"] = {"scaling": {"rows": [], "generated_utc": "later"}}
        merged = run_all.merge_sections(quick_document, fresh)
        assert set(merged["sections"]) == set(run_all.SECTION_NAMES)
        assert merged["sections"]["scaling"] == fresh["sections"]["scaling"]
        assert merged["sections"]["lattices"] == quick_document["sections"]["lattices"]
        assert merged["generated_utc"] == "later"
        bad_schema = {**fresh, "schema_version": -1}
        with pytest.raises(run_all.BenchmarkError, match="schema_version"):
            run_all.merge_sections(quick_document, bad_schema)
        bad_budget = {**fresh, "budget": {}}
        with pytest.raises(run_all.BenchmarkError, match="budget"):
            run_all.merge_sections(quick_document, bad_budget)

    def test_unknown_section_is_rejected(self) -> None:
        with pytest.raises(run_all.BenchmarkError, match="unknown section"):
            run_all.run_all(run_all.QUICK_BUDGET, sections=["nope"])
