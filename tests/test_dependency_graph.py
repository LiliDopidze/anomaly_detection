from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PREFIXES = (
    "telemetry_contract",
    "telemetry_eval_contract",
    "telemetry_packs",
    "telemetry_adapters",
    "telemetry_runtime",
)


def package_name(path: Path) -> str:
    relative = path.relative_to(SRC)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_root(module: str) -> str | None:
    return next((prefix for prefix in PREFIXES if module.startswith(prefix)), None)


def dependency_graph() -> dict[str, set[str]]:
    graph = {prefix: set() for prefix in PREFIXES}
    for path in SRC.rglob("*.py"):
        origin = package_root(package_name(path))
        if origin is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                target = package_root(name)
                if target is not None and target != origin:
                    graph[origin].add(target)
    return graph


def find_cycle(graph: dict[str, set[str]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, path: list[str]) -> list[str]:
        if node in visiting:
            return path[path.index(node) :] + [node]
        if node in visited:
            return []
        visiting.add(node)
        for child in graph[node]:
            cycle = visit(child, path + [child])
            if cycle:
                return cycle
        visiting.remove(node)
        visited.add(node)
        return []

    for node in graph:
        cycle = visit(node, [node])
        if cycle:
            return cycle
    return []


class DependencyGraphTests(unittest.TestCase):
    def test_internal_package_graph_is_acyclic(self):
        graph = dependency_graph()
        self.assertEqual(find_cycle(graph), [], graph)

    def test_contract_does_not_depend_on_pack_adapter_runtime_or_eval(self):
        graph = dependency_graph()
        self.assertEqual(graph["telemetry_contract"], set())

    def test_runtime_depends_on_no_other_internal_package(self):
        graph = dependency_graph()
        self.assertEqual(graph["telemetry_runtime"], set())


if __name__ == "__main__":
    unittest.main()
