from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "anomaly_detection"
MODULES = {
    "core",
    "evaluation",
    "packs",
    "telecom",
    "oil_well",
    "runtime",
    "workflows",
    "cli",
}


def dependency_graph() -> dict[str, set[str]]:
    graph = {name: set() for name in MODULES}
    for origin in MODULES:
        path = PACKAGE / f"{origin}.py"
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue
            target = node.module.split(".", 1)[0]
            if target in MODULES and target != origin:
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
    def test_internal_module_graph_is_acyclic(self):
        graph = dependency_graph()
        self.assertEqual(find_cycle(graph), [], graph)

    def test_contract_modules_are_dependency_roots(self):
        graph = dependency_graph()
        self.assertEqual(graph["core"], set())
        self.assertEqual(graph["evaluation"], set())

    def test_runtime_never_depends_on_evaluation_or_sector_modules(self):
        graph = dependency_graph()
        self.assertFalse(
            graph["runtime"] & {"evaluation", "telecom", "oil_well", "workflows"},
            graph,
        )


if __name__ == "__main__":
    unittest.main()
