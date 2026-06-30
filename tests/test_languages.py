"""Phase 8A: Go / Rust / Java tree-sitter symbol extraction.

Each language auto-skips if its optional grammar package isn't installed, so a
missing grammar never fails CI (mirrors the indexer's graceful degradation).
"""

import importlib.util

import pytest

from agent.retrieval.tree_sitter_indexer import TreeSitterIndexer


def _grammar_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@pytest.fixture(scope="module")
def indexer():
    return TreeSitterIndexer()


def _symbols_by_type(symbols):
    out = {}
    for s in symbols:
        out.setdefault(s.type, set()).add(s.name)
    return out


# --- Go ----------------------------------------------------------------------

GO_SOURCE = """package main

type Point struct {
\tX int
\tY int
}

type Shape interface {
\tArea() int
}

func Add(a int, b int) int {
\treturn a + b
}

func (p Point) Norm() int {
\treturn p.X + p.Y
}
"""


@pytest.mark.skipif(not _grammar_installed("tree_sitter_go"),
                    reason="tree-sitter-go grammar not installed")
def test_go_symbols(indexer):
    assert ".go" in indexer.get_supported_extensions()
    by_type = _symbols_by_type(indexer.parse_file("main.go", GO_SOURCE))
    # a function + a struct (type) at minimum
    assert "Add" in by_type.get("function", set())
    assert "Point" in by_type.get("struct", set())
    assert "Norm" in by_type.get("method", set())
    assert "Shape" in by_type.get("interface", set())


# --- Rust --------------------------------------------------------------------

RUST_SOURCE = """
struct Point {
    x: i32,
    y: i32,
}

enum Color {
    Red,
    Green,
}

trait Shape {
    fn area(&self) -> i32;
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

impl Point {
    fn norm(&self) -> i32 {
        self.x + self.y
    }
}
"""


@pytest.mark.skipif(not _grammar_installed("tree_sitter_rust"),
                    reason="tree-sitter-rust grammar not installed")
def test_rust_symbols(indexer):
    assert ".rs" in indexer.get_supported_extensions()
    by_type = _symbols_by_type(indexer.parse_file("lib.rs", RUST_SOURCE))
    # a function + a struct (type) at minimum
    assert "add" in by_type.get("function", set())
    assert "Point" in by_type.get("struct", set())
    assert "Color" in by_type.get("enum", set())
    assert "norm" in by_type.get("method", set())


# --- Java --------------------------------------------------------------------

JAVA_SOURCE = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}

interface Shape {
    int area();
}
"""


@pytest.mark.skipif(not _grammar_installed("tree_sitter_java"),
                    reason="tree-sitter-java grammar not installed")
def test_java_symbols(indexer):
    assert ".java" in indexer.get_supported_extensions()
    by_type = _symbols_by_type(indexer.parse_file("Calculator.java", JAVA_SOURCE))
    # a method + a class (type) at minimum
    assert "Calculator" in by_type.get("class", set())
    assert "add" in by_type.get("method", set())
    assert "Shape" in by_type.get("interface", set())


def test_unknown_extension_returns_empty(indexer):
    assert indexer.parse_file("notes.txt", "hello world") == []
