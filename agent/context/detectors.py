"""Phase 9 — framework/tech detection, entry-point discovery, convention inference.

Pure functions over a :class:`ScanResult` (the manifests are already read by the
scanner), so this module does no file IO of its own and is trivially testable.
Covers the five ecosystems called out in the plan: python, node, go, rust, java.

100% local: manifest parsing only, never a network call.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Set

from agent.config import logger
from agent.context.schemas import Conventions, EntryPoint, TechStack

try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover - 3.10 fallback path
    tomllib = None

# dependency-name substring -> human framework label.
_FRAMEWORK_HINTS = {
    # python
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
    "starlette": "Starlette", "pydantic": "Pydantic", "pytest": "pytest",
    "sqlalchemy": "SQLAlchemy", "aiohttp": "aiohttp", "tornado": "Tornado",
    "celery": "Celery", "numpy": "NumPy", "pandas": "pandas", "torch": "PyTorch",
    # node
    "react": "React", "next": "Next.js", "vue": "Vue", "@angular/core": "Angular",
    "express": "Express", "@nestjs/core": "NestJS", "svelte": "Svelte",
    "jest": "Jest", "vitest": "Vitest", "webpack": "Webpack", "vite": "Vite",
    # go
    "github.com/gin-gonic/gin": "Gin", "github.com/labstack/echo": "Echo",
    "github.com/gofiber/fiber": "Fiber",
    # rust
    "actix-web": "Actix", "rocket": "Rocket", "axum": "Axum", "tokio": "Tokio",
    "serde": "Serde", "clap": "clap",
    # java
    "spring-boot": "Spring Boot", "springframework": "Spring", "junit": "JUnit",
}

_CODE_EXTS = (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
              ".rb", ".php", ".c", ".cpp", ".cs")

_LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust",
    ".java": "Java", ".rb": "Ruby", ".php": "PHP", ".cs": "C#",
}


# --- manifest parsing helpers ------------------------------------------------

def _parse_toml(text: str) -> dict:
    if tomllib is not None:
        try:
            return tomllib.loads(text)
        except Exception:
            return {}
    return {}


def _deps_from_pyproject(text: str) -> List[str]:
    data = _parse_toml(text)
    deps: List[str] = []
    if data:
        proj = data.get("project", {})
        for d in proj.get("dependencies", []) or []:
            deps.append(_dep_name(d))
        for group in (proj.get("optional-dependencies", {}) or {}).values():
            for d in group:
                deps.append(_dep_name(d))
        # poetry
        poetry = data.get("tool", {}).get("poetry", {})
        deps.extend(list((poetry.get("dependencies", {}) or {}).keys()))
    if not deps:  # 3.10 fallback / unparsable — best-effort regex.
        for m in re.finditer(r'["\']([A-Za-z0-9_\-\.]+)\s*(?:[<>=~!\[].*)?["\']', text):
            deps.append(_dep_name(m.group(1)))
    return [d for d in deps if d and d.lower() != "python"]


def _dep_name(spec: str) -> str:
    return re.split(r"[<>=!~\[\s;]", spec.strip(), 1)[0].lower()


def _deps_from_requirements(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        out.append(_dep_name(line))
    return out


def _deps_from_package_json(text: str) -> List[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    out = []
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        out.extend(list((data.get(key, {}) or {}).keys()))
    return out


def _deps_from_go_mod(text: str) -> List[str]:
    out = []
    for m in re.finditer(r"^\s*([\w\.\-/]+)\s+v[\d]", text, re.M):
        out.append(m.group(1))
    return out


def _deps_from_cargo(text: str) -> List[str]:
    data = _parse_toml(text)
    out: List[str] = []
    if data:
        for section in ("dependencies", "dev-dependencies"):
            out.extend(list((data.get(section, {}) or {}).keys()))
    if not out:
        in_deps = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("["):
                in_deps = "dependencies" in s
                continue
            if in_deps and "=" in s:
                out.append(s.split("=", 1)[0].strip())
    return out


def _deps_from_pom(text: str) -> List[str]:
    return [m.group(1) for m in re.finditer(r"<artifactId>([^<]+)</artifactId>", text)]


# --- public detection API ----------------------------------------------------

def detect_tech_stack(manifests: Dict[str, str]) -> List[TechStack]:
    """Detect ecosystems + their declared dependencies from manifests."""
    stacks: List[TechStack] = []
    for rel, text in manifests.items():
        base = rel.rsplit("/", 1)[-1]
        try:
            if base == "pyproject.toml":
                stacks.append(TechStack(ecosystem="python", manifest=rel,
                                        dependencies=_deps_from_pyproject(text)))
            elif base == "requirements.txt":
                stacks.append(TechStack(ecosystem="python", manifest=rel,
                                        dependencies=_deps_from_requirements(text)))
            elif base == "package.json":
                stacks.append(TechStack(ecosystem="node", manifest=rel,
                                        dependencies=_deps_from_package_json(text)))
            elif base == "go.mod":
                stacks.append(TechStack(ecosystem="go", manifest=rel,
                                        dependencies=_deps_from_go_mod(text)))
            elif base == "Cargo.toml":
                stacks.append(TechStack(ecosystem="rust", manifest=rel,
                                        dependencies=_deps_from_cargo(text)))
            elif base == "pom.xml":
                stacks.append(TechStack(ecosystem="java", manifest=rel,
                                        dependencies=_deps_from_pom(text)))
        except Exception as e:
            logger.debug(f"Context engine: manifest parse failed for {rel}: {e}")
    return stacks


def detect_frameworks(tech_stack: List[TechStack]) -> List[str]:
    """Map declared dependencies onto human framework labels."""
    found: Set[str] = set()
    for stack in tech_stack:
        for dep in stack.dependencies:
            low = dep.lower()
            for hint, label in _FRAMEWORK_HINTS.items():
                if hint in low:
                    found.add(label)
    return sorted(found)


def detect_entry_points(files: List[str], manifests: Dict[str, str]) -> List[EntryPoint]:
    """Discover entry points: well-known files + declared scripts."""
    eps: List[EntryPoint] = []
    seen: Set[str] = set()

    def add(kind: str, target: str, evidence: str):
        key = f"{kind}:{target}"
        if key not in seen:
            seen.add(key)
            eps.append(EntryPoint(kind=kind, target=target, evidence=evidence))

    file_basenames = {f.rsplit("/", 1)[-1]: f for f in files}

    # Well-known entry files (basename match keeps it language-agnostic).
    KNOWN = {
        "main.py": "main", "__main__.py": "main", "app.py": "server",
        "cli.py": "cli", "manage.py": "cli", "wsgi.py": "server", "asgi.py": "server",
        "index.js": "index", "index.ts": "index", "server.js": "server",
        "main.js": "main", "main.ts": "main",
        "main.go": "main", "main.rs": "main", "Main.java": "main",
    }
    for base, kind in KNOWN.items():
        if base in file_basenames:
            add(kind, file_basenames[base], f"well-known {kind} file")

    # package.json "scripts" / "bin".
    for rel, text in manifests.items():
        base = rel.rsplit("/", 1)[-1]
        if base == "package.json":
            try:
                data = json.loads(text)
            except Exception:
                continue
            for name, cmd in (data.get("scripts", {}) or {}).items():
                add("script", f"npm run {name}", f"package.json scripts.{name}")
            binv = data.get("bin")
            if isinstance(binv, str):
                add("cli", binv, "package.json bin")
            elif isinstance(binv, dict):
                for name, path in binv.items():
                    add("cli", path, f"package.json bin.{name}")
        elif base == "pyproject.toml":
            data = _parse_toml(text)
            for name, target in (data.get("project", {}).get("scripts", {}) or {}).items():
                add("cli", f"{name} -> {target}", "pyproject [project.scripts]")

    return eps


def infer_conventions(scan_files: List[str], file_types: Dict[str, int],
                      manifests: Dict[str, str]) -> Conventions:
    """Infer lightweight conventions (MVP). Richer inference is an enhancement."""
    conv = Conventions()

    # Primary language = most common code extension.
    code_counts = {ext: n for ext, n in file_types.items() if ext in _CODE_EXTS}
    if code_counts:
        top_ext = max(code_counts.items(), key=lambda kv: kv[1])[0]
        conv.primary_language = _LANG_BY_EXT.get(top_ext, top_ext)

    # Naming style — sample file stems.
    stems = [f.rsplit("/", 1)[-1].split(".")[0] for f in scan_files if "." in f]
    snake = sum(1 for s in stems if "_" in s and s.islower())
    camel = sum(1 for s in stems if re.search(r"[a-z][A-Z]", s))
    if snake or camel:
        conv.naming_style = "snake_case" if snake >= camel else "camelCase"

    # Test layout.
    if any(f.startswith("tests/") or f.startswith("test/") for f in scan_files):
        conv.test_layout = "tests/ directory"
    elif any(f.endswith("_test.go") for f in scan_files):
        conv.test_layout = "*_test.go (co-located)"
    elif any(re.search(r"(^|/)test_.*\.py$", f) for f in scan_files):
        conv.test_layout = "test_*.py"
    elif any(re.search(r"\.(test|spec)\.[tj]sx?$", f) for f in scan_files):
        conv.test_layout = "*.test/*.spec (co-located)"

    # Lint/format tooling.
    lint_map = {
        "ruff": "ruff", "flake8": "flake8", "pylintrc": "pylint",
        "eslintrc": "eslint", "prettierrc": "prettier", "rustfmt": "rustfmt",
        "golangci": "golangci-lint", "pre-commit": "pre-commit",
    }
    tools: Set[str] = set()
    for rel in list(manifests.keys()) + scan_files:
        base = rel.rsplit("/", 1)[-1].lower()
        for needle, tool in lint_map.items():
            if needle in base:
                tools.add(tool)
    # ruff/pytest declared in pyproject also count.
    for rel, text in manifests.items():
        if rel.rsplit("/", 1)[-1] == "pyproject.toml":
            if "tool.ruff" in text or "[tool.ruff]" in text:
                tools.add("ruff")
    conv.lint_tools = sorted(tools)
    conv.has_lint_config = bool(tools)

    # Top-level source layout (directories).
    top_dirs: Set[str] = set()
    for f in scan_files:
        if "/" in f:
            top_dirs.add(f.split("/", 1)[0])
    conv.source_layout = sorted(top_dirs)

    return conv
