"""OpenSpiel stays a dev-only oracle; the arena runtime does not depend on `strategy/`."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = Path(__file__).resolve().parent / "oracle"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _python_files(*dirs: Path):
    for d in dirs:
        yield from sorted(d.rglob("*.py"))


def test_openspiel_is_only_imported_by_the_oracle_tests():
    offenders = []
    for path in _python_files(ROOT / "strategy", ROOT / "arena", ROOT / "tests"):
        if ORACLE_DIR in path.parents or path.name == "test_openspiel_oracle.py":
            continue
        if _imports(path) & {"pyspiel", "open_spiel"}:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_arena_does_not_import_strategy():
    offenders = [str(p.relative_to(ROOT)) for p in _python_files(ROOT / "arena") if "strategy" in _imports(p)]
    assert offenders == []


def test_numba_is_confined_to_the_kernels():
    users = {str(p.relative_to(ROOT)) for p in _python_files(ROOT / "strategy") if "numba" in _imports(p)}
    assert users == {"strategy/solve/kernels.py", "strategy/endgame/kernels.py", "strategy/blueprint/kernels.py"}


def test_oracle_and_training_packages_are_not_runtime_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = " ".join(project["project"]["dependencies"]).lower()
    assert "open-spiel" not in runtime and "open_spiel" not in runtime
    assert "numba" not in runtime
    groups = project["dependency-groups"]
    assert any(dep.startswith("open-spiel") for dep in groups["oracle"])
    assert "oracle" not in project["tool"]["uv"]["default-groups"]
    assert "train" not in project["tool"]["uv"]["default-groups"]
