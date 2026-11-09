"""Windows-importability of the worker and the end-of-run timeout report."""

import subprocess
import sys
import textwrap

from arena.cli import _print_timeout_report, _timeout_report


def test_worker_imports_without_posix_modules():
    code = textwrap.dedent("""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name in ("fcntl", "resource"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import arena.agents.worker, arena.agents.sandbox, arena.cli, arena.api.app, arena.advisor.session
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _summary(timeouts: int, decisions: int = 1000, base=(0, 0)):
    return {
        "entrants": [{"entrant": 0, "name": "bot", "decisions": decisions,
                      "fallback_total": timeouts, "fallbacks": {"timeout": timeouts} if timeouts else {}}],
        "baselines_combined": {"decisions": base[1], "timeout_fallbacks": base[0]},
    }


def test_timeout_report_warns_above_one_percent(capsys):
    _print_timeout_report(_summary(11), [0], file=sys.stdout)
    out = capsys.readouterr().out
    assert "1.10%" in out and "WARNING" in out and "--jobs" in out


def test_timeout_report_quiet_at_or_below_one_percent(capsys):
    lines, warn = _timeout_report(_summary(10, base=(5, 1000)), [0])
    assert not warn and len(lines) == 2
    _print_timeout_report(_summary(10), [0], file=sys.stdout)
    assert "WARNING" not in capsys.readouterr().out


def test_baseline_timeouts_trigger_warning():
    _, warn = _timeout_report(_summary(0, base=(50, 1000)), [0])
    assert warn
