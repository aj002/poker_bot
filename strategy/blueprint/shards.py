"""Sharded blueprint training across machines that share only a directory (offline).

Each shard is an ordinary :mod:`strategy.blueprint.train` run forked from a shared base
checkpoint with its own seed, so it samples its own deal and start-state stream. After the
linear-discount intervals have ended (``progress.intervals >= config.linear_intervals``)
every update is a plain sum, so shards merge exactly as ``base + sum(shard - base)`` for
regrets and strategy sums alike (regrets re-floored at ``regret_floor``).

Shared directory layout (``--dir``)::

    base/epoch-NNNN/        merged checkpoint of epoch N, states pool, config, signals
    base/LATEST             current epoch number
    epochs/NNNN/SYNC        written by the merger: shards stop, push and wait for epoch N+1
    epochs/NNNN/shard-SS/   a shard's latest pushed checkpoint in epoch N
    workers/shard-SS.json   heartbeat: state, epoch, pairs, CPU ledger, restarts
    merger.json             merger heartbeat, CPU total, projection
    STOP                    budget reached: shards push and exit; the merger writes final/
    final/                  the final merged checkpoint

Arrays are stored zlib-compressed in parts below the store's per-file size limit with a
manifest written last; readers verify sha256. Every machine meters its own busy CPU from
``/proc/stat`` (the whole VM, so evaluation and monitoring count too) into its heartbeat;
the merger sums the heartbeats with ``--base-cpu-hours`` and writes ``STOP`` at
``--stop-at``.

    python -m strategy.blueprint.shards publish --run RUN --dir DIR          # epoch 0 from a run
    python -m strategy.blueprint.shards work --shard 1 --dir DIR             # on every machine
    python -m strategy.blueprint.shards merger --dir DIR --stop-at 292 ...   # on one machine
    python -m strategy.blueprint.shards status --dir DIR --expect 3          # health line
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PART_BYTES = 48 << 20
ARRAYS = ("regret", "strat")
POLL_S = 30
STALE_S = 15 * 60
SYNC_TIMEOUT_S = 25 * 60


def shard_seed(epoch: int, shard: int) -> int:
    return 100_000 + 1_000 * epoch + shard


def _now() -> float:
    return time.time()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------------- blobs


def put_arrays(dest: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Write arrays as compressed parts plus ``manifest.json`` (last); drop older parts."""
    dest.mkdir(parents=True, exist_ok=True)
    push = f"{int(_now() * 1000):x}"
    entries = {}
    for name, a in arrays.items():
        raw = np.ascontiguousarray(a).tobytes()
        comp = zlib.compress(raw, 1)
        parts = []
        for k in range(0, max(1, len(comp)), PART_BYTES):
            p = f"{name}.{push}.{k // PART_BYTES:03d}"
            (dest / p).write_bytes(comp[k:k + PART_BYTES])
            parts.append(p)
        entries[name] = {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(raw).hexdigest(),
                         "parts": parts}
    _write_json(dest / "manifest.json", {"push": push, "arrays": entries, "meta": meta, "written": _now()})
    keep = {p for e in entries.values() for p in e["parts"]}
    for f in dest.iterdir():
        if f.name.split(".")[0] in arrays and f.name not in keep:
            try:
                f.unlink()
            except FileNotFoundError:
                pass


def get_arrays(src: Path, retries: int = 3) -> tuple[dict[str, np.ndarray], dict]:
    for attempt in range(retries):
        man = _read_json(src / "manifest.json")
        if man is None:
            raise FileNotFoundError(f"{src}: no manifest")
        try:
            out = {}
            for name, e in man["arrays"].items():
                raw = zlib.decompress(b"".join((src / p).read_bytes() for p in e["parts"]))
                if hashlib.sha256(raw).hexdigest() != e["sha256"]:
                    raise ValueError(f"{src}/{name}: sha256 mismatch")
                out[name] = np.frombuffer(raw, dtype=e["dtype"]).reshape(e["shape"]).copy()
            return out, man["meta"]
        except (FileNotFoundError, ValueError, zlib.error):
            if attempt == retries - 1:
                raise
            time.sleep(5)
    raise AssertionError


# ----------------------------------------------------------------------------- merge


def merge(base: dict[str, np.ndarray], shards: list[dict[str, np.ndarray]], regret_floor: float) -> dict[str, np.ndarray]:
    """``base + sum(shard - base)``; regrets re-floored."""
    out = {}
    for name in ARRAYS:
        acc = base[name].astype(np.float64)
        b = base[name].astype(np.float64)
        for s in shards:
            acc += s[name].astype(np.float64) - b
        if name == "regret":
            np.maximum(acc, regret_floor, out=acc)
        out[name] = acc.astype(np.float32)
    return out


def merge_progress(base: dict, shards: list[dict], interval_pairs: int) -> dict:
    out = dict(base)
    for k in ("chunks", "pairs", "traversals", "nodes", "wall_seconds", "value_sum"):
        out[k] = base[k] + sum(s[k] - base[k] for s in shards)
    out["intervals"] = max(base["intervals"], out["pairs"] // interval_pairs)
    return out


def signals(prev: dict[str, np.ndarray] | None, cur: dict[str, np.ndarray], pairs: int) -> dict:
    """Cheap convergence signals from the arrays alone."""
    from strategy.blueprint import abstraction as A

    r = cur["regret"].reshape(-1, A.NUM_ACTIONS)
    s = cur["strat"].reshape(-1, A.NUM_ACTIONS).astype(np.float64)
    mass = s.sum(axis=1)
    reached = mass > 0
    pos = np.maximum(r[reached], 0).max(axis=1).astype(np.float64)
    out = {"pairs": int(pairs), "reached_keys": int(reached.sum()),
           "mean_max_positive_regret": float(pos.mean()) if pos.size else 0.0,
           "sum_max_positive_regret_per_pair": float(pos.sum() / max(1, pairs))}
    if prev is not None:
        ps = prev["strat"].reshape(-1, A.NUM_ACTIONS).astype(np.float64)
        pm = ps.sum(axis=1)
        both = (pm > 0) & reached
        pp = ps[both] / pm[both, None]
        cp = s[both] / mass[both, None]
        l1 = np.abs(pp - cp).sum(axis=1)
        w = mass[both] / mass[both].sum()
        out["strategy_l1_mass_weighted"] = float((l1 * w).sum())
        out["strategy_l1_mean"] = float(l1.mean())
    return out


def pack_size(strat: np.ndarray, work: Path) -> dict:
    from strategy.blueprint.policy import build_levels, write_levels

    path = work / "probe-pack.npz"
    arrays = build_levels(strat)
    prov = write_levels(path, arrays, {"probe": True})
    nbytes = sum(a.nbytes for a in arrays.values())
    size = path.stat().st_size
    path.unlink()
    return {"rows": prov["rows"], "file_bytes": size, "loaded_bytes": int(nbytes)}


# ----------------------------------------------------------------------------- CPU ledger


class Ledger:
    """Busy CPU seconds of this machine since the ledger began, kept across restarts."""

    def __init__(self, path: Path, start_cpu_s: float = 0.0) -> None:
        self.path = path
        st = _read_json(path) or {"cpu_s": start_cpu_s, "boot_id": None, "last": None}
        self.cpu_s = float(st["cpu_s"])
        self.boot_id = st["boot_id"]
        self.last = st["last"]

    @staticmethod
    def _busy() -> float:
        f = Path("/proc/stat").read_text().split("\n")[0].split()
        user, nice, system, idle, iowait, irq, softirq, steal = (int(x) for x in f[1:9])
        return (user + nice + system + irq + softirq + steal) / os.sysconf("SC_CLK_TCK")

    def tick(self) -> float:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        busy = self._busy()
        if self.boot_id == boot and self.last is not None and busy >= self.last:
            self.cpu_s += busy - self.last
        self.boot_id, self.last = boot, busy
        _write_json(self.path, {"cpu_s": self.cpu_s, "boot_id": self.boot_id, "last": self.last})
        return self.cpu_s


# ----------------------------------------------------------------------------- epochs


def latest_epoch(d: Path) -> int | None:
    try:
        return int((d / "base" / "LATEST").read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def base_dir(d: Path, epoch: int) -> Path:
    return d / "base" / f"epoch-{epoch:04d}"


def epoch_dir(d: Path, epoch: int) -> Path:
    return d / "epochs" / f"{epoch:04d}"


def publish(d: Path, epoch: int, arrays: dict[str, np.ndarray], config: dict, progress: dict, states: Path,
            extra: dict | None = None) -> None:
    from strategy.blueprint.train import TrainConfig

    if progress["intervals"] < config["linear_intervals"]:
        raise SystemExit("publish after the linear-discount intervals end: merging is exact only then")
    b = base_dir(d, epoch)
    b.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(states, b / "states.npz")
    put_arrays(b, arrays, {"epoch": epoch, "config": config, "config_digest": TrainConfig(**config).digest(),
                           "progress": progress, **(extra or {})})
    _write_json(b / "published.json", {"epoch": epoch, "time": _now(), "progress": progress, **(extra or {})})
    (d / "base" / ".LATEST.tmp").write_text(str(epoch))
    (d / "base" / ".LATEST.tmp").replace(d / "base" / "LATEST")


def cmd_publish(a) -> None:
    run = a.run
    cfg = json.loads((run / "config.json").read_text())
    meta = json.loads((run / "ckpt" / "meta.json").read_text())
    arrays = {n: np.load(run / "ckpt" / f"{n}.npy") for n in ARRAYS}
    if hashlib.sha256(arrays["regret"].tobytes()).hexdigest() != meta["regret_sha256"]:
        raise SystemExit("checkpoint changed while reading; retry")
    states = REPO / cfg["states"] if not Path(cfg["states"]).is_absolute() else Path(cfg["states"])
    sig = signals(None, arrays, meta["progress"]["pairs"])
    sig["pack"] = pack_size(arrays["strat"], run)
    publish(a.dir, a.epoch, arrays, cfg, meta["progress"], states, {"source_run": str(run), "signals": sig})
    print(f"published epoch {a.epoch}: {meta['progress']['pairs']:,} pairs; {json.dumps(sig)}")


# ----------------------------------------------------------------------------- worker


class Worker:
    def __init__(self, a) -> None:
        self.a = a
        self.d: Path = a.dir
        self.shard: int = a.shard
        self.name = f"shard-{self.shard:02d}"
        self.local = REPO / "artifacts" / "blueprint" / "shards" / self.name
        self.local.mkdir(parents=True, exist_ok=True)
        hb = _read_json(self.d / "workers" / f"{self.name}.json") or {}
        self.ledger = Ledger(self.local / "ledger.json", start_cpu_s=float(hb.get("cpu_s", 0.0)))
        self.restarts = int(hb.get("restarts", 0))
        self.proc: subprocess.Popen | None = None
        self.epoch: int | None = None
        self.pushed_mtime = 0.0
        self.last_push = 0.0
        self.state = "starting"
        self.since = _now()
        self.log = (self.local / "worker.log").open("a")

    def say(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {self.name} {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    @property
    def run(self) -> Path:
        return self.local / f"epoch-{self.epoch:04d}"

    def progress(self) -> dict | None:
        m = _read_json(self.run / "ckpt" / "meta.json") if self.epoch is not None else None
        return m["progress"] if m else None

    def heartbeat(self) -> None:
        cpu = self.ledger.tick()
        p = self.progress() or {}
        _write_json(self.d / "workers" / f"{self.name}.json", {
            "shard": self.shard, "host": socket.gethostname(), "state": self.state, "epoch": self.epoch,
            "pairs": p.get("pairs"), "cpu_s": cpu, "restarts": self.restarts, "time": _now(),
            "last_push": self.last_push, "since": self.since, "threads": self.a.threads,
            "producers": self.a.producers})

    def fork(self, epoch: int) -> None:
        """Local run for this shard in ``epoch``: the base checkpoint under this shard's seed."""
        from strategy.blueprint.train import TrainConfig

        self.epoch = epoch
        run = self.run
        if (run / "ckpt" / "meta.json").exists():
            return
        b = base_dir(self.d, epoch)
        arrays, meta = get_arrays(b)
        cfg = dict(meta["config"], seed=shard_seed(epoch, self.shard))
        states = Path(cfg["states"])
        states = states if states.is_absolute() else REPO / states
        if not states.exists():
            states.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(b / "states.npz", states)
        tc = TrainConfig(**cfg)
        tmp = self.local / f".fork-{epoch:04d}"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "ckpt").mkdir(parents=True)
        for n in ARRAYS:
            np.save(tmp / "ckpt" / f"{n}.npy", arrays[n])
        (tmp / "ckpt" / "meta.json").write_text(json.dumps({
            "config_digest": tc.digest(), "progress": meta["progress"],
            "regret_sha256": hashlib.sha256(arrays["regret"].tobytes()).hexdigest(),
            "strat_sha256": hashlib.sha256(arrays["strat"].tobytes()).hexdigest()}, indent=1))
        (tmp / "config.json").write_text(json.dumps(asdict(tc), indent=1, sort_keys=True))
        (tmp / "fork.json").write_text(json.dumps({"epoch": epoch, "shard": self.shard, "base_progress": meta["progress"],
                                                   "base_config_digest": meta["config_digest"]}, indent=1))
        tmp.rename(run)
        self.say(f"forked epoch {epoch} at {meta['progress']['pairs']:,} pairs, seed {cfg['seed']}")

    def _orphans(self) -> list[int]:
        """Trainers on this shard's run left behind by an earlier worker process."""
        out = []
        for p in Path("/proc").iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"strategy.blueprint.train" in cmd and str(self.local).encode() in b" ".join(cmd):
                out.append(int(p.name))
        return out

    def start(self) -> None:
        stale = self._orphans()
        if stale:
            self.say(f"stopping orphaned trainers {stale}")
            for r in self.local.glob("epoch-*"):
                (r / "STOP").touch()
            for _ in range(120):
                if not self._orphans():
                    break
                time.sleep(5)
            for pid in self._orphans():
                os.kill(pid, signal.SIGKILL)
        (self.run / "STOP").unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "strategy.blueprint.train", "--run", str(self.run), "--threads",
               str(self.a.threads), "--producers", str(self.a.producers), "--hours", "10000",
               "--checkpoint-minutes", str(self.a.push_minutes)]
        out = (self.run / "train.log").open("a")
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, ARENA_REQUIRE_NUMBA="1", PYTHONUNBUFFERED="1"))
        self.state = "training"
        self.since = _now()
        self.say(f"training epoch {self.epoch} (pid {self.proc.pid})")

    def stop_training(self) -> None:
        if self.proc is None:
            return
        (self.run / "STOP").touch()
        try:
            self.proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self.say("trainer ignored STOP for 10 min; interrupting")
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=300)
        self.proc = None

    def push(self, force: bool = False) -> None:
        ck = self.run / "ckpt"
        try:
            mtime = (ck / "meta.json").stat().st_mtime
        except FileNotFoundError:
            return
        if not force and mtime == self.pushed_mtime:
            return
        meta = json.loads((ck / "meta.json").read_text())
        try:
            arrays = {n: np.load(ck / f"{n}.npy") for n in ARRAYS}
        except (FileNotFoundError, ValueError):
            return
        if (hashlib.sha256(arrays["strat"].tobytes()).hexdigest() != meta["strat_sha256"]
                or hashlib.sha256(arrays["regret"].tobytes()).hexdigest() != meta["regret_sha256"]):
            return
        fork = json.loads((self.run / "fork.json").read_text())
        if meta["progress"]["pairs"] == fork["base_progress"]["pairs"]:
            return
        put_arrays(epoch_dir(self.d, self.epoch) / self.name, arrays,
                   {"shard": self.shard, "epoch": self.epoch, "progress": meta["progress"],
                    "seed": shard_seed(self.epoch, self.shard)})
        self.pushed_mtime = mtime
        self.last_push = _now()
        self.say(f"pushed {meta['progress']['pairs']:,} pairs")

    def loop(self) -> None:
        while latest_epoch(self.d) is None:
            self.state = "waiting-base"
            self.heartbeat()
            time.sleep(POLL_S)
        while True:
            if (self.d / "STOP").exists():
                self.state = "stopped"
                self.heartbeat()
                self.say("STOP present; exiting")
                return
            self.fork(latest_epoch(self.d))
            self.start()
            while True:
                time.sleep(POLL_S)
                self.heartbeat()
                self.push()
                sync = (epoch_dir(self.d, self.epoch) / "SYNC").exists()
                stop = (self.d / "STOP").exists()
                if sync or stop:
                    self.say("STOP" if stop else f"SYNC for epoch {self.epoch}")
                    self.stop_training()
                    self.push(force=True)
                    break
                if self.proc is not None and self.proc.poll() is not None:
                    self.restarts += 1
                    self.say(f"trainer exited with {self.proc.returncode}; restart {self.restarts} from its checkpoint")
                    self.start()
            if (self.d / "STOP").exists():
                self.state = "stopped"
                self.heartbeat()
                self.say("final push done; exiting")
                return
            self.state = "synced"
            done = self.epoch
            while latest_epoch(self.d) == done and not (self.d / "STOP").exists():
                self.heartbeat()
                time.sleep(POLL_S)
            shutil.rmtree(self.local / f"epoch-{done:04d}", ignore_errors=True)


def cmd_work(a) -> None:
    Worker(a).loop()


# ----------------------------------------------------------------------------- merger


def heartbeats(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted((d / "workers").glob("shard-*.json")) if (d / "workers").exists() else []:
        hb = _read_json(f)
        if hb:
            out[f.stem] = hb
    return out


def cpu_total(d: Path, base_cpu_hours: float) -> float:
    return base_cpu_hours + sum(hb.get("cpu_s", 0.0) for hb in heartbeats(d).values()) / 3600


def merge_epoch(d: Path, epoch: int, out_epoch: int | None, work: Path, log) -> dict:
    """Merge every pushed shard of ``epoch`` into ``out_epoch`` (or ``final/`` when None)."""
    base, bmeta = get_arrays(base_dir(d, epoch))
    cfg = bmeta["config"]
    pushed = sorted(p for p in epoch_dir(d, epoch).glob("shard-*") if (p / "manifest.json").exists())
    shard_arrays, shard_progress = [], []
    for p in pushed:
        arr, meta = get_arrays(p)
        shard_arrays.append(arr)
        shard_progress.append(meta["progress"])
    merged = merge(base, shard_arrays, cfg["regret_floor"]) if shard_arrays else base
    progress = merge_progress(bmeta["progress"], shard_progress, cfg["interval_pairs"])
    sig = signals(base, merged, progress["pairs"])
    sig["pack"] = pack_size(merged["strat"], work)
    info = {"from_epoch": epoch, "shards": [p.name for p in pushed],
            "shard_pairs": [s["pairs"] - bmeta["progress"]["pairs"] for s in shard_progress], "signals": sig}
    states = base_dir(d, epoch) / "states.npz"
    if out_epoch is None:
        f = d / "final"
        f.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(states, f / "states.npz")
        put_arrays(f, merged, {"config": cfg, "progress": progress, **info})
        _write_json(f / "published.json", {"time": _now(), "progress": progress, **info})
    else:
        publish(d, out_epoch, merged, cfg, progress, states, info)
    log(f"merged epoch {epoch} ({len(pushed)} shards, +{progress['pairs'] - bmeta['progress']['pairs']:,} pairs) "
        f"-> {'final' if out_epoch is None else f'epoch {out_epoch}'}: {json.dumps(sig)}")
    return info


def projection(hist: list[tuple[float, float]], total: float, stop_at: float) -> dict:
    """CPU-hours per wall hour over the last two hours and the projected stop time."""
    recent = [(t, c) for t, c in hist if t >= hist[-1][0] - 7200]
    if len(recent) < 2 or recent[-1][0] - recent[0][0] < 300:
        return {"rate_cpu_h_per_h": None, "eta_utc": None}
    rate = (recent[-1][1] - recent[0][1]) / ((recent[-1][0] - recent[0][0]) / 3600)
    eta = hist[-1][0] + max(0.0, stop_at - total) / rate * 3600 if rate > 0 else None
    return {"rate_cpu_h_per_h": round(rate, 2),
            "eta_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(eta)) if eta else None}


def cmd_merger(a) -> None:
    d: Path = a.dir
    work = REPO / "artifacts" / "blueprint" / "shards" / "merger"
    work.mkdir(parents=True, exist_ok=True)
    logf = (work / "merger.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} merger {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    hist: list[tuple[float, float]] = (_read_json(work / "history.json") or [])
    state = "running"
    while True:
        e = latest_epoch(d)
        total = cpu_total(d, a.base_cpu_hours)
        hist.append((_now(), total))
        hist = hist[-2000:]
        _write_json(work / "history.json", hist)
        proj = projection(hist, total, a.stop_at)
        pub = _read_json(base_dir(d, e) / "published.json") if e is not None else None
        _write_json(d / "merger.json", {"time": _now(), "state": state, "epoch": e, "cpu_hours_total": round(total, 2),
                                        "stop_at": a.stop_at, "base_cpu_hours": a.base_cpu_hours,
                                        "every_hours": a.every_hours, **proj,
                                        "epoch_published": pub["time"] if pub else None,
                                        "signals": pub.get("signals") if pub else None})
        if state == "done":
            return
        if total >= a.stop_at and not (d / "STOP").exists():
            (d / "STOP").write_text(json.dumps({"time": _now(), "cpu_hours_total": total}))
            log(f"budget reached ({total:.1f} CPU-hours): STOP")
        if (d / "STOP").exists() and e is not None:
            state = "stopping"
            if _wait_for(d, e, ("stopped",), log):
                merge_epoch(d, e, None, work, log)
                state = "done"
                log("final merge written; exiting")
            continue
        if e is not None and pub and _now() - pub["time"] >= a.every_hours * 3600:
            sync = epoch_dir(d, e) / "SYNC"
            sync.parent.mkdir(parents=True, exist_ok=True)
            sync.write_text(str(_now()))
            log(f"SYNC epoch {e}")
            state = "syncing"
            _wait_for(d, e, ("synced", "stopped"), log)
            merge_epoch(d, e, e + 1, work, log)
            state = "running"
            continue
        time.sleep(60)


def _wait_for(d: Path, epoch: int, states: tuple[str, ...], log) -> bool:
    """Wait until every live shard of ``epoch`` reports one of ``states`` (dead shards are skipped)."""
    t0 = _now()
    while True:
        live = {k: hb for k, hb in heartbeats(d).items() if _now() - hb["time"] < STALE_S and hb.get("epoch") == epoch}
        pending = [k for k, hb in live.items() if hb["state"] not in states]
        if not pending:
            return True
        if _now() - t0 > SYNC_TIMEOUT_S:
            log(f"sync timeout; merging without waiting for {pending}")
            return True
        time.sleep(POLL_S)


# ----------------------------------------------------------------------------- status


def status(d: Path, expect: int) -> tuple[str, list[str], list[str]]:
    now = _now()
    m = _read_json(d / "merger.json")
    hbs = heartbeats(d)
    problems, lines = [], []
    if m is None:
        problems.append("no merger heartbeat")
    else:
        if now - m["time"] > 10 * 60 and m["state"] != "done":
            problems.append(f"merger heartbeat {int((now - m['time']) / 60)} min old")
        pack = ((m.get("signals") or {}).get("pack") or {})
        if pack.get("loaded_bytes", 0) > 60e6:
            problems.append(f"pack loads {pack['loaded_bytes'] / 1e6:.0f} MB (limit context: 100 MB per bot process)")
        lines.append(f"merger {m['state']} epoch {m['epoch']} cpu {m['cpu_hours_total']:.1f}/{m['stop_at']} CPU-h "
                     f"rate {m.get('rate_cpu_h_per_h')} CPU-h/h eta {m.get('eta_utc')} UTC "
                     f"pack {pack.get('file_bytes', 0) / 1e6:.1f} MB file / {pack.get('loaded_bytes', 0) / 1e6:.1f} MB loaded")
    live = 0
    for k, hb in hbs.items():
        age = (now - hb["time"]) / 60
        push_age = (now - hb["last_push"]) / 60 if hb.get("last_push") else None
        lines.append(f"{k} {hb['state']} epoch {hb['epoch']} pairs {hb.get('pairs') or 0:,} cpu {hb['cpu_s'] / 3600:.1f} h "
                     f"heartbeat {age:.0f} min push {'-' if push_age is None else f'{push_age:.0f} min'} "
                     f"restarts {hb['restarts']} host {hb['host']}")
        if hb["state"] in ("stopped",):
            continue
        if age > 10:
            problems.append(f"{k} heartbeat {age:.0f} min old")
            continue
        live += 1
        quiet = (now - max(hb.get("last_push") or 0, hb.get("since") or 0)) / 60
        if hb["state"] == "training" and quiet > 30:
            problems.append(f"{k} has not pushed for {quiet:.0f} min")
        if hb["restarts"] >= 3:
            problems.append(f"{k} restarted {hb['restarts']} times")
    if (m or {}).get("state") != "done" and not (d / "STOP").exists() and live < expect:
        problems.append(f"{live} live shards, expected {expect}")
    if (m or {}).get("state") == "done":
        verdict = "DONE"
    else:
        verdict = "ESCALATE" if problems else "HEALTHY"
    return verdict, lines, problems


def cmd_status(a) -> None:
    verdict, lines, problems = status(a.dir, a.expect)
    print(f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC {verdict}" +
          (": " + "; ".join(problems) if problems else ""))
    for line in lines:
        print("  " + line)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=0)
    w = sub.add_parser("work")
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--dir", type=Path, required=True)
    w.add_argument("--threads", type=int, default=4)
    w.add_argument("--producers", type=int, default=5)
    w.add_argument("--push-minutes", type=float, default=10.0)
    g = sub.add_parser("merger")
    g.add_argument("--dir", type=Path, required=True)
    g.add_argument("--stop-at", type=float, required=True)
    g.add_argument("--base-cpu-hours", type=float, required=True)
    g.add_argument("--every-hours", type=float, default=3.0)
    s = sub.add_parser("status")
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--expect", type=int, default=3)
    a = ap.parse_args(argv)
    {"publish": cmd_publish, "work": cmd_work, "merger": cmd_merger, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
