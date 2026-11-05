"""HTTP routes for advisor mode: one session per game, kept on disk so a restart or a sleeping Mac loses nothing."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from arena.advisor.ops import do
from arena.advisor.session import Advisor, AdvisorError, new_session
from arena.api.manager import ApiError


def advisor_dir_from_env() -> Path:
    return Path(os.environ.get("ARENA_ADVISOR_DIR", "artifacts/advisor"))


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.advisor = Advisor()

    def _path(self, session_id: str) -> Path | None:
        return self.root / f"{session_id}.json" if self.root is not None else None

    def get(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id]
            path = self._path(session_id)
            if path is not None and session_id.isalnum() and path.is_file():
                self.sessions[session_id] = json.loads(path.read_text())
                return self.sessions[session_id]
        raise ApiError(404, "session_not_found", "no such advisor session on this server")

    def save(self, session: dict[str, Any]) -> None:
        path = self._path(session["id"])
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(session))
        tmp.replace(path)


def register_advisor(app: FastAPI, store: SessionStore) -> None:
    def guarded(fn):
        try:
            return fn()
        except AdvisorError as exc:
            raise ApiError(exc.status, exc.code, exc.message) from None

    @app.post("/advisor/sessions", status_code=201)
    def create_session(body: dict[str, Any]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            with store.lock:
                session = new_session(body)
                store.advisor.view(session)
                store.sessions[session["id"]] = session
                store.save(session)
                return store.advisor.view(session)

        return guarded(run)

    @app.get("/advisor/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        with store.lock:
            session = store.get(session_id)
            return guarded(lambda: store.advisor.view(session))

    @app.post("/advisor/sessions/{session_id}/do")
    def operate(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        with store.lock:
            session = store.get(session_id)
            before = json.dumps(session)

            def run() -> dict[str, Any]:
                do(store.advisor, session, body)
                return store.advisor.view(session)

            try:
                out = guarded(run)
            except Exception:
                store.sessions[session_id] = json.loads(before)
                raise
            store.save(session)
            return out
