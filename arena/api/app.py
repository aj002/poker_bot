"""HTTP and WebSocket transport for the playable table (spec 9.1).

Seat and host credentials travel in ``Authorization: Bearer ...`` headers, and over
WebSocket in the first message, never in URLs. HTTP is the only mutation path; the
stream only publishes committed projections. Run with::

    uv run uvicorn --factory arena.api.app:create_app --port 8317

``ARENA_STORE=postgres`` (with ``ARENA_DATABASE_URL``) keeps tables in PostgreSQL;
the default ``memory`` store lasts as long as the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from arena.api.clock import Clock, FakeClock
from arena.api.advisor import SessionStore, advisor_dir_from_env, register_advisor
from arena.api.manager import (
    ApiError,
    CreateTableRequest,
    JoinRequest,
    Principal,
    TableManager,
)
from arena.api.operator import is_operator, operator_token_from_env
from arena.api.party import same_origin
from arena.api.replay import ReplayCache, ReplayError, check_perspective
from arena.api.runs import RunManager, RunRejected, RunRequest
from arena.api.seats import BotRegistrationError
from arena.api.study import StudyLibrary, study_dir_from_env
from arena.api.styles import styles_catalog
from arena.domain.models import RULESET_IDS
from arena.persistence.store import MemoryStore, Store
from arena.table.projections import ProjectionDenied

log = logging.getLogger("arena.api")

DEFAULT_ORIGINS = ("http://localhost:5317", "http://127.0.0.1:5317")
WS_AUTH_TIMEOUT_S = 5.0
DEFAULT_DATABASE_URL = "postgresql://arena:arena@127.0.0.1:54329/arena"


def store_from_env() -> Store:
    kind = os.environ.get("ARENA_STORE", "memory")
    if kind == "memory":
        return MemoryStore()
    if kind == "postgres":
        from arena.persistence.postgres import PostgresStore

        return PostgresStore(os.environ.get("ARENA_DATABASE_URL", DEFAULT_DATABASE_URL))
    raise ValueError(f"ARENA_STORE must be memory or postgres, not {kind!r}")


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, "invalid_credential", "use Authorization: Bearer <credential>")
    return token.strip()


class RegisterBotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: StrictStr = Field(min_length=1, max_length=4096)


class AdvanceClockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ms: int = Field(ge=0, le=3_600_000)


def create_app(
    manager: TableManager | None = None,
    *,
    clock: Clock | None = None,
    allowed_origins: tuple[str, ...] | None = None,
    background_pump: bool = True,
    pump_interval_s: float = 0.02,
    operator_token: str | None = None,
    runs: RunManager | None = None,
    test_hooks: bool | None = None,
    study: StudyLibrary | None = None,
    advisor_dir: Path | str | None = "env",
) -> FastAPI:
    """``ARENA_FAKE_CLOCK=1`` (end-to-end tests only) swaps in a fake clock that
    advances solely through ``POST /__test__/clock/advance``."""
    if test_hooks is None:
        test_hooks = os.environ.get("ARENA_FAKE_CLOCK") == "1"
    if clock is None and manager is None and test_hooks:
        clock = FakeClock()
    if manager is None:
        manager = TableManager(
            clock,
            bot_delay_ms=int(os.environ.get("ARENA_BOT_DELAY_MS", "700")),
            hand_pause_ms=int(os.environ.get("ARENA_HAND_PAUSE_MS", "1500")),
            store=store_from_env(),
            writer_id=os.environ.get("ARENA_WRITER_ID", "local-writer"),
        )
    if operator_token is None:
        operator_token = operator_token_from_env()
    if runs is None:
        runs = RunManager(manager.bots)
    replays = ReplayCache()
    if study is None:
        study = StudyLibrary(study_dir_from_env())
    if allowed_origins is None:
        extra = tuple(o for o in os.environ.get("ARENA_ALLOWED_ORIGINS", "").split(",") if o)
        allowed_origins = DEFAULT_ORIGINS + extra

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()
        thread = None
        if background_pump:
            def pump_forever() -> None:
                while not stop.wait(pump_interval_s):
                    try:
                        manager.pump()
                    except Exception:  # noqa: BLE001 - the pump must outlive any one failure
                        log.exception("pump iteration failed")

            thread = threading.Thread(target=pump_forever, name="arena-pump", daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                await run_in_threadpool(thread.join, 5.0)
            await run_in_threadpool(manager.close)
            await run_in_threadpool(manager.bots.close)

    app = FastAPI(title="arena practice table", version="0.1.0", lifespan=lifespan)
    register_advisor(app, SessionStore(advisor_dir_from_env() if advisor_dir == "env" else advisor_dir))
    app.state.manager = manager
    app.state.runs = runs

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(ProjectionDenied)
    async def _denied(request: Request, exc: ProjectionDenied) -> JSONResponse:
        return JSONResponse({"error": {"code": "projection_denied", "message": str(exc)}}, status_code=403)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": "request rejected", "details": details}},
            status_code=422,
        )

    @app.exception_handler(RunRejected)
    async def _run_rejected(request: Request, exc: RunRejected) -> JSONResponse:
        return JSONResponse(ApiError(422, exc.code, exc.message, exc.details).body(), status_code=422)

    @app.exception_handler(BotRegistrationError)
    async def _bot_rejected(request: Request, exc: BotRegistrationError) -> JSONResponse:
        return JSONResponse(ApiError(422, exc.code, str(exc)).body(), status_code=422)

    def principal_for(request: Request, table_id: str) -> Principal:
        return manager.authenticate(table_id, _bearer(request))

    def operator(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and is_operator(operator_token, token.strip())

    def require_operator(request: Request) -> None:
        if not operator(request):
            status = 403 if request.headers.get("authorization") else 401
            raise ApiError(status, "operator_required", "this needs the operator credential")

    operator_only = Depends(require_operator)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "compatibility_status": "practice_assumptions"}

    @app.get("/catalog")
    async def catalog(request: Request) -> dict[str, Any]:
        is_op = operator(request)
        return {"rulesets": list(RULESET_IDS), "policies": manager.bots.catalog(operator=is_op),
                "max_seats": 8, "min_seats": 2, "operator": is_op}

    @app.get("/styles")
    async def styles() -> dict[str, Any]:
        return styles_catalog()

    @app.get("/study/packs")
    def study_packs() -> dict[str, Any]:
        return {"packs": study.packs(), "compatibility_status": "practice_assumptions"}

    @app.get("/study/packs/{pack_id}/hands")
    def study_hands(pack_id: str, filter: str = "all", offset: int = 0, limit: int = 50) -> dict[str, Any]:
        return study.hands(pack_id, filter=filter, offset=offset, limit=limit)

    @app.get("/study/packs/{pack_id}/hands/{hand_id}")
    def study_hand(pack_id: str, hand_id: str) -> dict[str, Any]:
        return study.hand(pack_id, hand_id)

    @app.post("/configs/resolve")
    def resolve(body: CreateTableRequest, request: Request) -> dict[str, Any]:
        return manager.resolve_config(body, operator=operator(request))

    @app.post("/tables", status_code=201)
    def create_table(body: CreateTableRequest, request: Request) -> dict[str, Any]:
        return manager.create_table(body, operator=operator(request))

    @app.post("/tables/{table_id}/join")
    def join(table_id: str, body: JoinRequest) -> dict[str, Any]:
        return manager.join(table_id, body.seat)

    @app.post("/tables/{table_id}/start")
    def start(table_id: str, request: Request) -> dict[str, Any]:
        return manager.start(table_id, principal_for(request, table_id))

    @app.get("/tables/{table_id}")
    def get_table(table_id: str, request: Request) -> dict[str, Any]:
        return manager.snapshot(table_id, principal_for(request, table_id))

    @app.post("/tables/{table_id}/actions")
    async def act(table_id: str, request: Request) -> dict[str, Any]:
        principal = await run_in_threadpool(principal_for, request, table_id)
        raw = await request.body()
        received_ns = manager.register_action(table_id)
        return await run_in_threadpool(manager.submit_action, table_id, principal, raw, received_ns,
                                       registered=True)

    @app.get("/tables/{table_id}/stats")
    def stats(table_id: str, request: Request) -> dict[str, Any]:
        return manager.stats(table_id, principal_for(request, table_id))

    @app.get("/tables/{table_id}/aids")
    def aids(table_id: str, request: Request) -> dict[str, Any]:
        return manager.aids(table_id, principal_for(request, table_id))

    @app.get("/tables/{table_id}/history")
    def history(table_id: str, request: Request, perspective: str | None = None) -> dict[str, Any]:
        return manager.history(table_id, principal_for(request, table_id), perspective)

    @app.get("/tables/{table_id}/replay")
    async def replay(table_id: str, request: Request, perspective: str | None = None,
                     hand: int | None = None) -> dict[str, Any]:
        principal = await run_in_threadpool(principal_for, request, table_id)
        seat = principal.seat if principal.role == "seat" else None
        log, complete = await run_in_threadpool(manager.committed_log, table_id)
        name = perspective or ("seat" if seat is not None else "spectator")
        try:
            chosen = check_perspective(name, seat=seat, host=principal.role == "host", complete=complete)
        except ValueError:
            raise ApiError(422, "invalid_perspective", "perspective must be seat, spectator or reviewer") from None
        notes = await run_in_threadpool(manager.reviewer_notes, table_id) if chosen == "reviewer" else None
        try:
            index = await asyncio.to_thread(replays.get, log, chosen, seat if chosen == "seat" else None, notes)
        except ReplayError as exc:
            raise ApiError(500, "replay_mismatch", str(exc)) from None
        hands = index["hands"]
        numbers = [h["hand_number"] for h in hands]
        if hand is None:
            hand = numbers[0]
        if hand not in numbers:
            raise ApiError(404, "hand_not_found", "no such hand in this table's log")
        available = [p for p, ok in (("seat", seat is not None), ("spectator", True),
                                     ("reviewer", principal.role == "host" and complete)) if ok]
        return {
            "table_id": table_id,
            "perspective": chosen,
            "viewer_seat": seat if chosen == "seat" else None,
            "table_complete": complete,
            "available_perspectives": available,
            "hands": [{"hand_number": h["hand_number"], "hand_id": h["hand_id"], "frames": len(h["frames"])}
                      for h in hands],
            "hand": hands[numbers.index(hand)],
            "compatibility_status": "practice_assumptions",
        }

    # ------------------------------------------------------------- operator

    @app.get("/operator", dependencies=[operator_only])
    async def whoami(request: Request) -> dict[str, Any]:
        return {"operator": True}

    @app.get("/bots", dependencies=[operator_only])
    async def list_bots(request: Request) -> dict[str, Any]:
        return {"bots": manager.bots.catalog(operator=True)}

    @app.post("/bots", status_code=201, dependencies=[operator_only])
    async def register_bot(body: RegisterBotRequest, request: Request) -> dict[str, Any]:
        spec = await asyncio.to_thread(manager.bots.register, body.path)
        return {"bot": spec.public(operator=True)}

    @app.post("/runs", status_code=202, dependencies=[operator_only])
    async def create_run(body: RunRequest, request: Request) -> dict[str, Any]:
        return runs.submit(body)

    @app.get("/runs", dependencies=[operator_only])
    async def list_runs(request: Request) -> dict[str, Any]:
        return {"runs": runs.summaries()}

    @app.get("/runs/{run_id}", dependencies=[operator_only])
    async def get_run(run_id: str, request: Request) -> dict[str, Any]:
        run = runs.get(run_id)
        if run is None:
            raise ApiError(404, "run_not_found", "no such run")
        return run

    if test_hooks:
        @app.post("/__test__/clock/advance")
        async def advance_clock(body: AdvanceClockRequest) -> dict[str, Any]:
            if not isinstance(manager.clock, FakeClock):
                raise ApiError(409, "real_clock", "this server runs on the real monotonic clock")
            manager.clock.advance_ms(body.ms)
            await run_in_threadpool(manager.pump)
            return {"now_ns": manager.clock.now_ns()}

    @app.websocket("/tables/{table_id}/stream")
    async def stream(ws: WebSocket, table_id: str) -> None:
        origin = ws.headers.get("origin")
        if origin is not None and origin not in allowed_origins and not same_origin(origin, ws.headers.get("host")):
            await ws.close(code=4403)
            return
        await ws.accept()
        try:
            hello = json.loads(await asyncio.wait_for(ws.receive_text(), WS_AUTH_TIMEOUT_S))
            if not isinstance(hello, dict) or hello.get("type") != "auth":
                raise ApiError(401, "auth_required", "first message must be {type: auth, token}")
            token = hello.get("token")
            principal = manager.authenticate(table_id, token if isinstance(token, str) else None)
            sub = await run_in_threadpool(manager.subscribe, table_id, principal, asyncio.get_running_loop())
        except ApiError as exc:
            await ws.send_json({"type": "error", **exc.body()})
            await ws.close(code=4000 + exc.status)
            return
        except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
            with contextlib.suppress(Exception):
                await ws.close(code=4401)
            return

        async def drain_client() -> None:
            while True:
                await ws.receive_text()

        reader = asyncio.create_task(drain_client())
        try:
            while True:
                frame_task = asyncio.create_task(sub.next())
                done, _ = await asyncio.wait({frame_task, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    frame_task.cancel()
                    break
                await ws.send_json(frame_task.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()
            await run_in_threadpool(manager.unsubscribe, table_id, sub)

    return app
