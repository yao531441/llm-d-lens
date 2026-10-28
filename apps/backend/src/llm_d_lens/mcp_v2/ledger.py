"""Ownership ledger and plan cache for MCP v2.

Lens does not expose who created a deployment, and storage volumes and model caches carry no free-form
tag. The ledger is therefore the authority for "created by this agent's Lens user". Anything missing from
it is treated as someone else's and is read-only for the agent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_TTL_SECONDS = 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    owner TEXT NOT NULL,
    cluster_id TEXT,
    lens_run_id TEXT,
    lens_execution_id TEXT,
    root_id TEXT NOT NULL,
    name TEXT,
    state TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    deleted_at REAL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS resources_owner ON resources (owner, kind, state);
CREATE INDEX IF NOT EXISTS resources_execution ON resources (lens_execution_id);
CREATE INDEX IF NOT EXISTS resources_root ON resources (root_id);
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    payload TEXT NOT NULL
);
"""


def default_ledger_path() -> Path:
    override = os.environ.get("LENS_MCP_V2_LEDGER")
    if override:
        return Path(override)
    from llm_d_lens.infrastructure.storage.paths import storage_path

    return storage_path("data", "mcp_v2", "ledger.sqlite3")


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    owner: str
    cluster_id: str | None
    lens_run_id: str | None
    lens_execution_id: str | None
    root_id: str
    name: str | None
    state: str
    created_at: float
    deleted_at: float | None
    meta: dict[str, Any]


def _resource(row: sqlite3.Row) -> Resource:
    return Resource(**{**dict(row), "meta": json.loads(row["meta"])})


class Ledger:
    def __init__(self, path: Path | str):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def _write(self, sql: str, args: tuple = ()) -> None:
        with self._lock, self._db:
            self._db.execute(sql, args)

    def _read(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def add_resource(
        self,
        *,
        kind: str,
        owner: str,
        cluster_id: str | None,
        name: str | None = None,
        lens_run_id: str | None = None,
        lens_execution_id: str | None = None,
        root_id: str | None = None,
        meta: dict[str, Any] | None = None,
        state: str = "active",
    ) -> Resource:
        resource_id = f"{kind[:3]}-{uuid.uuid4().hex[:8]}"
        self._write(
            "INSERT INTO resources (id, kind, owner, cluster_id, lens_run_id, lens_execution_id, root_id, name,"
            " state, created_at, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                resource_id,
                kind,
                owner,
                cluster_id,
                lens_run_id,
                lens_execution_id,
                root_id or resource_id,
                name,
                state,
                time.time(),
                json.dumps(meta or {}),
            ),
        )
        return self.get(resource_id)  # type: ignore[return-value]

    def get(self, resource_id: str) -> Resource | None:
        rows = self._read("SELECT * FROM resources WHERE id = ?", (resource_id,))
        return _resource(rows[0]) if rows else None

    def find_by_execution(self, execution_id: str) -> Resource | None:
        rows = self._read(
            "SELECT * FROM resources WHERE lens_execution_id = ? AND state != 'deleted' ORDER BY created_at DESC",
            (execution_id,),
        )
        return _resource(rows[0]) if rows else None

    def list_resources(
        self, *, owner: str, kind: str | None = None, cluster_id: str | None = None, include_deleted: bool = False
    ) -> list[Resource]:
        sql, args = "SELECT * FROM resources WHERE owner = ?", [owner]
        if kind:
            sql, args = sql + " AND kind = ?", [*args, kind]
        if cluster_id:
            sql, args = sql + " AND cluster_id = ?", [*args, cluster_id]
        if not include_deleted:
            sql += " AND state != 'deleted'"
        return [_resource(row) for row in self._read(sql + " ORDER BY created_at DESC", tuple(args))]

    def children(self, root_id: str) -> list[Resource]:
        rows = self._read(
            "SELECT * FROM resources WHERE root_id = ? AND id != ? AND state != 'deleted'", (root_id, root_id)
        )
        return [_resource(row) for row in rows]

    def bind_run(self, resource_id: str, run_id: str) -> None:
        self._write("UPDATE resources SET lens_run_id = ? WHERE id = ?", (run_id, resource_id))

    def bind_execution(self, resource_id: str, execution_id: str) -> None:
        self._write("UPDATE resources SET lens_execution_id = ? WHERE id = ?", (execution_id, resource_id))

    def set_state(self, resource_id: str, state: str) -> None:
        deleted_at = time.time() if state == "deleted" else None
        self._write("UPDATE resources SET state = ?, deleted_at = ? WHERE id = ?", (state, deleted_at, resource_id))

    def put_plan(self, *, owner: str, cluster_id: str, payload: dict[str, Any]) -> str:
        now = time.time()
        self._write("DELETE FROM plans WHERE expires_at < ?", (now,))
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        self._write(
            "INSERT INTO plans (plan_id, owner, cluster_id, created_at, expires_at, payload) VALUES (?,?,?,?,?,?)",
            (plan_id, owner, cluster_id, now, now + PLAN_TTL_SECONDS, json.dumps(payload)),
        )
        return plan_id

    def get_plan(self, plan_id: str, owner: str) -> dict[str, Any] | None:
        rows = self._read(
            "SELECT * FROM plans WHERE plan_id = ? AND owner = ? AND expires_at >= ?", (plan_id, owner, time.time())
        )
        return json.loads(rows[0]["payload"]) if rows else None

    def drop_plan(self, plan_id: str) -> None:
        self._write("DELETE FROM plans WHERE plan_id = ?", (plan_id,))
