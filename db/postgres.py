"""Postgres integration."""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from config import get_settings

try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    from psycopg2.extras import RealDictCursor
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False
    pg_pool = None  # type: ignore

_pool: Any = None


def _get_pool():
    global _pool
    if not _PG_AVAILABLE:
        raise RuntimeError("psycopg2 is not installed. Run: pip install psycopg2-binary")
    if _pool is None:
        s = get_settings()
        _pool = pg_pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=s.postgres_host,
            port=s.postgres_port,
            dbname=s.postgres_db,
            user=s.postgres_user,
            password=s.postgres_password,
        )
    return _pool


@contextmanager
def get_conn() -> Generator:
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def init_schema() -> None:
    """Create tables if they do not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS shelves (
        shelf_key     TEXT PRIMARY KEY,
        store_id      TEXT,
        planogram_id  TEXT,
        metadata      JSONB DEFAULT '{}'::jsonb,
        created_at    TIMESTAMPTZ DEFAULT NOW(),
        updated_at    TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS shelf_checks (
        id              SERIAL PRIMARY KEY,
        shelf_key       TEXT REFERENCES shelves(shelf_key),
        task_id         TEXT,
        compliance_json JSONB,
        assets          JSONB DEFAULT '[]'::jsonb,
        checked_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)


def upsert_shelf(shelf_key: str, store_id: str = "", planogram_id: str = "", metadata: Dict = None) -> None:
    sql = """
    INSERT INTO shelves (shelf_key, store_id, planogram_id, metadata)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (shelf_key) DO UPDATE
    SET store_id = EXCLUDED.store_id,
        planogram_id = EXCLUDED.planogram_id,
        metadata = EXCLUDED.metadata,
        updated_at = NOW()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (shelf_key, store_id, planogram_id, json.dumps(metadata or {})))


def get_shelf(shelf_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM shelves WHERE shelf_key = %s", (shelf_key,))
            row = cur.fetchone()
            return dict(row) if row else None


def list_shelves(store_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if store_id:
                cur.execute("SELECT * FROM shelves WHERE store_id = %s ORDER BY shelf_key", (store_id,))
            else:
                cur.execute("SELECT * FROM shelves ORDER BY shelf_key")
            return [dict(r) for r in cur.fetchall()]


def save_shelf_check(shelf_key: str, task_id: str, compliance_json: Dict, assets: List[str]) -> None:
    sql = """
    INSERT INTO shelf_checks (shelf_key, task_id, compliance_json, assets)
    VALUES (%s, %s, %s, %s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (shelf_key, task_id, json.dumps(compliance_json), json.dumps(assets)))
