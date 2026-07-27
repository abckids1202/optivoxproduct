from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

from .config import DATABASE_PATH


class DatabaseError(RuntimeError):
    pass


def dict_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


@contextmanager
def get_connection(path=DATABASE_PATH):
    con = sqlite3.connect(path, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
      con.execute("PRAGMA busy_timeout=10000")
      con.execute("PRAGMA foreign_keys=ON")
      try:
          con.execute("PRAGMA journal_mode=WAL")
      except sqlite3.OperationalError:
          pass
      yield con
    except sqlite3.Error as exc:
      raise DatabaseError(str(exc)) from exc
    finally:
      con.close()


def fetch_all(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with get_connection() as con:
        return [dict(row) for row in con.execute(sql, tuple(params)).fetchall()]


def fetch_one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with get_connection() as con:
        return dict_row(con.execute(sql, tuple(params)).fetchone())


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    with get_connection() as con:
        cur = con.execute(sql, tuple(params))
        con.commit()
        return int(cur.lastrowid or cur.rowcount or 0)


def table_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with get_connection() as con:
        rows = con.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
        ).fetchall()
        for row in rows:
            name = row["name"]
            counts[name] = con.execute(f"select count(*) as c from {name}").fetchone()["c"]
    return counts

