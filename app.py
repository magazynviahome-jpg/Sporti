# app.py — Sport Manager (Streamlit + SQLAlchemy + Postgres/Neon)
# Nowości:
# - Przy tworzeniu grupy możesz dodać wiele zajęć w tym samym dniu (format: "HH:MM;Nazwa")
# - Aplikacja generuje 12 tygodni wydarzeń dla KAŻDEGO slotu (czas + nazwa)
# - events.name (nazwa zajęć) — pokazywana przy wyborze wydarzenia
# Pozostałe funkcje: logowanie telefonem, filtry, zapisy/płatności, gole/asysty, usuwanie grupy

import os
from datetime import datetime, date, timedelta, time as dt_time
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String,
    DateTime, Boolean, ForeignKey, UniqueConstraint, select, func,
    insert, update, and_, text
)
from sqlalchemy.engine import Engine

# ---------------------------
# Sekrety / ENV
# ---------------------------
def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

def _get_database_url() -> str:
    url = _get_secret("DATABASE_URL", "")
    if url:
        return url
    os.makedirs("data", exist_ok=True)
    return "sqlite:///data/sport.db"

DATABASE_URL = _get_database_url()
DB_SCHEMA = (_get_secret("DB_SCHEMA", "public") or "public").strip()

# ---------------------------
# DB Engine
# ---------------------------
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=0,
    future=True,
    connect_args={"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 5}
)

metadata = MetaData()

# ---------------------------
# Katalogi dyscyplin
# ---------------------------
TEAM_SPORTS = [
    "Piłka nożna (Hala)",
    "Siatkówka (Hala)",
    "Koszykówka (Hala)",
    "Piłka ręczna (Hala)",
    "Hokej halowy",
    "Piłka nożna (Orlik)",
    "Koszykówka (Street)",
    "Rugby (Boisko)",
    "Siatkówka plażowa",
    "Piłka nożna plażowa",
]

FITNESS_CLASSES = [
    "Fitness: Cross",
    "Fitness: Trening obwodowy",
    "Fitness: Joga",
    "Fitness: Pilates",
    "Fitness: Mobility & Stretching",
    "Fitness: Zdrowy kręgosłup",
    "Fitness: HIIT",
    "Fitness: Indoor Cycling",
]

ALL_DISCIPLINES = TEAM_SPORTS + FITNESS_CLASSES

def is_team_sport(sport_name: str) -> bool:
    return sport_name in TEAM_SPORTS

# ---------------------------
# Tabele
# ---------------------------
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("phone", String(64)),
    Column("is_admin", Boolean, nullable=False, server_default=text("false") if engine.dialect.name != "sqlite" else text("0")),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

groups = Table(
    "groups", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("city", String(255), nullable=False),
    Column("venue", String(255), nullable=False),
    Column("weekday", Integer, nullable=False),
    Column("start_time", String(5), nullable=False),
    Column("price_cents", Integer, nullable=False),
    Column("duration_minutes", Integer, nullable=False, server_default=text("60")),
    Column("blik_phone", String(64), nullable=False, server_default=text("''") if engine.dialect.name=="postgresql" else text("''")),
    Column("sport", String(64), nullable=False, server_default=text("'Piłka nożna (Hala)'")),
    Column("postal_code", String(16), nullable=True),
    Column("created_by", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="SET NULL"), nullable=False),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

memberships = Table(
    "memberships", metadata,
    Column("user_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", Integer, ForeignKey(f"{DB_SCHEMA}.groups.id", ondelete="CASCADE"), primary_key=True),
    Column("role", String(16), nullable=False, server_default=text("'member'")),
    schema=DB_SCHEMA,
)

events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True),
    Column("group_id", Integer, ForeignKey(f"{DB_SCHEMA}.groups.id", ondelete="CASCADE"), nullable=False),
    Column("starts_at", DateTime, nullable=False),
    Column("price_cents", Integer, nullable=False),
    Column("generated", Boolean, nullable=False, server_default=text("true") if engine.dialect.name!="sqlite" else text("1")),
    Column("locked", Boolean, nullable=False, server_default=text("false") if engine.dialect.name!="sqlite" else text("0")),
    Column("name", String(255), nullable=True),  # nazwa zajęć (np. "Pilates", "Joga")
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

event_signups = Table(
    "event_signups", metadata,
    Column("event_id", Integer, ForeignKey(f"{DB_SCHEMA}.events.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="CASCADE"), primary_key=True),
    Column("signed_at", DateTime, nullable=False),
    schema=DB_SCHEMA,
)

payments = Table(
    "payments", metadata,
    Column("event_id", Integer, ForeignKey(f"{DB_SCHEMA}.events.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="CASCADE"), primary_key=True),
    Column("user_marked_paid", Boolean, nullable=False, server_default=text("false")),
    Column("moderator_confirmed", Boolean, nullable=False, server_default=text("false")),
    schema=DB_SCHEMA,
)

teams = Table(
    "teams", metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey(f"{DB_SCHEMA}.events.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(255), nullable=False),
    Column("idx", Integer, nullable=False),
    Column("goals", Integer, nullable=False, server_default=text("0")),
    UniqueConstraint("event_id", "idx", name="uq_teams_event_idx"),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

team_members = Table(
    "team_members", metadata,
    Column("team_id", Integer, ForeignKey(f"{DB_SCHEMA}.teams.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="CASCADE"), primary_key=True),
    schema=DB_SCHEMA,
)

goals = Table(
    "goals", metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey(f"{DB_SCHEMA}.events.id", ondelete="CASCADE"), nullable=False),
    Column("scorer_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="SET NULL"), nullable=False),
    Column("assist_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="SET NULL")),
    Column("minute", Integer),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

def init_db():
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.groups ADD COLUMN IF NOT EXISTS duration_minutes INTEGER NOT NULL DEFAULT 60;")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.groups ADD COLUMN IF NOT EXISTS blik_phone TEXT NOT NULL DEFAULT '';")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.groups ADD COLUMN IF NOT EXISTS sport TEXT NOT NULL DEFAULT 'Piłka nożna (Hala)';")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.groups ADD COLUMN IF NOT EXISTS postal_code TEXT;")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.memberships ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member';")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.events ADD COLUMN IF NOT EXISTS name TEXT;")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_events_group_starts ON {DB_SCHEMA}.events (group_id, starts_at);")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_signups_event ON {DB_SCHEMA}.event_signups (event_id);")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_payments_event_user ON {DB_SCHEMA}.payments (event_id, user_id);")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_goals_event_scorer ON {DB_SCHEMA}.goals (event_id, scorer_id);")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_goals_event_assist ON {DB_SCHEMA}.goals (event_id, assist_id);")

# ---------------------------
# Utils
# ---------------------------
def cents_to_str(cents: int) -> str:
    return f"{cents/100:.2f} zł"

def time_label(weekday: int, hhmm: str) -> str:
    days = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Nd"]
    return f"{days[weekday]} {hhmm}"

def next_dates_for_weekday(start_from: date, weekday: int, count: int) -> List[date]:
    days_ahead = (weekday - start_from.weekday()) % 7
    first = start_from + timedelta(days=days_ahead)
    return [first + timedelta(days=7*i) for i in range(count)]

# ---------------------------
# Cache helpers
# ---------------------------
@st.cache_data(ttl=30)
def cached_list_groups_for_user(user_id: int, schema: str,
                                activity_type: Optional[str],
                                discipline: Optional[str],
                                city_filter: str,
                                postal_filter: str) -> pd.DataFrame:
    sql = f"""
    SELECT g.id, g.name, g.city, g.venue, g.weekday, g.start_time, g.price_cents,
           g.duration_minutes, g.blik_phone, g.sport, g.postal_code,
           CASE WHEN m.role='moderator' THEN 1 ELSE 0 END AS is_mod
    FROM {schema}.groups g
    JOIN {schema}.memberships m ON m.group_id=g.id
    WHERE m.user_id=%(uid)s
    """
    params = {"uid": int(user_id)}

    if activity_type == "Sporty drużynowe":
        sql += " AND g.sport = ANY (%(ts)s)"
        params["ts"] = TEAM_SPORTS
    elif activity_type == "Zajęcia fitness":
        sql += " AND g.sport = ANY (%(fs)s)"
        params["fs"] = FITNESS_CLASSES

    if discipline and discipline != "Wszystkie":
        sql += " AND g.sport=%(sp)s"
        params["sp"] = discipline

    if city_filter:
        sql += " AND LOWER(g.city) LIKE %(city)s"
        params["city"] = f"%{city_filter.lower()}%"

    if postal_filter:
        sql += " AND COALESCE(g.postal_code,'') ILIKE %(pc)s"
        params["pc"] = f"%{postal_filter}%"

    sql += " ORDER BY g.city, g.name"
    return pd.read_sql_query(sql, engine, params=params)

@st.cache_data(ttl=30)
def cached_all_groups(uid: int, schema: str,
                      activity_type: Optional[str],
                      discipline: Optional[str],
                      city_filter: str,
                      postal_filter: str) -> pd.DataFrame:
    sql = f"""
    SELECT g.id, g.name, g.city, g.venue, g.weekday, g.start_time, g.price_cents,
           g.duration_minutes, g.blik_phone, g.sport, g.postal_code,
           EXISTS (
             SELECT 1 FROM {schema}.memberships m
             WHERE m.user_id=%(u)s AND m.group_id=g.id
           ) AS is_member
    FROM {schema}.groups g
    WHERE 1=1
    """
    params = {"u": int(uid)}

    if activity_type == "Sporty drużynowe":
        sql += " AND g.sport = ANY (%(ts)s)"
        params["ts"] = TEAM_SPORTS
    elif activity_type == "Zajęcia fitness":
        sql += " AND g.sport = ANY (%(fs)s)"
        params["fs"] = FITNESS_CLASSES

    if discipline and discipline != "Wszystkie":
        sql += " AND g.sport=%(sp)s"
        params["sp"] = discipline

    if city_filter:
        sql += " AND LOWER(g.city) LIKE %(city)s"
        params["city"] = f"%{city_filter.lower()}%"

    if postal_filter:
        sql += " AND COALESCE(g.postal_code,'') ILIKE %(pc)s"
        params["pc"] = f"%{postal_filter}%"

    sql += " ORDER BY g.city, g.name"
    return pd.read_sql_query(sql, engine, params=params)

@st.cache_data(ttl=20)
def cached_events_df(group_id: int, schema: str) -> pd.DataFrame:
    base = f"SELECT id, starts_at, price_cents, locked, name FROM {schema}.events WHERE group_id=%(gid)s ORDER BY starts_at"
    return pd.read_sql_query(base, engine, params={"gid": int(group_id)}, parse_dates=["starts_at"])

@st.cache_data(ttl=20)
def cached_signups(event_id: int, schema: str) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT es.user_id, u.name
        FROM {schema}.event_signups es
        JOIN {schema}.users u ON u.id=es.user_id
        WHERE es.event_id=%(eid)s
        ORDER BY u.name
        """,
        engine, params={"eid": int(event_id)}
    )

@st.cache_data(ttl=20)
def cached_signups_with_payments(event_id: int, schema: str) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT es.user_id, u.name,
               COALESCE(p.user_marked_paid, false) AS user_marked_paid,
               COALESCE(p.moderator_confirmed, false) AS moderator_confirmed
        FROM {schema}.event_signups es
        JOIN {schema}.users u ON u.id=es.user_id
        LEFT JOIN {schema}.payments p ON p.event_id=es.event_id AND p.user_id=es.user_id
        WHERE es.event_id=%(eid)s
        ORDER BY u.name
        """,
        engine, params={"eid": int(event_id)}
    )

@st.cache_data(ttl=20)
def cached_event_goals(event_id: int, schema: str) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT g.id, g.event_id, g.scorer_id, s.name AS scorer_name,
               g.assist_id, a.name AS assist_name,
               g.minute
        FROM {schema}.goals g
        LEFT JOIN {schema}.users s ON s.id=g.scorer_id
        LEFT JOIN {schema}.users a ON a.id=g.assist_id
        WHERE g.event_id=%(eid)s
        ORDER BY COALESCE(g.minute,9999), g.id
        """,
        engine, params={"eid": int(event_id)}
    )

# ---------------------------
# Role / zaległości
# ---------------------------
def is_moderator(user_id: int, group_id: int) -> bool:
    with engine.begin() as conn:
        q = select(memberships.c.user_id).where(
            and_(memberships.c.user_id == user_id,
                 memberships.c.group_id == group_id,
                 memberships.c.role == "moderator")
        )
        return conn.execute(q).first() is not None

def user_has_unpaid_past(user_id: int, group_id: int) -> bool:
    with engine.begin() as conn:
        sql = f"""
        SELECT EXISTS (
          SELECT 1
          FROM {DB_SCHEMA}.event_signups es
          JOIN {DB_SCHEMA}.events e ON e.id=es.event_id
          LEFT JOIN {DB_SCHEMA}.payments p ON p.event_id=es.event_id AND p.user_id=es.user_id
          WHERE es.user_id=%(u)s
            AND e.group_id=%(g)s
            AND e.starts_at < NOW()
            AND COALESCE(p.user_marked_paid, false) = false
        ) AS has_debt
        """
        return bool(conn.exec_driver_sql(sql, {"u": int(user_id), "g": int(group_id)}).scalar_one())

# ---------------------------
# Mutacje
# ---------------------------
def ensure_user(name: str, phone: str = "") -> int:
    with engine.begin() as conn:
        q = select(users.c.id).where(
            and_(users.c.name == name, func.coalesce(users.c.phone, "") == (phone or ""))
        )
        row = conn.execute(q).first()
        if row:
            uid = int(row.id)
            if phone:
                conn.execute(update(users).where(users.c.id == uid).values(phone=phone))
        else:
            uid = int(conn.execute(
                insert(users).values(name=name, phone=phone or None).returning(users.c.id)
            ).scalar_one())
        return uid

def join_group(user_id: int, group_id: int):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"""
            INSERT INTO {DB_SCHEMA}.memberships (user_id, group_id, role)
            VALUES (%(u)s, %(g)s, 'member')
            ON CONFLICT (user_id, group_id) DO NOTHING;
            """,
            {"u": int(user_id), "g": int(group_id)},
        )

def create_group(name: str, city: str, venue: str, weekday: int, start_time: str,
                 price_cents: int, blik_phone: str, created_by: int, duration_minutes: int = 60,
                 sport: str = "Piłka nożna (Hala)", postal_code: str = "") -> int:
    with engine.begin() as conn:
        gid = int(conn.execute(
            insert(groups).values(
                name=name, city=city, venue=venue, weekday=weekday, start_time=start_time,
                price_cents=price_cents, duration_minutes=duration_minutes, blik_phone=blik_phone,
                sport=sport, created_by=created_by, postal_code=postal_code or None
            ).returning(groups.c.id)
        ).scalar_one())
        conn.exec_driver_sql(
            f"""
            INSERT INTO {DB_SCHEMA}.memberships (user_id, group_id, role)
            VALUES (%(u)s, %(g)s, 'moderator')
            ON CONFLICT (user_id, group_id) DO NOTHING;
            """,
            {"u": created_by, "g": gid},
        )
        return gid

def delete_group(group_id: int):
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DELETE FROM {DB_SCHEMA}.groups WHERE id=%(g)s", {"g": int(group_id)})

def create_recurring_events(group_id: int, weekday: int, base_price_cents: int,
                            slots: List[Tuple[str, Optional[str]]], weeks_ahead: int = 12):
    """
    slots: lista (HH:MM, nazwa_zajec_lub_None)
    Dla każdego slotu generuje 12 nadchodzących terminów (co tydzień).
    """
    dates = next_dates_for_weekday(date.today(), weekday, weeks_ahead)
    with engine.begin() as conn:
        for hhmm, ev_name in slots:
            h, m = map(int, hhmm.split(":"))
            for d in dates:
                starts_at = datetime.combine(d, dt_time(hour=h, minute=m))
                exists = conn.execute(
                    select(events.c.id).where(and_(events.c.group_id == group_id, events.c.starts_at == starts_at))
                ).first()
                if not exists:
                    conn.execute(
                        insert(events).values(
                            group_id=group_id,
                            starts_at=starts_at,
                            price_cents=base_price_cents,
                            generated=True,
                            name=(ev_name.strip() if ev_name else None)
                        )
                    )

def upsert_events_for_group(group_id: int, weeks_ahead: int = 12):
    # zachowujemy dotychczasowe zachowanie dla pojedynczego bazowego slotu (g.start_time bez nazwy)
    with engine.begin() as conn:
        g = conn.execute(
            select(groups.c.weekday, groups.c.start_time, groups.c.price_cents).where(groups.c.id == group_id)
        ).first()
    if not g:
        return
    create_recurring_events(group_id, int(g.weekday), int(g.price_cents), [(g.start_time, None)], weeks_ahead)

def sign_up(event_id: int, user_id: int):
    now = datetime.now()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"""
            INSERT INTO {DB_SCHEMA}.event_signups (event_id, user_id, signed_at)
            VALUES (%(e)s, %(u)s, %(t)s)
            ON CONFLICT (event_id, user_id) DO NOTHING;
            """,
            {"e": int(event_id), "u": int(user_id), "t": now},
        )
        conn.exec_driver_sql(
            f"""
            INSERT INTO {DB_SCHEMA}.payments (event_id, user_id, user_marked_paid, moderator_confirmed)
            SELECT %(e)s, %(u)s, false, false
            WHERE NOT EXISTS (
               SELECT 1 FROM {DB_SCHEMA}.payments WHERE event_id=%(e)s AND user_id=%(u)s
            );
            """,
            {"e": int(event_id), "u": int(user_id)},
        )
        conn.exec_driver_sql(
            f"""
            INSERT INTO {DB_SCHEMA}.memberships (user_id, group_id, role)
            SELECT %(u)s, e.group_id, 'member' FROM {DB_SCHEMA}.events e WHERE e.id=%(e)s
            ON CONFLICT (user_id, group_id) DO NOTHING;
            """,
            {"u": int(user_id), "e": int(event_id)},
        )

def withdraw(event_id: int, user_id: int):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"DELETE FROM {DB_SCHEMA}.payments WHERE event_id=%(e)s AND user_id=%(u)s",
            {"e": int(event_id), "u": int(user_id)}
        )
        conn.exec_driver_sql(
            f"DELETE FROM {DB_SCHEMA}.event_signups WHERE event_id=%(e)s AND user_id=%(u)s",
            {"e": int(event_id), "u": int(user_id)}
        )

def payment_toggle(event_id: int, user_id: int, field: str, value: int):
    if field not in ("user_marked_paid", "moderator_confirmed"):
        return
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"UPDATE {DB_SCHEMA}.payments SET {field}=%(v)s WHERE event_id=%(e)s AND user_id=%(u)s",
            {"v": bool(value), "e": int(event_id), "u": int(user_id)},
        )

# ---- Goals CRUD ----
def add_goal(event_id: int, scorer_id: int, assist_id: Optional[int], minute: Optional[int]):
    with engine.begin() as conn:
        conn.execute(
            insert(goals).values(event_id=event_id, scorer_id=scorer_id, assist_id=assist_id or None, minute=minute)
        )

def update_goal(goal_id: int, scorer_id: int, assist_id: Optional[int], minute: Optional[int], editor_uid: int, is_mod: bool):
    with engine.begin() as conn:
        owner = conn.execute(select(goals.c.scorer_id).where(goals.c.id == goal_id)).scalar_one_or_none()
        if owner is None:
            return
        if (not is_mod) and int(owner) != int(editor_uid):
            return
        conn.execute(
            update(goals).where(goals.c.id == goal_id).values(scorer_id=scorer_id, assist_id=assist_id or None, minute=minute)
        )

def delete_goal(goal_id: int, editor_uid: int, is_mod: bool):
    with engine.begin() as conn:
        owner = conn.execute(select(goals.c.scorer_id).where(goals.c.id == goal_id)).scalar_one_or_none()
        if owner is None:
            return
        if (not is_mod) and int(owner) != int(editor_uid):
            return
        conn.exec_driver_sql(f"DELETE FROM {DB_SCHEMA}.goals WHERE id=%(g)s", {"g": int(goal_id)})

# ---------------------------
# UI helpers
# ---------------------------
def get_event(event_id: int):
    with engine.begin() as conn:
        return conn.execute(
            select(events.c.id, events.c.group_id, events.c.starts_at, events.c.price_cents, events.c.locked, events.c.name)
            .where(events.c.id == event_id)
        ).first()

def participants_table(group_id: int, event_id: int, show_pay=False):
    # rozpoznaj, czy sport drużynowy
    with engine.begin() as _c:
        grp = _c.execute(select(groups.c.sport).where(groups.c.id == group_id)).first()
    team_mode = bool(grp and is_team_sport(grp.sport))

    if show_pay:
        df = pd.read_sql_query(
            f"""
            SELECT es.user_id, u.name,
                   COALESCE(p.user_marked_paid,false) AS user_marked_paid,
                   COALESCE(p.moderator_confirmed,false) AS moderator_confirmed,
                   COALESCE(SUM(CASE WHEN g.scorer_id=es.user_id THEN 1 ELSE 0 END),0) AS goals,
                   COALESCE(SUM(CASE WHEN g.assist_id=es.user_id THEN 1 ELSE 0 END),0) AS assists
            FROM {DB_SCHEMA}.event_signups es
            JOIN {DB_SCHEMA}.users u ON u.id=es.user_id
            LEFT JOIN {DB_SCHEMA}.payments p ON p.event_id=es.event_id AND p.user_id=es.user_id
            LEFT JOIN {DB_SCHEMA}.goals g ON g.event_id=es.event_id
            WHERE es.event_id=%(eid)s
            GROUP BY es.user_id,u.name,p.user_marked_paid,p.moderator_confirmed
            ORDER BY user_marked_paid DESC, u.name
            """,
            engine, params={"eid": int(event_id)}
        )
    else:
        signups_df = cached_signups(event_id, DB_SCHEMA)
        if signups_df.empty:
            st.caption("Brak zapisanych.")
            return

        e = get_event(event_id)
        year = pd.to_datetime(e.starts_at).year

        stats = pd.read_sql_query(
            f"""
            SELECT u.id AS user_id,
                   u.name AS name_stat,
                   COALESCE(SUM(CASE WHEN g.scorer_id=u.id THEN 1 ELSE 0 END),0) AS goals,
                   COALESCE(SUM(CASE WHEN g.assist_id=u.id THEN 1 ELSE 0 END),0) AS assists
            FROM {DB_SCHEMA}.users u
            JOIN {DB_SCHEMA}.memberships m ON m.user_id=u.id AND m.group_id=%(gid)s
            LEFT JOIN {DB_SCHEMA}.events e ON e.group_id=m.group_id
            LEFT JOIN {DB_SCHEMA}.goals g ON g.event_id=e.id
            WHERE e.id IS NOT NULL
              AND EXTRACT(YEAR FROM e.starts_at)=%(yr)s
            GROUP BY u.id, u.name
            """,
            engine, params={"gid": int(e.group_id), "yr": int(year)}
        )

        df = signups_df.merge(stats, on="user_id", how="left", suffixes=("", "_stat"))
        if "name" not in df.columns and "name_stat" in df.columns:
            df["name"] = df["name_stat"]
        df["goals"] = df["goals"].fillna(0).astype(int)
        df["assists"] = df["assists"].fillna(0).astype(int)
        df = df.sort_values("name" if "name" in df.columns else "user_id")

    if team_mode:
        df["Statystyki"] = df.apply(lambda r: f"⚽ {int(r['goals'])}  |  🅰 {int(r['assists'])}", axis=1)
    else:
        df["Statystyki"] = "—"

    # kolumny widoku
    if "name" not in df.columns and "name_stat" in df.columns:
        df = df.rename(columns={"name_stat": "name"})
    if "name" not in df.columns:
        df["name"] = df["user_id"].astype(str)

    view_cols = ["name", "Statystyki"]
    if show_pay:
        df["Zapłacone"] = df["user_marked_paid"].astype(bool)
        df["Potwierdzone (mod)"] = df["moderator_confirmed"].astype(bool)
        view_cols += ["Zapłacone", "Potwierdzone (mod)"]

    st.dataframe(
        df.rename(columns={"name": "Uczestnik"})[["Uczestnik"] + [c for c in view_cols if c != "name"]],
        hide_index=True, use_container_width=True
    )

# ---------------------------
# Widoki wydarzeń
# ---------------------------
def upcoming_event_view(event_id: int, uid: int, duration_minutes: int):
    e = get_event(event_id)
    starts = pd.to_datetime(e.starts_at)
    gid = int(e.group_id)

    has_debt = user_has_unpaid_past(uid, gid)
    signups_df = cached_signups(event_id, DB_SCHEMA)
    is_signed = (not signups_df.empty) and (uid in set(signups_df["user_id"]))

    with st.container(border=True):
        title = starts.strftime("%d.%m.%Y %H:%M")
        if e.name:
            title += f" · {e.name}"
        st.subheader("Nadchodzące · " + title)

        if has_debt:
            st.error("**Niezapłacone poprzednie wydarzenie — brak możliwości zapisania się.** Przejdź do zakładki **Przeszłe** i oznacz płatność.")

        with st.form(f"up_ev_{event_id}", clear_on_submit=False):
            c1, c2 = st.columns([1,3])
            if is_signed:
                btn = c1.form_submit_button("Wypisz się", disabled=False)
                if btn:
                    withdraw(event_id, uid)
                    st.success("Wypisano z wydarzenia.")
            else:
                btn = c1.form_submit_button("Zapisz się", disabled=has_debt)
                if btn:
                    sign_up(event_id, uid)
                    st.success("Zapisano na wydarzenie.")

            count = 0 if signups_df.empty else len(signups_df)
            approx = (int(e.price_cents)/100) / max(1, count) if count else 0
            c2.caption(f"Obecnie zapisanych: **{count}** · przewidywany koszt/os.: **{approx:.2f} zł** (ostatecznie po meczu)")

        st.markdown("**Uczestnicy (roczne statystyki w grupie):**")
        participants_table(gid, event_id, show_pay=False)

def past_event_view(event_id: int, uid: int, duration_minutes: int, is_mod: bool, blik_phone: str):
    e = get_event(event_id)
    starts = pd.to_datetime(e.starts_at)
    signups_df = cached_signups_with_payments(event_id, DB_SCHEMA)
    count = 0 if signups_df.empty else len(signups_df)
    per_head = (int(e.price_cents) / 100 / max(1, count)) if count else 0.0

    with st.container(border=True):
        title = starts.strftime("%d.%m.%Y %H:%M")
        if e.name:
            title += f" · {e.name}"
        st.subheader("Przeszłe · " + title)
        st.markdown(f"**Cena obiektu:** {cents_to_str(int(e.price_cents))} · **Zapisanych:** {count} · **Kwota/os.:** **{per_head:.2f} zł**")

        with st.expander("💳 Zapłać / oznacz zapłatę"):
            st.markdown(f"**Numer BLIK / telefon:** `{blik_phone}`")
            if not signups_df.empty and uid in set(signups_df["user_id"]):
                my_row = signups_df[signups_df["user_id"] == uid].iloc[0]
                cur_paid = bool(my_row["user_marked_paid"])
                with st.form(f"pay_me_{event_id}", clear_on_submit=False):
                    new_paid = st.checkbox("Oznaczam: zapłacone", value=cur_paid)
                    paid_btn = st.form_submit_button("Zapisz")
                    if paid_btn and bool(new_paid) != bool(cur_paid):
                        payment_toggle(event_id, uid, 'user_marked_paid', int(bool(new_paid)))
                        st.success("Zapisano status płatności.")
            else:
                st.info("Nie byłeś zapisany na to wydarzenie.")

        st.markdown("**Uczestnicy · płatności + statystyki w tym meczu:**")
        participants_table(int(e.group_id), event_id, show_pay=True)

        # ---- Gole / Asysty (tylko dla sportów drużynowych sensownie, ale pozostawiamy opcję edycji) ----
        st.markdown("---")
        st.subheader("Gole i asysty (edytuj / dodaj)")

        signups = cached_signups(event_id, DB_SCHEMA)
        user_map = {int(r.user_id): r.name for r in signups.itertuples()}
        user_options = [(name, uid_) for uid_, name in sorted(user_map.items(), key=lambda x: x[1])]

        # Dodaj gol
        with st.form(f"add_goal_{event_id}", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([3,3,2,2])
            scorer = c1.selectbox("Strzelec", user_options, format_func=lambda x: x[0] if isinstance(x, tuple) else x, key=f"sc_{event_id}")
            assist = c2.selectbox("Asysta (opcjonalnie)", [("— brak —", None)] + user_options, format_func=lambda x: x[0] if isinstance(x, tuple) else x, key=f"as_{event_id}")
            minute = c3.number_input("Minuta", min_value=0, max_value=200, step=1, value=0)
            add_btn = c4.form_submit_button("Dodaj gola")
        if 'add_btn' in locals() and add_btn:
            scorer_id = int(scorer[1])
            assist_id = None if (assist[1] is None) else int(assist[1])
            # uprawnienia: własny gol lub moderator
            if is_mod or scorer_id == uid:
                add_goal(event_id, scorer_id, assist_id, int(minute))
                st.success("Dodano gola.")
                st.cache_data.clear()
            else:
                st.error("Możesz dodawać tylko własne gole. Moderator może dodać dowolne.")

        # Lista goli z edycją
        goals_df = cached_event_goals(event_id, DB_SCHEMA)
        if goals_df.empty:
            st.caption("Brak zapisanych goli.")
        else:
            for row in goals_df.itertuples():
                with st.container(border=True):
                    cols = st.columns([4,3,1.5,1.5])
                    cols[0].markdown(f"**Gol #{row.id}** — {row.scorer_name or '—'} (asysta: {row.assist_name or '—'})")
                    with cols[1].form(f"edit_goal_{row.id}", clear_on_submit=False):
                        # strzelec
                        sc_idx = 0
                        for i, (_, u) in enumerate(user_options):
                            if u == row.scorer_id:
                                sc_idx = i
                                break
                        sc_sel = st.selectbox("Strzelec", user_options, index=sc_idx, key=f"edit_sc_{row.id}")
                        # asysta
                        as_opts = [("— brak —", None)] + user_options
                        as_idx = 0
                        for i,(label,uidx) in enumerate(as_opts):
                            if uidx == row.assist_id:
                                as_idx = i
                                break
                        as_sel = st.selectbox("Asysta", as_opts, index=as_idx, key=f"edit_as_{row.id}")
                        minute_val = st.number_input("Minuta", min_value=0, max_value=200, step=1, value=int(row.minute) if row.minute is not None else 0, key=f"edit_min_{row.id}")
                        save = st.form_submit_button("Zapisz")
                    del_btn = cols[2].button("Usuń", key=f"del_goal_{row.id}")
                    cols[3].markdown("")

                    if save:
                        sc_id = int(sc_sel[1])
                        as_id = None if (as_sel[1] is None) else int(as_sel[1])
                        if is_mod or sc_id == uid or row.scorer_id == uid:
                            update_goal(int(row.id), sc_id, as_id, int(minute_val), editor_uid=uid, is_mod=is_mod)
                            st.success("Zaktualizowano gola.")
                            st.cache_data.clear()
                        else:
                            st.error("Brak uprawnień do edycji.")

                    if del_btn:
                        if is_mod or row.scorer_id == uid:
                            delete_goal(int(row.id), editor_uid=uid, is_mod=is_mod)
                            st.success("Usunięto gola.")
                            st.cache_data.clear()
                        else:
                            st.error("Brak uprawnień do usunięcia.")

# ---------------------------
# Strony
# ---------------------------
def sidebar_auth_and_filters():
    st.sidebar.header("Panel")

    # Logowanie / rejestracja — telefon wymagany (działa jako "hasło")
    name = st.sidebar.text_input("Imię / nick")
    phone = st.sidebar.text_input("Telefon (wymagany, działa jako hasło)")
    login = st.sidebar.button("Zaloguj / Rejestruj")

    if login:
        if not name.strip() or not phone.strip():
            st.sidebar.error("Podaj imię i telefon.")
        else:
            # sprawdź, czy user istnieje – jeśli tak, weryfikuj telefon
            with engine.begin() as conn:
                row = conn.execute(
                    select(users.c.id, users.c.phone).where(users.c.name == name.strip())
                ).first()
            if row:
                if (row.phone or "") != phone.strip():
                    st.sidebar.error("Błędny telefon dla tego użytkownika.")
                else:
                    st.session_state["user_id"] = int(row.id)
                    st.session_state["user_name"] = name.strip()
                    st.sidebar.success(f"Witaj, {name.strip()}!")
            else:
                uid = ensure_user(name.strip(), phone.strip())
                st.session_state["user_id"] = uid
                st.session_state["user_name"] = name.strip()
                st.sidebar.success(f"Utworzono konto. Witaj, {name.strip()}!")

    st.sidebar.markdown("---")

    # Filtry (bez nagłówka "Filtr: Sport")
    activity_type = st.sidebar.selectbox(
        "Typ aktywności",
        ["Wszystkie", "Sporty drużynowe", "Zajęcia fitness"],
        index=0
    )

    if activity_type == "Sporty drużynowe":
        discipline = st.sidebar.selectbox("Dyscyplina", ["Wszystkie"] + TEAM_SPORTS, index=0)
    elif activity_type == "Zajęcia fitness":
        discipline = st.sidebar.selectbox("Zajęcia", ["Wszystkie"] + FITNESS_CLASSES, index=0)
    else:
        discipline = st.sidebar.selectbox("Dyscyplina / Zajęcia", ["Wszystkie"] + ALL_DISCIPLINES, index=0)

    city_filter = st.sidebar.text_input("Miejscowość (filtr)", value="")
    postal_filter = st.sidebar.text_input("Kod pocztowy (filtr)", value="")

    st.session_state["activity_type"] = activity_type
    st.session_state["discipline"] = discipline
    st.session_state["city_filter"] = city_filter.strip()
    st.session_state["postal_filter"] = postal_filter.strip()

    if "user_id" in st.session_state:
        st.sidebar.info(f"Zalogowano jako: {st.session_state['user_name']}")
        if st.sidebar.button("Wyloguj"):
            for k in ["user_id","user_name","selected_group_id","selected_event_id","nav","go_panel","go_groups",
                      "activity_type","discipline","city_filter","postal_filter"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.sidebar.caption("Zaloguj się, aby zapisywać się i zarządzać wydarzeniami.")

def page_groups():
    st.header("Grupy")

    uid = st.session_state.get("user_id")
    activity_type = st.session_state.get("activity_type", "Wszystkie")
    discipline = st.session_state.get("discipline", "Wszystkie")
    city_filter = st.session_state.get("city_filter", "")
    postal_filter = st.session_state.get("postal_filter", "")

    # Moje grupy
    if uid:
        try:
            my_df = cached_list_groups_for_user(uid, DB_SCHEMA, activity_type, discipline, city_filter, postal_filter)
        except Exception as e:
            st.error(f"Nie mogę pobrać listy Twoich grup: {e}")
            my_df = pd.DataFrame()

        with st.expander("Twoje grupy", expanded=True):
            if my_df.empty:
                st.caption("Nie należysz jeszcze do żadnej grupy.")
            else:
                for _, g in my_df.iterrows():
                    with st.container(border=True):
                        cols = st.columns([3,2,2,2,1.2])
                        cols[0].markdown(f"**{g['name']}** · {g['sport']}\n\n{g['city']} ({g.get('postal_code','') or ''}) — {g['venue']}")
                        cols[1].markdown(f"{time_label(int(g['weekday']), g['start_time'])}")
                        cols[2].markdown(f"Cena: {cents_to_str(int(g['price_cents']))}")
                        cols[3].markdown(f"📱 BLIK: **{g['blik_phone']}**")
                        if cols[4].button("Wejdź", key=f"enter_my_{g['id']}"):
                            st.session_state["selected_group_id"] = int(g['id'])
                            st.session_state["go_panel"] = True
                            st.rerun()

    # Wszystkie grupy
    st.subheader("Wszystkie grupy")
    if uid is None:
        st.caption("Zaloguj się, aby dołączać i zapisywać się na wydarzenia.")
    try:
        all_df = cached_all_groups(uid or 0, DB_SCHEMA, activity_type, discipline, city_filter, postal_filter)
    except Exception as e:
        st.error(f"Nie mogę pobrać katalogu grup: {e}")
        return

    if all_df.empty:
        st.caption("Brak grup w systemie.")
    else:
        for _, g2 in all_df.iterrows():
            with st.container(border=True):
                c = st.columns([3,2,2,2,1.5])
                c[0].markdown(f"**{g2['name']}** · {g2['sport']}\n\n{g2['city']} ({g2.get('postal_code','') or ''}) — {g2['venue']}")
                c[1].markdown(f"{time_label(int(g2['weekday']), g2['start_time'])}")
                c[2].markdown(f"Cena: {cents_to_str(int(g2['price_cents']))}")
                c[3].markdown(f"📱 BLIK: **{g2['blik_phone']}**")
                if uid:
                    if bool(g2["is_member"]):
                        if c[4].button("Wejdź", key=f"enter_all_{g2['id']}"):
                            st.session_state["selected_group_id"] = int(g2['id'])
                            st.session_state["go_panel"] = True
                            st.rerun()
                    else:
                        if c[4].button("Dołącz", key=f"join_{g2['id']}"):
                            join_group(int(uid), int(g2['id']))
                            st.session_state["selected_group_id"] = int(g2['id'])
                            st.session_state["go_panel"] = True
                            st.rerun()
                else:
                    c[4].caption("Zaloguj się, aby wejść")

    # Utwórz nową grupę
    st.markdown("---")
    with st.expander("➕ Utwórz nową grupę", expanded=False):
        with st.form("create_group_form", clear_on_submit=False):
            col1, col2, col3 = st.columns([1.4,1,1])
            # Typ i dyscyplina
            activity_type_f = col1.selectbox("Typ aktywności", ["Sporty drużynowe", "Zajęcia fitness"])
            if activity_type_f == "Sporty drużynowe":
                sport_sel = col1.selectbox("Dyscyplina", TEAM_SPORTS, index=0)
            else:
                sport_sel = col1.selectbox("Zajęcia", FITNESS_CLASSES, index=0)

            name = col1.text_input("Nazwa grupy")
            city = col2.text_input("Miejscowość")
            postal_code = col2.text_input("Kod pocztowy (np. 00-001)")
            venue = col2.text_input("Miejsce wydarzenia (hala/boisko/plaża)")

            weekday = col3.selectbox("Dzień tygodnia", list(range(7)),
                                     format_func=lambda i: ["Pon","Wt","Śr","Czw","Pt","Sob","Nd"][i])
            start_time = col3.text_input("Godzina bazowa (HH:MM)", value="21:00")
            duration_minutes = col3.number_input("Czas gry / zajęć (min)", min_value=30, max_value=240, step=15, value=60)
            price = col3.number_input("Cena za obiekt/zajęcia (zł)", min_value=0.0, step=1.0)
            blik = col3.text_input("Numer BLIK/telefon do płatności")

            st.markdown(
                "#### Dodatkowe zajęcia w tym dniu (opcjonalnie)\n"
                "Wpisz **po jednej** linii w formacie: `HH:MM;Nazwa`  \n"
                "Przykład:\n"
                "`09:00;Pilates`\n"
                "`12:00;Joga`"
            )
            extra_raw = st.text_area("Lista slotów (godzina;nazwa)", height=120, key="extra_slots")

            submitted = st.form_submit_button("Utwórz grupę")
        if submitted:
            if "user_id" not in st.session_state:
                st.error("Zaloguj się, aby tworzyć grupy.")
            elif not all([name.strip(), city.strip(), venue.strip(), blik.strip()]):
                st.error("Uzupełnij wszystkie pola (w tym numer BLIK).")
            elif ":" not in start_time or len(start_time) != 5:
                st.error("Podaj **godzinę bazową** w formacie HH:MM (np. 21:00).")
            else:
                # parse extra slots
                slots: List[Tuple[str, Optional[str]]] = []
                # bazowy slot (bez nazwy)
                slots.append((start_time.strip(), None))
                if extra_raw.strip():
                    for line in extra_raw.strip().splitlines():
                        if ";" in line:
                            hhmm, nm = line.split(";", 1)
                            hhmm = hhmm.strip()
                            nm = nm.strip()
                            if len(hhmm) == 5 and ":" in hhmm and nm:
                                slots.append((hhmm, nm))
                        else:
                            # pozwól też na samą godzinę bez nazwy (rzadki przypadek)
                            hhmm = line.strip()
                            if len(hhmm) == 5 and ":" in hhmm:
                                slots.append((hhmm, None))

                try:
                    gid = create_group(
                        name.strip(), city.strip(), venue.strip(),
                        int(weekday), start_time.strip(),
                        int(round(price * 100)), blik.strip(),
                        int(st.session_state["user_id"]), int(duration_minutes),
                        sport_sel, postal_code.strip()
                    )
                    # Generuj wydarzenia dla WSZYSTKICH slotów
                    create_recurring_events(gid, int(weekday), int(round(price * 100)), slots, weeks_ahead=12)
                    st.success("Grupa i wydarzenia utworzone.")
                except Exception as e:
                    st.error(f"Nie udało się utworzyć grupy: {e}")

def page_group_dashboard(group_id: int):
    with engine.begin() as conn:
        g = conn.execute(
            select(
                groups.c.id, groups.c.name, groups.c.city, groups.c.venue, groups.c.weekday,
                groups.c.start_time, groups.c.price_cents, groups.c.duration_minutes, groups.c.blik_phone, groups.c.sport
            ).where(groups.c.id == group_id)
        ).first()
    if not g:
        st.error("Grupa nie istnieje")
        return

    gid, name, city, venue, weekday, start_time, price_cents, duration_minutes, blik_phone, sport = \
        int(g.id), g.name, g.city, g.venue, int(g.weekday), g.start_time, int(g.price_cents), int(g.duration_minutes), g.blik_phone, g.sport

    st.header(f"{name} — {city} · {venue} · {sport}")
    st.caption(f"Termin bazowy: {time_label(weekday, start_time)} · {duration_minutes} min · Cena: {cents_to_str(price_cents)} · BLIK: {blik_phone}")

    uid = st.session_state.get("user_id")
    if not uid:
        st.info("Zaloguj się, aby zapisywać się i zarządzać wydarzeniami.")
        return
    uid = int(uid)

    mod = is_moderator(uid, gid)

    section = st.radio("Sekcja", ["Nadchodzące", "Przeszłe", "Statystyki" + (" (admin)" if mod else "")],
                       horizontal=True, label_visibility="collapsed")

    if section == "Nadchodzące":
        df_all = cached_events_df(gid, DB_SCHEMA)
        if df_all.empty:
            st.info("Brak wydarzeń w kalendarzu")
        else:
            now = pd.Timestamp.now()
            future = df_all[df_all["starts_at"] >= now]
            if future.empty:
                st.caption("Brak nadchodzących wydarzeń.")
            else:
                def _fmt(i):
                    dt = pd.to_datetime(future.loc[future["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M")
                    nm = future.loc[future["id"]==i, "name"].values[0]
                    return f"{dt} · {nm}" if pd.notna(nm) and str(nm).strip() else dt
                pick = st.selectbox("Wybierz termin", list(future["id"]), format_func=_fmt)
                upcoming_event_view(int(pick), uid, duration_minutes)

    elif section == "Przeszłe":
        df_all = cached_events_df(gid, DB_SCHEMA)
        if df_all.empty:
            st.info("Brak wydarzeń")
        else:
            now = pd.Timestamp.now()
            past = df_all[df_all["starts_at"] < now]
            if past.empty:
                st.caption("Brak przeszłych wydarzeń.")
            else:
                def _fmtp(i):
                    dt = pd.to_datetime(df_all.loc[df_all["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M")
                    nm = df_all.loc[df_all["id"]==i, "name"].values[0]
                    return f"{dt} · {nm}" if pd.notna(nm) and str(nm).strip() else dt
                pickp = st.selectbox("Wybierz wydarzenie", list(past["id"])[::-1], format_func=_fmtp)
                past_event_view(int(pickp), uid, duration_minutes, mod, blik_phone)

    else:
        st.info("Tu później ranking i wykresy. Teraz priorytet: zapisy, płatności, gole/asysty.")
        if mod:
            st.markdown("---")
            st.subheader("Narzędzia moderatora")
            if st.button("Wygeneruj 12 kolejnych wydarzeń (bazowy slot)"):
                upsert_events_for_group(gid, 12)
                st.success("Dodano brakujące wydarzenia.")
            with st.expander("🛑 Usuń grupę (nieodwracalne)"):
                st.warning("Usunięcie grupy skasuje **wszystko** w tej grupie.")
                confirm_name = st.text_input("Przepisz nazwę grupy, aby potwierdzić:", key="del_confirm")
                colA, colB = st.columns([1,3])
                if colA.button("Usuń grupę", type="primary", use_container_width=True):
                    if confirm_name.strip() != name:
                        st.error("Nazwa nie pasuje.")
                    else:
                        try:
                            delete_group(gid)
                            st.success("Grupa usunięta.")
                            st.session_state.pop("selected_group_id", None)
                            st.session_state["go_groups"] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Nie udało się usunąć grupy: {e}")

# ---------------------------
# Main
# ---------------------------
def main():
    st.set_page_config("Sport Manager", layout="wide")
    init_db()

    if st.session_state.get("go_panel"):
        st.session_state["go_panel"] = False
        st.session_state["nav"] = "Panel grupy"
    if st.session_state.get("go_groups"):
        st.session_state["go_groups"] = False
        st.session_state["nav"] = "Grupy"

    sidebar_auth_and_filters()

    page = st.sidebar.radio("Nawigacja", ["Grupy", "Panel grupy"], key="nav", label_visibility="collapsed")

    if page == "Grupy":
        page_groups()
    else:
        gid = st.session_state.get("selected_group_id")
        if not gid:
            st.info("Wybierz grupę z listy (Grupy) lub dołącz do jednej.")
        else:
            page_group_dashboard(int(gid))

if __name__ == "__main__":
    main()
