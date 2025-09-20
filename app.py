# app.py — Futsal Manager (Streamlit + SQLAlchemy Core + Postgres) z kwalifikacją schematu

import os
from datetime import datetime, date, timedelta, time as dt_time
from typing import Optional, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String,
    DateTime, Boolean, ForeignKey, UniqueConstraint, select, func,
    insert, update, and_, text
)
from sqlalchemy.engine import Engine

APP_TITLE = "Futsal Manager (Postgres)"

# ---------------------------
# DB init (Postgres from secrets, fallback to SQLite)
# ---------------------------

def _get_secret(name: str, default: str = "") -> str:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

def _get_database_url() -> str:
    if "DATABASE_URL" in st.secrets or os.getenv("DATABASE_URL"):
        return _get_secret("DATABASE_URL")
    os.makedirs("data", exist_ok=True)
    return "sqlite:///data/futsal.db"

DATABASE_URL = _get_database_url()
DB_SCHEMA = _get_secret("DB_SCHEMA", "public").strip() or "public"

engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
metadata = MetaData()

# ---------------------------
# Tables (SQLAlchemy Core) — ze schematem
# ---------------------------

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("phone", String(64)),
    Column("email", String(255)),
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
    Column("created_by", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="SET NULL"), nullable=False),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

memberships = Table(
    "memberships", metadata,
    Column("user_id", Integer, ForeignKey(f"{DB_SCHEMA}.users.id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", Integer, ForeignKey(f"{DB_SCHEMA}.groups.id", ondelete="CASCADE"), primary_key=True),
    Column("role", String(16), nullable=False, server_default=text("'member'") if engine.dialect.name!="sqlite" else text("'member'")),
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
    Column("user_marked_paid", Boolean, nullable=False, server_default=text("false") if engine.dialect.name!="sqlite" else text("0")),
    Column("moderator_confirmed", Boolean, nullable=False, server_default=text("false") if engine.dialect.name!="sqlite" else text("0")),
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
    Column("team_id", Integer, ForeignKey(f"{DB_SCHEMA}.teams.id", ondelete="SET NULL")),
    Column("minute", Integer),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA,
)

def init_db():
    """Create tables and add missing columns (idempotent)."""
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.groups ADD COLUMN IF NOT EXISTS duration_minutes INTEGER NOT NULL DEFAULT 60;")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.groups ADD COLUMN IF NOT EXISTS blik_phone TEXT NOT NULL DEFAULT '';")
        conn.exec_driver_sql(f"ALTER TABLE {DB_SCHEMA}.memberships ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member';")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_events_group_starts ON {DB_SCHEMA}.events (group_id, starts_at);")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_signups_event ON {DB_SCHEMA}.event_signups (event_id);")
        conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS idx_payments_event_user ON {DB_SCHEMA}.payments (event_id, user_id);")

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

def is_moderator(user_id: int, group_id: int) -> bool:
    with engine.begin() as conn:
        q = select(memberships.c.user_id).where(
            and_(memberships.c.user_id == user_id,
                 memberships.c.group_id == group_id,
                 memberships.c.role == "moderator")
        )
        return conn.execute(q).first() is not None

# ---------------------------
# Auth (lightweight)
# ---------------------------

def ensure_user(name: str, phone: str = "", email: str = "") -> int:
    with engine.begin() as conn:
        q = select(users.c.id).where(
            and_(users.c.name == name, func.coalesce(users.c.email, "") == (email or ""))
        )
        row = conn.execute(q).first()
        if row:
            uid = int(row.id)
            conn.execute(
                update(users)
                .where(users.c.id == uid)
                .values(phone=phone or users.c.phone)
            )
        else:
            uid = int(conn.execute(
                insert(users).values(name=name, phone=phone or None, email=email or None).returning(users.c.id)
            ).scalar_one())
        return uid

# ---------------------------
# Data access
# ---------------------------

def create_group(name: str, city: str, venue: str, weekday: int, start_time: str,
                 price_cents: int, blik_phone: str, created_by: int, duration_minutes: int = 60) -> int:
    with engine.begin() as conn:
        gid = int(conn.execute(
            insert(groups).values(
                name=name, city=city, venue=venue, weekday=weekday, start_time=start_time,
                price_cents=price_cents, duration_minutes=duration_minutes, blik_phone=blik_phone,
                created_by=created_by
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

def upsert_events_for_group(group_id: int, weeks_ahead: int = 12):
    with engine.begin() as conn:
        g = conn.execute(
            select(groups.c.weekday, groups.c.start_time, groups.c.price_cents).where(groups.c.id == group_id)
        ).first()
        if not g:
            return
        weekday, start_time, price_cents = int(g.weekday), g.start_time, int(g.price_cents)
        today = date.today()
        dates = next_dates_for_weekday(today, weekday, weeks_ahead)
        for d in dates:
            h, m = map(int, start_time.split(":"))
            starts_at = datetime.combine(d, dt_time(hour=h, minute=m))
            exists = conn.execute(
                select(events.c.id).where(and_(events.c.group_id == group_id, events.c.starts_at == starts_at))
            ).first()
            if not exists:
                conn.execute(
                    insert(events).values(group_id=group_id, starts_at=starts_at, price_cents=price_cents, generated=True)
                )

def list_groups_for_user(user_id: int) -> pd.DataFrame:
    sql = f"""
    SELECT g.id, g.name, g.city, g.venue, g.weekday, g.start_time, g.price_cents, g.duration_minutes, g.blik_phone,
           CASE WHEN m.role='moderator' THEN 1 ELSE 0 END AS is_mod
    FROM {DB_SCHEMA}.groups g
    JOIN {DB_SCHEMA}.memberships m ON m.group_id=g.id
    WHERE m.user_id=%(uid)s
    ORDER BY g.city, g.name
    """
    return pd.read_sql_query(sql, engine, params={"uid": int(user_id)})

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

def get_group(group_id: int):
    with engine.begin() as conn:
        return conn.execute(
            select(
                groups.c.id, groups.c.name, groups.c.city, groups.c.venue, groups.c.weekday,
                groups.c.start_time, groups.c.price_cents, groups.c.duration_minutes, groups.c.blik_phone
            ).where(groups.c.id == group_id)
        ).first()

def events_df(group_id: int, only_future: bool = True) -> pd.DataFrame:
    if engine.dialect.name == "postgresql":
        cond = "starts_at >= NOW() - interval '1 day'"
    else:
        cond = "datetime(starts_at) >= datetime('now','-1 day')"
    base = f"SELECT id, starts_at, price_cents, locked FROM {DB_SCHEMA}.events WHERE group_id=%(gid)s"
    if only_future:
        base += f" AND {cond}"
    base += " ORDER BY starts_at"
    return pd.read_sql_query(base, engine, params={"gid": int(group_id)}, parse_dates=["starts_at"])

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
            VALUES (%(e)s, %(u)s, false, false)
            ON CONFLICT (event_id, user_id) DO NOTHING;
            """,
            {"e": int(event_id), "u": int(user_id)},
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

def get_event_context(event_id: int):
    with engine.begin() as conn:
        e = conn.execute(
            select(events.c.id, events.c.group_id, events.c.starts_at, events.c.price_cents, events.c.locked)
            .where(events.c.id == event_id)
        ).first()

        signups_sql = f"""
        SELECT es.user_id, u.name,
               COALESCE(p.user_marked_paid, false) AS user_marked_paid,
               COALESCE(p.moderator_confirmed, false) AS moderator_confirmed
        FROM {DB_SCHEMA}.event_signups es
        JOIN {DB_SCHEMA}.users u ON u.id=es.user_id
        LEFT JOIN {DB_SCHEMA}.payments p ON p.event_id=es.event_id AND p.user_id=es.user_id
        WHERE es.event_id=%(eid)s
        ORDER BY u.name
        """
        signups_df = pd.read_sql_query(signups_sql, engine, params={"eid": int(event_id)})

        teams_df = pd.read_sql_query(
            f"SELECT id, name, idx, goals FROM {DB_SCHEMA}.teams WHERE event_id=%(eid)s ORDER BY idx",
            engine, params={"eid": int(event_id)}
        )
    return e, signups_df, teams_df

def set_team_goals(team_id: int, goals_value: int):
    with engine.begin() as conn:
        conn.execute(update(teams).where(teams.c.id == team_id).values(goals=int(goals_value)))

def create_team_(event_id: int, name: str, idx: int) -> int:
    with engine.begin() as conn:
        return int(conn.execute(
            insert(teams).values(event_id=int(event_id), name=name, idx=int(idx), goals=0).returning(teams.c.id)
        ).scalar_one())

def add_member_to_team(team_id: int, user_id: int):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"""
            INSERT INTO {DB_SCHEMA}.team_members (team_id, user_id)
            VALUES (%(t)s, %(u)s)
            ON CONFLICT (team_id, user_id) DO NOTHING;
            """,
            {"t": int(team_id), "u": int(user_id)},
        )

def remove_member_from_team(team_id: int, user_id: int):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"DELETE FROM {DB_SCHEMA}.team_members WHERE team_id=%(t)s AND user_id=%(u)s",
            {"t": int(team_id), "u": int(user_id)}
        )

def list_team_members(team_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT tm.user_id, u.name
        FROM {DB_SCHEMA}.team_members tm JOIN {DB_SCHEMA}.users u ON u.id=tm.user_id
        WHERE tm.team_id=%(tid)s ORDER BY u.name
        """,
        engine, params={"tid": int(team_id)}
    )

def record_goal(event_id: int, scorer_id: int, assist_id: Optional[int], team_id: Optional[int], minute: Optional[int]):
    with engine.begin() as conn:
        conn.execute(
            insert(goals).values(
                event_id=int(event_id), scorer_id=int(scorer_id),
                assist_id=int(assist_id) if assist_id else None,
                team_id=int(team_id) if team_id else None,
                minute=int(minute) if minute is not None else None
            )
        )

def goals_df(event_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT g.id, g.minute, s.name AS scorer, a.name AS assist, t.name AS team
        FROM {DB_SCHEMA}.goals g
        LEFT JOIN {DB_SCHEMA}.users s ON s.id=g.scorer_id
        LEFT JOIN {DB_SCHEMA}.users a ON a.id=g.assist_id
        LEFT JOIN {DB_SCHEMA}.teams t ON t.id=g.team_id
        WHERE g.event_id=%(eid)s
        ORDER BY COALESCE(g.minute, 0), g.id
        """,
        engine, params={"eid": int(event_id)}
    )

def total_team_goals(event_id: int) -> Tuple[int, List[int]]:
    df = pd.read_sql_query(
        f"SELECT goals FROM {DB_SCHEMA}.teams WHERE event_id=%(eid)s ORDER BY idx",
        engine, params={"eid": int(event_id)}
    )
    ls = df["goals"].astype(int).tolist() if not df.empty else []
    return sum(ls), ls

def computed_stats(group_id: int, year: int) -> pd.DataFrame:
    df_users = pd.read_sql_query(f"SELECT id, name FROM {DB_SCHEMA}.users", engine)
    if engine.dialect.name == "postgresql":
        df_g = pd.read_sql_query(
            f"""
            SELECT g.event_id, g.scorer_id, g.assist_id
            FROM {DB_SCHEMA}.goals g JOIN {DB_SCHEMA}.events e ON e.id=g.event_id
            WHERE e.group_id=%(gid)s AND EXTRACT(YEAR FROM e.starts_at)=%(yr)s
            """,
            engine, params={"gid": int(group_id), "yr": int(year)}
        )
    else:
        df_g = pd.read_sql_query(
            f"""
            SELECT g.event_id, g.scorer_id, g.assist_id
            FROM {DB_SCHEMA}.goals g JOIN {DB_SCHEMA}.events e ON e.id=g.event_id
            WHERE e.group_id=%(gid)s AND strftime('%Y', e.starts_at)=%(yr)s
            """,
            engine, params={"gid": int(group_id), "yr": f"{year:04d}"}
        )

    df_tm = pd.read_sql_query(
        f"""
        SELECT t.event_id, t.id AS team_id, tm.user_id, t.goals
        FROM {DB_SCHEMA}.teams t JOIN {DB_SCHEMA}.team_members tm ON tm.team_id=t.id
        WHERE t.event_id IN (SELECT id FROM {DB_SCHEMA}.events WHERE group_id=%(gid)s)
        """,
        engine, params={"gid": int(group_id)}
    )
    df_tg = pd.read_sql_query(
        f"SELECT event_id, MAX(goals) AS maxg FROM {DB_SCHEMA}.teams GROUP BY event_id", engine
    )

    stats = {int(u): {"name": n, "goals": 0, "assists": 0, "wins": 0, "losses": 0, "draws": 0}
             for u, n in (df_users.itertuples(index=False) if not df_users.empty else [])}

    for _, row in (df_g.iterrows() if not df_g.empty else []):
        if pd.notna(row["scorer_id"]):
            stats[int(row["scorer_id"])]["goals"] += 1
        if pd.notna(row["assist_id"]):
            stats[int(row["assist_id"])]["assists"] += 1

    if not df_tm.empty and not df_tg.empty:
        merged = df_tm.merge(df_tg, on="event_id", how="left")
        for _, r in merged.iterrows():
            if pd.isna(r["maxg"]):
                continue
            if int(r["goals"]) == int(r["maxg"]):
                stats[int(r["user_id"])]["wins"] += 1
            elif int(r["goals"]) < int(r["maxg"]):
                stats[int(r["user_id"])]["losses"] += 1

    out = pd.DataFrame([{"user_id": uid, **s} for uid, s in stats.items()])
    if out.empty:
        return out
    out["points"] = out["wins"]*3 + out["draws"]
    return out.sort_values(["points","goals","assists"], ascending=False)

# ---------------------------
# UI helpers
# ---------------------------

def render_event_card(event_id: int):
    e, signups_df, teams_df = get_event_context(int(event_id))
    starts = pd.to_datetime(e.starts_at)
    gid = int(e.group_id)
    g = get_group(gid)
    duration_minutes = int(g.duration_minutes if g else 60)
    blik_phone = g.blik_phone if g else ""
    uid = st.session_state.get("user_id")

    with st.container(border=True):
        st.subheader(starts.strftime("%d.%m.%Y %H:%M"))
        c1, c2, c3 = st.columns([2,2,2])

        is_signed = (not signups_df.empty) and (uid in set(signups_df["user_id"]))
        if is_signed:
            if c1.button("Wypisz się", key=f"wd_{e.id}"):
                withdraw(e.id, uid)
                st.rerun()
        else:
            with engine.begin() as conn:
                pay = conn.exec_driver_sql(
                    f"SELECT user_marked_paid FROM {DB_SCHEMA}.payments WHERE event_id=%(e)s AND user_id=%(u)s",
                    {"e": int(e.id), "u": int(uid)}
                ).first()
            can_signup = bool(pay and pay[0])
            if not can_signup:
                c1.info("Aby zapisać się, kliknij najpierw 'Zapłacę BLIK' i opłać udział.")
            if can_signup and c1.button("Zapisz się", key=f"su_{e.id}"):
                sign_up(e.id, uid)
                st.rerun()

        with c2.expander("💳 Zapłacę BLIK"):
            st.markdown(f"**Numer do BLIK / telefon:** {blik_phone}")
            st.caption("Skopiuj numer, zapłać i oznacz poniżej.")
            row = None
            with engine.begin() as conn:
                row = conn.exec_driver_sql(
                    f"SELECT user_marked_paid FROM {DB_SCHEMA}.payments WHERE event_id=%(e)s AND user_id=%(u)s",
                    {"e": int(e.id), "u": int(uid)}
                ).first()
            cur_val = bool(row[0]) if row else False
            new_val = st.checkbox("Oznaczam: zapłacone", value=cur_val, key=f"ump_{e.id}")
            if bool(new_val) != bool(cur_val):
                payment_toggle(e.id, uid, 'user_marked_paid', int(bool(new_val)))
                if new_val and not is_signed:
                    st.success("Znakomicie! Teraz możesz się zapisać na grę.")

        count = 0 if signups_df.empty else len(signups_df)
        per_head = int(e.price_cents) / 100 / max(1, count)
        c3.metric("Zapisani", f"{count}")
        c3.metric("Koszt na osobę", f"{per_head:.2f} zł")
        if not signups_df.empty:
            st.dataframe(
                signups_df.rename(columns={
                    "name":"Uczestnik",
                    "user_marked_paid":"Zapłacone (użytkownik)",
                    "moderator_confirmed":"Potwierdzone (mod)"
                })[["Uczestnik","Zapłacone (użytkownik)","Potwierdzone (mod)"]],
                hide_index=True, use_container_width=True
            )
        else:
            st.caption("Brak zapisanych.")

# ---------------------------
# UI
# ---------------------------

def sidebar_auth():
    st.sidebar.header("Logowanie")
    st.sidebar.caption(f"🔌 DB: `{engine.dialect.name}` · schema: `{DB_SCHEMA}`")
    name = st.sidebar.text_input("Imię / nick", key="login_name")
    phone = st.sidebar.text_input("Telefon (opcjonalnie)")
    email = st.sidebar.text_input("Email (opcjonalnie)")
    login = st.sidebar.button("Zaloguj / Rejestruj")
    if login:
        if not name:
            st.sidebar.error("Podaj przynajmniej imię/nick")
        else:
            uid = ensure_user(name.strip(), phone.strip(), email.strip())
            st.session_state["user_id"] = uid
            st.session_state["user_name"] = name.strip()
            st.sidebar.success(f"Witaj, {name}!")

    with st.sidebar.expander("🧪 Diagnostyka"):
        st.write({
            "engine": engine.dialect.name,
            "schema": DB_SCHEMA,
            "DATABASE_URL_present": bool(("DATABASE_URL" in st.secrets) or os.getenv("DATABASE_URL")),
        })
        try:
            if engine.dialect.name == "postgresql":
                tables = pd.read_sql_query(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema=%(s)s ORDER BY table_name",
                    engine, params={"s": DB_SCHEMA}
                )
            else:
                tables = pd.read_sql_query("SELECT name AS table_name FROM sqlite_master WHERE type='table' ORDER BY name", engine)
            st.write("Tabele:", tables["table_name"].tolist())
        except Exception as e:
            st.error(f"DB check error: {e}")

    if "user_id" in st.session_state:
        st.sidebar.info(f"Zalogowano jako: {st.session_state['user_name']}")
        if st.sidebar.button("Wyloguj"):
            for k in ["user_id","user_name","selected_group_id","selected_event_id","nav"]:
                st.session_state.pop(k, None)
            st.rerun()

def page_groups():
    st.header("Twoje grupy")
    uid = st.session_state.get("user_id")
    if not uid:
        st.info("Zaloguj się z lewego panelu.")
        return

    with st.expander("➕ Utwórz nową grupę", expanded=False):
        col1, col2 = st.columns(2)
        name = col1.text_input("Nazwa grupy")
        city = col1.text_input("Miejscowość")
        venue = col1.text_input("Miejsce wydarzenia (hala)")
        weekday = col2.selectbox("Dzień tygodnia", list(range(7)), format_func=lambda i: ["Pon","Wt","Śr","Czw","Pt","Sob","Nd"][i])
        start_time = col2.text_input("Godzina startu (HH:MM)", value="21:00")
        duration_minutes = col2.number_input("Czas gry (min)", min_value=30, max_value=240, step=15, value=60)
        price = col2.number_input("Cena za halę (zł)", min_value=0.0, step=1.0)
        blik = col2.text_input("Numer BLIK/telefon do płatności")
        if st.button("Utwórz grupę"):
            if all([name, city, venue, blik]) and ":" in start_time:
                gid = create_group(name, city, venue, int(weekday), start_time, int(round(price*100)), blik, int(uid), int(duration_minutes))
                upsert_events_for_group(gid)
                st.success("Grupa utworzona. Dodano nadchodzące wydarzenia na 12 tygodni.")
            else:
                st.error("Uzupełnij wszystkie pola.")

    df = list_groups_for_user(uid)
    if df.empty:
        st.info("Nie należysz jeszcze do żadnej grupy. Utwórz grupę powyżej lub poproś moderatora o dodanie.")
        return

    for _, g in df.iterrows():
        with st.container(border=True):
            cols = st.columns([2,2,2,2,1])
            cols[0].markdown(f"**{g['name']}**\n\n{g['city']} — {g['venue']}")
            cols[1].markdown(f"{time_label(int(g['weekday']), g['start_time'])}")
            cols[2].markdown(f"Cena: {cents_to_str(int(g['price_cents']))} · {int(g['duration_minutes'])} min")
            cols[3].markdown(f"Płatność BLIK: **{g['blik_phone']}**")
            if cols[4].button("Wejdź", key=f"enter_{g['id']}"):
                st.session_state["selected_group_id"] = int(g['id'])
                upsert_events_for_group(int(g['id']))
                st.session_state["nav"] = "Panel grupy"
                st.rerun()

def page_group_dashboard(group_id: int):
    g = get_group(group_id)
    if not g:
        st.error("Grupa nie istnieje")
        return
    gid, name, city, venue, weekday, start_time, price_cents, duration_minutes, blik_phone = \
        int(g.id), g.name, g.city, g.venue, int(g.weekday), g.start_time, int(g.price_cents), int(g.duration_minutes), g.blik_phone

    st.header(f"{name} — {city} · {venue}")
    st.caption(f"Gramy: {time_label(weekday, start_time)} · {duration_minutes} min · Cena hali: {cents_to_str(price_cents)} · BLIK: {blik_phone}")

    uid = int(st.session_state.get("user_id"))
    mod = is_moderator(uid, gid)

    tabs = st.tabs(["Nadchodzące", "Płatności", "Drużyny & Wynik", "Statystyki" + (" (admin)" if mod else "")])

    with tabs[0]:
        df = events_df(gid, only_future=True)
        if df.empty:
            st.info("Brak nadchodzących lub bieżących wydarzeń")
        else:
            now = pd.Timestamp.now()
            def is_current(row):
                start = row["starts_at"]
                end = start + pd.Timedelta(minutes=duration_minutes)
                return (now >= start) and (now < end)
            current = df[df.apply(is_current, axis=1)]
            upcoming = df[df["starts_at"] >= now]

            st.subheader("🔥 Aktualne")
            if current.empty:
                st.caption("Brak wydarzenia w trakcie.")
            for _, ev in current.iterrows():
                render_event_card(int(ev["id"]))

            st.subheader("⏭️ Nadchodzące")
            if upcoming.empty:
                st.caption("Brak nadchodzących wydarzeń.")
            for _, ev in upcoming.iterrows():
                render_event_card(int(ev["id"]))

    with tabs[1]:
        df = events_df(gid, only_future=False)
        if df.empty:
            st.info("Brak wydarzeń")
        else:
            pick = st.selectbox("Wybierz wydarzenie", list(df["id"]),
                                format_func=lambda i: pd.to_datetime(df.loc[df["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M"))
            e, signups_df, _ = get_event_context(int(pick))
            st.subheader("Płatności — " + pd.to_datetime(e.starts_at).strftime("%d.%m.%Y %H:%M"))
            if mod:
                if not signups_df.empty:
                    for _, r in signups_df.iterrows():
                        cols = st.columns([3,1,1,1])
                        cols[0].markdown(f"**{r['name']}**")
                        cols[1].checkbox("Zapłacił", value=bool(r['user_marked_paid']), key=f"u_{pick}_{r['user_id']}", disabled=True)
                        new_conf = cols[2].checkbox("Potwierdź (mod)", value=bool(r['moderator_confirmed']), key=f"m_{pick}_{r['user_id']}")
                        if bool(new_conf) != bool(r['moderator_confirmed']):
                            payment_toggle(int(pick), int(r['user_id']), 'moderator_confirmed', int(bool(new_conf)))
                    st.caption("Uwaga: bez potwierdzenia moderatora płatność nie jest finalna.")
                else:
                    st.caption("Brak zapisanych.")
            else:
                if not signups_df.empty:
                    st.dataframe(
                        signups_df.rename(columns={"name":"Uczestnik","user_marked_paid":"Zapłacone (użytkownik)","moderator_confirmed":"Potwierdzone (mod)"})[
                            ["Uczestnik","Zapłacone (użytkownik)","Potwierdzone (mod)"]
                        ],
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.caption("Brak zapisanych.")

    with tabs[2]:
        df = events_df(gid, only_future=False)
        if df.empty:
            st.info("Brak wydarzeń")
        else:
            pick = st.selectbox("Wydarzenie", list(df["id"]), key="ev_for_teams",
                                format_func=lambda i: pd.to_datetime(df.loc[df["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M"))
            e, signups_df, teams_df = get_event_context(int(pick))

            if mod:
                st.subheader("Zarządzanie drużynami (moderator)")
                cols = st.columns(3)
                for i in range(3):
                    with cols[i]:
                        label = f"Drużyna {i+1}"
                        existing = teams_df[teams_df["idx"]==i+1]
                        if existing.empty:
                            if st.button(f"Dodaj {label}", key=f"add_team_{i}"):
                                create_team_(int(pick), label, i+1)
                                st.rerun()
                        else:
                            tid = int(existing.iloc[0]["id"])
                            st.markdown(f"**{existing.iloc[0]['name']}**")
                            cur_members = list_team_members(tid)
                            options = [] if signups_df.empty else [(int(u), n) for u, n in signups_df[["user_id","name"]].itertuples(index=False)]
                            if options:
                                add_choice = st.selectbox("Dodaj gracza", options=options, format_func=lambda x: x[1], key=f"sel_{tid}")
                                if st.button("➕ Dodaj", key=f"btn_add_{tid}"):
                                    add_member_to_team(tid, int(add_choice[0]))
                            if not cur_members.empty:
                                rem_choice = st.selectbox("Usuń gracza", options=[(int(u), n) for u, n in cur_members.itertuples(index=False)], format_func=lambda x: x[1], key=f"rem_{tid}")
                                if st.button("➖ Usuń", key=f"btn_rem_{tid}"):
                                    remove_member_from_team(tid, int(rem_choice[0]))
                            goals_val = st.number_input("Gole", min_value=0, step=1, value=int(existing.iloc[0]['goals']), key=f"g_{tid}")
                            if st.button("Zapisz gole", key=f"saveg_{tid}"):
                                set_team_goals(tid, int(goals_val)); st.success("Zapisano")

            st.divider()
            st.subheader("Gole i asysty (samo-raportowanie)")
            team_map = {int(r["id"]): f"{r['name']}" for _, r in teams_df.iterrows()}
            tchoice_keys = [None] + list(team_map.keys())
            team_choice = st.selectbox("Drużyna (opcjonalnie)", options=tchoice_keys, format_func=lambda x: "—" if x is None else team_map[x])
            if signups_df.empty:
                st.info("Brak zapisanych na to wydarzenie.")
            else:
                scorer_choice = st.selectbox("Strzelec", options=[(int(i), n) for i,n in signups_df[["user_id","name"]].itertuples(index=False)], format_func=lambda x: x[1])
                assist_choice = st.selectbox("Asysta (opcjonalnie)", options=[None] + [(int(i), n) for i,n in signups_df[["user_id","name"]].itertuples(index=False)], format_func=lambda x: "—" if x is None else x[1])
                minute = st.number_input("Minuta (opcjonalnie)", min_value=0, max_value=200, step=1, value=0)
                if st.button("Dodaj gola"):
                    record_goal(int(pick), int(scorer_choice[0]), int(assist_choice[0]) if assist_choice else None, int(team_choice) if team_choice else None, int(minute) if minute else None)
                    st.success("Dodano gola")

            st.markdown("### Lista goli")
            gdf = goals_df(int(pick))
            st.dataframe(gdf, hide_index=True, use_container_width=True)

            total_g, team_g = total_team_goals(int(pick))
            if len(gdf) != total_g and total_g > 0:
                st.warning(f"Suma goli wprowadzonych ({len(gdf)}) ≠ suma bramek drużyn ({total_g}). Uzupełnij dane.")
            elif total_g == 0:
                st.info("Podaj gole drużyn, aby móc zweryfikować zgodność.")
            else:
                st.success("Liczba goli się zgadza ✅")

    with tabs[3]:
        year = st.selectbox("Rok", options=list(range(datetime.now().year, datetime.now().year-5, -1)))
        df_stats = computed_stats(gid, int(year))
        if df_stats.empty:
            st.info("Brak statystyk na wybrany rok")
        else:
            st.dataframe(
                df_stats.rename(columns={"name":"Zawodnik","goals":"Gole","assists":"Asysty","wins":"Wygrane","losses":"Przegrane","draws":"Remisy","points":"Punkty"})[
                    ["Zawodnik","Gole","Asysty","Wygrane","Przegrane","Remisy","Punkty"]
                ],
                hide_index=True, use_container_width=True
            )

        if mod:
            st.markdown("---")
            st.subheader("Narzędzia moderatora")
            if st.button("Wygeneruj 12 kolejnych wydarzeń"):
                upsert_events_for_group(gid, 12)
                st.success("Dodano brakujące wydarzenia w kalendarzu.")

            st.markdown("### Zarządzanie członkami i rolami")
            members = pd.read_sql_query(
                f"""
                SELECT u.id AS user_id, u.name, m.role
                FROM {DB_SCHEMA}.memberships m JOIN {DB_SCHEMA}.users u ON u.id=m.user_id
                WHERE m.group_id=%(gid)s
                ORDER BY u.name
                """,
                engine, params={"gid": gid}
            )
            if members.empty:
                st.caption("Brak członków?")
            else:
                for _, r in members.iterrows():
                    cols = st.columns([3,2,2])
                    cols[0].markdown(f"**{r['name']}**")
                    is_mod_now = r['role'] == 'moderator'
                    new_is_mod = cols[1].checkbox("Moderator", value=bool(is_mod_now), key=f"role_{r['user_id']}")
                    if bool(new_is_mod) != bool(is_mod_now):
                        with engine.begin() as conn:
                            conn.execute(
                                update(memberships)
                                .where(and_(memberships.c.user_id == int(r['user_id']), memberships.c.group_id == gid))
                                .values(role="moderator" if new_is_mod else "member")
                            )
                        st.success("Zaktualizowano rolę")

# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    init_db()

    sidebar_auth()

    page = st.sidebar.radio("Nawigacja", ["Grupy", "Panel grupy"], key="nav", label_visibility="collapsed")

    if page == "Grupy":
        page_groups()
    else:
        gid = st.session_state.get("selected_group_id")
        if not gid:
            st.info("Wybierz grupę z listy lub utwórz nową.")
        else:
            page_group_dashboard(int(gid))

if __name__ == "__main__":
    main()
