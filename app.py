# app.py — Futsal Manager (Streamlit + SQLAlchemy Core + Postgres)
# Zasady UX:
# - Zapisy na "Nadchodzące" bez płatności.
# - Płatność dopiero po wydarzeniu (zakładka "Po wydarzeniu"): kwota = cena / liczba zapisanych.
# - Jeśli masz nieopłacone minione wydarzenie w grupie → blokada zapisów na przyszłe.
# - Lista uczestników zawsze widoczna; przy płatnych ✅.
# - Moderator = twórca grupy: może usuwać grupę i generować terminy.

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

# mała pula wystarczy; pool_pre_ping=True wykrywa zryte połączenia
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=0, future=True)
metadata = MetaData()

# ---------------------------
# Tabele (SQLAlchemy Core) — ze schematem
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

def event_is_past(starts_at: pd.Timestamp, duration_minutes: int) -> bool:
    now = pd.Timestamp.now(tz=starts_at.tz) if hasattr(starts_at, "tz") else pd.Timestamp.now()
    return now >= (starts_at + pd.Timedelta(minutes=duration_minutes))

# ---------------------------
# Cache helpers (redukcja zapytań / rerunów)
# ---------------------------

@st.cache_data(ttl=30)
def cached_list_groups_for_user(user_id: int, schema: str) -> pd.DataFrame:
    sql = f"""
    SELECT g.id, g.name, g.city, g.venue, g.weekday, g.start_time, g.price_cents, g.duration_minutes, g.blik_phone,
           CASE WHEN m.role='moderator' THEN 1 ELSE 0 END AS is_mod
    FROM {schema}.groups g
    JOIN {schema}.memberships m ON m.group_id=g.id
    WHERE m.user_id=%(uid)s
    ORDER BY g.city, g.name
    """
    return pd.read_sql_query(sql, engine, params={"uid": int(user_id)})

@st.cache_data(ttl=30)
def cached_get_group(group_id: int, schema: str):
    with engine.begin() as conn:
        return conn.execute(
            select(
                groups.c.id, groups.c.name, groups.c.city, groups.c.venue, groups.c.weekday,
                groups.c.start_time, groups.c.price_cents, groups.c.duration_minutes, groups.c.blik_phone
            ).where(groups.c.id == group_id)
        ).first()

@st.cache_data(ttl=15)
def cached_events_df(group_id: int, schema: str) -> pd.DataFrame:
    base = f"SELECT id, starts_at, price_cents, locked FROM {schema}.events WHERE group_id=%(gid)s ORDER BY starts_at"
    return pd.read_sql_query(base, engine, params={"gid": int(group_id)}, parse_dates=["starts_at"])

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

def cache_clear():
    st.cache_data.clear()

# ---------------------------
# Logika ról i zaległości
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
    """Czy użytkownik ma nieopłacone minione wydarzenie w tej grupie? (wg user_marked_paid=false)"""
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
# Mutacje
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

def delete_group(group_id: int):
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DELETE FROM {DB_SCHEMA}.groups WHERE id=%(g)s", {"g": int(group_id)})

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

# ---------------------------
# Pobrania
# ---------------------------

def get_event(event_id: int):
    with engine.begin() as conn:
        return conn.execute(
            select(events.c.id, events.c.group_id, events.c.starts_at, events.c.price_cents, events.c.locked)
            .where(events.c.id == event_id)
        ).first()

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

# ---------------------------
# UI helpers
# ---------------------------

def upcoming_event_view(event_id: int, uid: int, duration_minutes: int):
    e = get_event(event_id)
    starts = pd.to_datetime(e.starts_at)
    gid = int(e.group_id)

    # zaległości blokują zapisy
    has_debt = user_has_unpaid_past(uid, gid)

    signups_df = cached_signups_with_payments(event_id, DB_SCHEMA)
    is_signed = (not signups_df.empty) and (uid in set(signups_df["user_id"]))

    st.subheader(starts.strftime("%d.%m.%Y %H:%M") + " · Nadchodzące")
    if has_debt:
        st.error("Masz nieopłacone minione wydarzenie w tej grupie. Opłać je w zakładce **Po wydarzeniu**, żeby móc zapisywać się dalej.")

    # działania w formie – 1 rerun
    with st.form(f"up_ev_{event_id}", clear_on_submit=False):
        c1, _ = st.columns([1,3])
        if is_signed:
            withdraw_btn = c1.form_submit_button("Wypisz się")
            if withdraw_btn:
                withdraw(event_id, uid)
                st.success("Wypisano.")
        else:
            signup_btn = c1.form_submit_button("Zapisz się", disabled=has_debt)
            if signup_btn:
                sign_up(event_id, uid)
                st.success("Zapisano na wydarzenie.")
        st.caption("Płatność za to wydarzenie będzie dostępna po jego zakończeniu.")

    st.markdown("**Uczestnicy (zapisani):**")
    if signups_df.empty:
        st.caption("Brak.")
    else:
        # ✅ Poprawka na KeyError: najpierw wybór 'name', potem rename → 'Uczestnik'
        st.dataframe(
            signups_df[["name"]].rename(columns={"name":"Uczestnik"}),
            hide_index=True, use_container_width=True
        )

def past_event_view(event_id: int, uid: int, duration_minutes: int, is_mod: bool, blik_phone: str):
    e = get_event(event_id)
    starts = pd.to_datetime(e.starts_at)
    signups_df = cached_signups_with_payments(event_id, DB_SCHEMA)
    count = 0 if signups_df.empty else len(signups_df)
    per_head = (int(e.price_cents) / 100 / max(1, count)) if count else 0.0

    st.subheader(starts.strftime("%d.%m.%Y %H:%M") + " · Po wydarzeniu")
    st.markdown(f"**Cena hali:** {cents_to_str(int(e.price_cents))} · **Zapisanych:** {count} · **Kwota na osobę:** {per_head:.2f} zł")
    st.markdown(f"**Numer do BLIK / telefon:** {blik_phone}")

    # Własna płatność
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

    # Lista uczestników + fajka przy opłaconych
    st.markdown("**Uczestnicy i płatności:**")
    if signups_df.empty:
        st.caption("Brak.")
    else:
        df = signups_df.copy()
        df["Zapłacone"] = df["user_marked_paid"].astype(bool)
        df["Potwierdzone (mod)"] = df["moderator_confirmed"].astype(bool)
        st.dataframe(
            df.rename(columns={"name":"Uczestnik"})[["Uczestnik","Zapłacone","Potwierdzone (mod)"]],
            hide_index=True, use_container_width=True
        )

    # Moderator: potwierdzenia zbiorcze
    if is_mod and not signups_df.empty:
        st.markdown("### Potwierdzenia moderatora")
        for _, r in signups_df.iterrows():
            cols = st.columns([3,2])
            cols[0].markdown(f"**{r['name']}**")
            with cols[1].form(f"mf_{event_id}_{r['user_id']}"):
                cur_conf = bool(r['moderator_confirmed'])
                new_conf = st.checkbox("Potwierdź (mod)", value=cur_conf, key=f"m_{event_id}_{r['user_id']}")
                save = st.form_submit_button("Zapisz")
                if save and bool(new_conf) != bool(cur_conf):
                    payment_toggle(event_id, int(r['user_id']), 'moderator_confirmed', int(bool(new_conf)))
                    st.success("Zapisano.")

# ---------------------------
# UI — strony
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
            for k in ["user_id","user_name","selected_group_id","selected_event_id","nav","go_panel","go_groups"]:
                st.session_state.pop(k, None)
            st.rerun()

def page_groups():
    st.header("Twoje grupy")
    uid = st.session_state.get("user_id")
    if not uid:
        st.info("Zaloguj się z lewego panelu.")
        return

    # --- Formularz tworzenia ---
    with st.expander("➕ Utwórz nową grupę", expanded=False):
        with st.form("create_group_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            name = col1.text_input("Nazwa grupy")
            city = col1.text_input("Miejscowość")
            venue = col1.text_input("Miejsce wydarzenia (hala)")
            weekday = col2.selectbox("Dzień tygodnia", list(range(7)),
                                     format_func=lambda i: ["Pon","Wt","Śr","Czw","Pt","Sob","Nd"][i])
            start_time = col2.text_input("Godzina startu (HH:MM)", value="21:00")
            duration_minutes = col2.number_input("Czas gry (min)", min_value=30, max_value=240, step=15, value=60)
            price = col2.number_input("Cena za halę (zł)", min_value=0.0, step=1.0)
            blik = col2.text_input("Numer BLIK/telefon do płatności")
            submitted = st.form_submit_button("Utwórz grupę")

        if submitted:
            if not all([name.strip(), city.strip(), venue.strip(), blik.strip()]):
                st.error("Uzupełnij wszystkie pola (w tym numer BLIK).")
            elif ":" not in start_time or len(start_time) != 5:
                st.error("Podaj godzinę w formacie HH:MM (np. 21:00).")
            else:
                try:
                    gid = create_group(
                        name.strip(), city.strip(), venue.strip(),
                        int(weekday), start_time.strip(),
                        int(round(price * 100)), blik.strip(), int(uid), int(duration_minutes)
                    )
                    # generowanie terminów możesz też robić z panelu admina
                    upsert_events_for_group(gid)
                    st.session_state["selected_group_id"] = int(gid)
                    st.session_state["go_panel"] = True
                    st.success("Grupa utworzona. Przechodzę do panelu…")
                    st.rerun()
                except Exception as e:
                    st.error(f"Nie udało się utworzyć grupy: {e}")

    # --- Lista grup ---
    try:
        df = cached_list_groups_for_user(uid, DB_SCHEMA)
    except Exception as e:
        st.error(f"Nie mogę pobrać listy grup: {e}")
        return

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
                st.session_state["go_panel"] = True
                st.rerun()

def page_group_dashboard(group_id: int):
    g = cached_get_group(group_id, DB_SCHEMA)
    if not g:
        st.error("Grupa nie istnieje")
        return
    gid, name, city, venue, weekday, start_time, price_cents, duration_minutes, blik_phone = \
        int(g.id), g.name, g.city, g.venue, int(g.weekday), g.start_time, int(g.price_cents), int(g.duration_minutes), g.blik_phone

    st.header(f"{name} — {city} · {venue}")
    st.caption(f"Gramy: {time_label(weekday, start_time)} · {duration_minutes} min · Cena hali: {cents_to_str(price_cents)} · BLIK: {blik_phone}")

    uid = int(st.session_state.get("user_id"))
    mod = is_moderator(uid, gid)

    # Jedna sekcja naraz (radio zamiast tabs) — mniej renderów = mniej „mrugania”
    section = st.radio("Sekcja", ["Nadchodzące", "Po wydarzeniu", "Statystyki" + (" (admin)" if mod else "")], horizontal=True, label_visibility="collapsed")

    if section.startswith("Nadchodzące"):
        df_all = cached_events_df(gid, DB_SCHEMA)
        if df_all.empty:
            st.info("Brak wydarzeń w kalendarzu")
        else:
            now = pd.Timestamp.now()
            future = df_all[df_all["starts_at"] >= now]
            if future.empty:
                st.caption("Brak nadchodzących wydarzeń.")
            else:
                pick = st.selectbox(
                    "Wybierz termin",
                    list(future["id"]),
                    format_func=lambda i: pd.to_datetime(future.loc[future["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M")
                )
                upcoming_event_view(int(pick), uid, duration_minutes)

    elif section.startswith("Po wydarzeniu"):
        df_all = cached_events_df(gid, DB_SCHEMA)
        if df_all.empty:
            st.info("Brak wydarzeń")
        else:
            now = pd.Timestamp.now()
            past = df_all[df_all["starts_at"] < now]
            if past.empty:
                st.caption("Brak minionych wydarzeń.")
            else:
                pickp = st.selectbox(
                    "Wybierz minione wydarzenie",
                    list(past["id"])[::-1],  # najnowsze pierwsze
                    format_func=lambda i: pd.to_datetime(df_all.loc[df_all["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M")
                )
                past_event_view(int(pickp), uid, duration_minutes, mod, blik_phone)

    else:
        year = st.selectbox("Rok", options=list(range(datetime.now().year, datetime.now().year-5, -1)))
        st.info("Widok statystyk można rozszerzać; teraz najważniejsze są płatności i zapisy.")

        if mod:
            st.markdown("---")
            st.subheader("Narzędzia moderatora")
            if st.button("Wygeneruj 12 kolejnych wydarzeń"):
                upsert_events_for_group(gid, 12)
                st.success("Dodano brakujące wydarzenia w kalendarzu.")

            st.markdown("### Ustawienia grupy")
            with st.expander("🛑 Usuń grupę (nieodwracalne)"):
                st.warning("Usunięcie grupy skasuje **wszystkie** wydarzenia, zapisy, płatności, drużyny i statystyki tej grupy. Tego nie da się cofnąć.")
                confirm_name = st.text_input("Przepisz dokładnie nazwę grupy, aby potwierdzić:", key="del_confirm")
                colA, colB = st.columns([1,3])
                if colA.button("Usuń grupę", type="primary", use_container_width=True):
                    if not mod:
                        st.error("Tylko moderator może usunąć grupę.")
                    elif confirm_name.strip() != name:
                        st.error("Nazwa nie pasuje. Przepisz dokładnie nazwę grupy.")
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
# Main (przekierowania PRZED radiem)
# ---------------------------

def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    init_db()

    sidebar_auth()

    # ustaw docelową zakładkę PRZED renderem widgetu 'radio'
    if st.session_state.get("go_panel"):
        st.session_state["go_panel"] = False
        st.session_state["nav"] = "Panel grupy"
    if st.session_state.get("go_groups"):
        st.session_state["go_groups"] = False
        st.session_state["nav"] = "Grupy"

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
