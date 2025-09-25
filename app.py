# app.py — Sport Manager (Streamlit + SQLAlchemy + Postgres/Neon/Railway/SQLite)
# Auth: klasyczne hasło (hash scrypt), reset hasła via e-mail (Gmail SMTP)
# Telefon nadal wymagany przy rejestracji (ale NIE jest już hasłem)

import os
import re
import hmac
import binascii
import smtplib
import ssl
import secrets
from email.message import EmailMessage
from datetime import datetime, date, timedelta, time as dt_time
from typing import List, Optional, Tuple, Dict

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String,
    DateTime, Boolean, ForeignKey, UniqueConstraint, select,
    insert, update, and_, text
)
from sqlalchemy.engine import Engine
import hashlib

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

SMTP_HOST = _get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_get_secret("SMTP_PORT", "465") or "465")
SMTP_USERNAME = _get_secret("SMTP_USERNAME", "")
SMTP_PASSWORD = _get_secret("SMTP_PASSWORD", "")
SMTP_FROM = _get_secret("SMTP_FROM", SMTP_USERNAME or "no-reply@example.com")
BASE_URL = _get_secret("BASE_URL", "http://localhost:8501")  # do reset linków

# ---------------------------
# DB Engine
# ---------------------------
pg_connect_args = {
    "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 5
} if "postgres" in DATABASE_URL else {}

engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=0,
    future=True,
    connect_args=pg_connect_args
)

# ---------------------------
# Dialect helpers (PG vs SQLite)
# ---------------------------
DIALECT = engine.dialect.name
IS_PG = DIALECT == "postgresql"

def T(table_name: str) -> str:
    return f"{DB_SCHEMA}.{table_name}" if IS_PG else table_name

def FK(table: str, col: str = "id") -> str:
    return f"{DB_SCHEMA}.{table}.{col}" if IS_PG else f"{table}.{col}"

def NOW_SQL() -> str:
    return "NOW()" if IS_PG else "CURRENT_TIMESTAMP"

def build_in_clause(name: str, values: List[str]) -> Tuple[str, Dict[str, str]]:
    keys = [f":{name}{i}" for i in range(len(values))]
    clause = "(" + ", ".join(keys) + ")" if keys else "(NULL)"
    params = {f"{name}{i}": v for i, v in enumerate(values)}
    return clause, params

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
# Walidacje
# ---------------------------
def validate_phone(phone: str) -> bool:
    if not phone:
        return False
    p = phone.strip().replace(" ", "").replace("-", "")
    return bool(re.fullmatch(r"\+?\d{9,15}", p))

def validate_email(email: str) -> bool:
    if not email:
        return False
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip(), re.IGNORECASE))

def validate_password_strength(pw: str) -> Optional[str]:
    if len(pw) < 10:
        return "Hasło powinno mieć min. 10 znaków."
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return "Hasło powinno zawierać litery i cyfry."
    return None

# ---------------------------
# Hashowanie haseł (scrypt)
# ---------------------------
SCRYPT_N = 2**14  # 16384
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_LEN = 16
SCRYPT_KEY_LEN = 32

def hash_password(password: str) -> Tuple[str, str, str]:
    salt = os.urandom(SCRYPT_SALT_LEN)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_KEY_LEN)
    return (
        binascii.hexlify(salt).decode(),
        binascii.hexlify(key).decode(),
        f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${SCRYPT_KEY_LEN}",
    )

def verify_password(password: str, salt_hex: str, key_hex: str, meta: str) -> bool:
    try:
        _algo, n, r, p, dk = meta.split("$")
        n, r, p, dk = int(n), int(r), int(p), int(dk)
    except Exception:
        # fallback do naszych stałych, jeśli meta brak/zepsute
        n, r, p, dk = SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_KEY_LEN
    salt = binascii.unhexlify(salt_hex.encode())
    expected = binascii.unhexlify(key_hex.encode())
    calc = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dk)
    return hmac.compare_digest(calc, expected)

# ---------------------------
# Tabele
# ---------------------------
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("email", String(255), nullable=False),
    Column("phone", String(64), nullable=False),
    Column("pwd_salt", String(255)),
    Column("pwd_hash", String(255)),
    Column("pwd_meta", String(255)),
    Column("is_admin", Boolean, nullable=False, server_default=text("false") if IS_PG else text("0")),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA if IS_PG else None,
)

# tokeny do resetu hasła
password_resets = Table(
    "password_resets", metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey(FK("users"), ondelete="CASCADE"), nullable=False),
    Column("token", String(255), nullable=False, unique=True),
    Column("expires_at", DateTime, nullable=False),
    Column("used", Boolean, nullable=False, server_default=text("false") if IS_PG else text("0")),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA if IS_PG else None,
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
    Column("blik_phone", String(64), nullable=False, server_default=text("''")),
    Column("sport", String(64), nullable=False, server_default=text("'Piłka nożna (Hala)'") if IS_PG else text("Piłka nożna (Hala)")),
    Column("postal_code", String(16), nullable=True),
    Column("created_by", Integer, ForeignKey(FK("users"), ondelete="SET NULL"), nullable=False),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA if IS_PG else None,
)

memberships = Table(
    "memberships", metadata,
    Column("user_id", Integer, ForeignKey(FK("users"), ondelete="CASCADE"), primary_key=True),
    Column("group_id", Integer, ForeignKey(FK("groups"), ondelete="CASCADE"), primary_key=True),
    Column("role", String(16), nullable=False, server_default=text("'member'") if IS_PG else text("member")),
    schema=DB_SCHEMA if IS_PG else None,
)

events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True),
    Column("group_id", Integer, ForeignKey(FK("groups"), ondelete="CASCADE"), nullable=False),
    Column("starts_at", DateTime, nullable=False),
    Column("price_cents", Integer, nullable=False),
    Column("generated", Boolean, nullable=False, server_default=text("true") if IS_PG else text("1")),
    Column("locked", Boolean, nullable=False, server_default=text("false") if IS_PG else text("0")),
    Column("name", String(255), nullable=True),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA if IS_PG else None,
)

event_signups = Table(
    "event_signups", metadata,
    Column("event_id", Integer, ForeignKey(FK("events"), ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey(FK("users"), ondelete="CASCADE"), primary_key=True),
    Column("signed_at", DateTime, nullable=False),
    schema=DB_SCHEMA if IS_PG else None,
)

payments = Table(
    "payments", metadata,
    Column("event_id", Integer, ForeignKey(FK("events"), ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey(FK("users"), ondelete="CASCADE"), primary_key=True),
    Column("user_marked_paid", Boolean, nullable=False, server_default=text("false") if IS_PG else text("0")),
    Column("moderator_confirmed", Boolean, nullable=False, server_default=text("false") if IS_PG else text("0")),
    schema=DB_SCHEMA if IS_PG else None,
)

teams = Table(
    "teams", metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey(FK("events"), ondelete="CASCADE"), nullable=False),
    Column("name", String(255), nullable=False),
    Column("idx", Integer, nullable=False),
    Column("goals", Integer, nullable=False, server_default=text("0")),
    UniqueConstraint("event_id", "idx", name="uq_teams_event_idx"),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA if IS_PG else None,
)

team_members = Table(
    "team_members", metadata,
    Column("team_id", Integer, ForeignKey(FK("teams"), ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey(FK("users"), ondelete="CASCADE"), primary_key=True),
    schema=DB_SCHEMA if IS_PG else None,
)

goals = Table(
    "goals", metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey(FK("events"), ondelete="CASCADE"), nullable=False),
    Column("scorer_id", Integer, ForeignKey(FK("users"), ondelete="SET NULL"), nullable=False),
    Column("assist_id", Integer, ForeignKey(FK("users"), ondelete="SET NULL")),
    Column("minute", Integer),
    sqlite_autoincrement=True,
    schema=DB_SCHEMA if IS_PG else None,
)

# ---------------------------
# Inicjalizacja / migracje
# ---------------------------
def init_db():
    metadata.create_all(engine)
    with engine.begin() as conn:
        # users: kolumny + unikalność email
        conn.execute(text(f"ALTER TABLE {T('users')} ADD COLUMN IF NOT EXISTS email TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {T('users')} ADD COLUMN IF NOT EXISTS phone TEXT NOT NULL DEFAULT ''"))
        conn.execute(text(f"ALTER TABLE {T('users')} ADD COLUMN IF NOT EXISTS pwd_salt TEXT"))
        conn.execute(text(f"ALTER TABLE {T('users')} ADD COLUMN IF NOT EXISTS pwd_hash TEXT"))
        conn.execute(text(f"ALTER TABLE {T('users')} ADD COLUMN IF NOT EXISTS pwd_meta TEXT"))
        # indeks/unikalność email
        if IS_PG:
            conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON {T('users')} (email)"))
        else:
            conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON {T('users')}(email)"))

        # reszta migracji
        conn.execute(text(f"ALTER TABLE {T('groups')} ADD COLUMN IF NOT EXISTS duration_minutes INTEGER NOT NULL DEFAULT 60;"))
        conn.execute(text(f"ALTER TABLE {T('groups')} ADD COLUMN IF NOT EXISTS blik_phone TEXT NOT NULL DEFAULT '';"))
        conn.execute(text(f"ALTER TABLE {T('groups')} ADD COLUMN IF NOT EXISTS sport TEXT NOT NULL DEFAULT 'Piłka nożna (Hala)';"))
        conn.execute(text(f"ALTER TABLE {T('groups')} ADD COLUMN IF NOT EXISTS postal_code TEXT;"))
        conn.execute(text(f"ALTER TABLE {T('memberships')} ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member';"))
        conn.execute(text(f"ALTER TABLE {T('events')} ADD COLUMN IF NOT EXISTS name TEXT;"))

        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_events_group_starts ON {T('events')} (group_id, starts_at);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_signups_event ON {T('event_signups')} (event_id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_payments_event_user ON {T('payments')} (event_id, user_id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_goals_event_scorer ON {T('goals')} (event_id, scorer_id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_goals_event_assist ON {T('goals')} (event_id, assist_id);"))

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
# E-mail
# ---------------------------
def send_email(to_email: str, subject: str, html_body: str, text_body: Optional[str] = None):
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        raise RuntimeError("Brak konfiguracji SMTP w st.secrets (SMTP_USERNAME / SMTP_PASSWORD).")
    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    if text_body:
        msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

def create_reset_token_for_user(user_id: int, minutes_valid: int = 15) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=minutes_valid)
    with engine.begin() as conn:
        conn.execute(
            insert(password_resets).values(
                user_id=int(user_id), token=token, expires_at=expires, used=False
            )
        )
    return token

def consume_reset_token(token: str) -> Optional[int]:
    with engine.begin() as conn:
        row = conn.execute(
            select(password_resets.c.id, password_resets.c.user_id, password_resets.c.expires_at, password_resets.c.used)
            .where(password_resets.c.token == token)
        ).first()
        if not row:
            return None
        if bool(row.used):
            return None
        if row.expires_at < datetime.utcnow():
            return None
        # oznacz jako użyty
        conn.execute(
            update(password_resets).where(password_resets.c.id == int(row.id)).values(used=True)
        )
        return int(row.user_id)

# ---------------------------
# Cache helpers (SQL safe with text())
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
    FROM {T('groups')} g
    JOIN {T('memberships')} m ON m.group_id=g.id
    WHERE m.user_id = :uid
    """
    params: Dict[str, object] = {"uid": int(user_id)}

    if activity_type == "Sporty drużynowe":
        clause, ps = build_in_clause("ts", TEAM_SPORTS)
        sql += f" AND g.sport IN {clause}"
        params.update(ps)
    elif activity_type == "Zajęcia fitness":
        clause, ps = build_in_clause("fs", FITNESS_CLASSES)
        sql += f" AND g.sport IN {clause}"
        params.update(ps)

    if activity_type == "Sporty drużynowe" and discipline and discipline != "Wszystkie":
        sql += " AND g.sport = :sp"
        params["sp"] = discipline

    if city_filter:
        sql += " AND LOWER(g.city) LIKE :city"
        params["city"] = f"%{city_filter.lower()}%"

    if postal_filter:
        sql += " AND LOWER(COALESCE(g.postal_code,'')) LIKE :pc"
        params["pc"] = f"%{postal_filter.lower()}%"

    sql += " ORDER BY g.city, g.name"
    return pd.read_sql_query(text(sql), engine, params=params)

@st.cache_data(ttl=30)
def cached_all_groups(uid: int, schema: str,
                      activity_type: Optional[str],
                      discipline: Optional[str],
                      city_filter: str,
                      postal_filter: str) -> pd.DataFrame:
    sql = f"""
    SELECT
        g.id, g.name, g.city, g.venue, g.weekday, g.start_time, g.price_cents,
        g.duration_minutes, g.blik_phone, g.sport, g.postal_code,
        CASE WHEN m.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_member
    FROM {T('groups')} g
    LEFT JOIN {T('memberships')} m
      ON m.group_id = g.id AND m.user_id = :u
    WHERE 1=1
    """
    params: Dict[str, object] = {"u": int(uid)}

    if activity_type == "Sporty drużynowe":
        clause, ps = build_in_clause("ts", TEAM_SPORTS)
        sql += f" AND g.sport IN {clause}"
        params.update(ps)
    elif activity_type == "Zajęcia fitness":
        clause, ps = build_in_clause("fs", FITNESS_CLASSES)
        sql += f" AND g.sport IN {clause}"
        params.update(ps)

    if activity_type == "Sporty drużynowe" and discipline and discipline != "Wszystkie":
        sql += " AND g.sport = :sp"
        params["sp"] = discipline

    if city_filter:
        sql += " AND LOWER(g.city) LIKE :city"
        params["city"] = f"%{city_filter.lower()}%"

    if postal_filter:
        sql += " AND LOWER(COALESCE(g.postal_code,'')) LIKE :pc"
        params["pc"] = f"%{postal_filter.lower()}%"

    sql += " ORDER BY g.city, g.name"
    return pd.read_sql_query(text(sql), engine, params=params)

@st.cache_data(ttl=20)
def cached_events_df(group_id: int, schema: str) -> pd.DataFrame:
    base = f"SELECT id, starts_at, price_cents, locked, name FROM {T('events')} WHERE group_id=:gid ORDER BY starts_at"
    return pd.read_sql_query(text(base), engine, params={"gid": int(group_id)}, parse_dates=["starts_at"])

@st.cache_data(ttl=20)
def cached_signups(event_id: int, schema: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text(
        f"""
        SELECT es.user_id, u.name
        FROM {T('event_signups')} es
        JOIN {T('users')} u ON u.id=es.user_id
        WHERE es.event_id=:eid
        ORDER BY u.name
        """),
        engine, params={"eid": int(event_id)}
    )

@st.cache_data(ttl=20)
def cached_signups_with_payments(event_id: int, schema: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text(
        f"""
        SELECT es.user_id, u.name,
               COALESCE(p.user_marked_paid, 0) AS user_marked_paid,
               COALESCE(p.moderator_confirmed, 0) AS moderator_confirmed
        FROM {T('event_signups')} es
        JOIN {T('users')} u ON u.id=es.user_id
        LEFT JOIN {T('payments')} p ON p.event_id=es.event_id AND p.user_id=es.user_id
        WHERE es.event_id=:eid
        ORDER BY u.name
        """),
        engine, params={"eid": int(event_id)}
    )

@st.cache_data(ttl=20)
def cached_event_goals(event_id: int, schema: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text(
        f"""
        SELECT g.id, g.event_id, g.scorer_id, s.name AS scorer_name,
               g.assist_id, a.name AS assist_name,
               g.minute
        FROM {T('goals')} g
        LEFT JOIN {T('users')} s ON s.id=g.scorer_id
        LEFT JOIN {T('users')} a ON a.id=g.assist_id
        WHERE g.event_id=:eid
        ORDER BY COALESCE(g.minute,9999), g.id
        """),
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
          FROM {T('event_signups')} es
          JOIN {T('events')} e ON e.id=es.event_id
          LEFT JOIN {T('payments')} p ON p.event_id=es.event_id AND p.user_id=es.user_id
          WHERE es.user_id=:u
            AND e.group_id=:g
            AND e.starts_at < {NOW_SQL()}
            AND COALESCE(p.user_marked_paid, 0) = 0
        ) AS has_debt
        """
        return bool(conn.execute(text(sql), {"u": int(user_id), "g": int(group_id)}).scalar_one())

# ---------------------------
# Mutacje
# ---------------------------
def ensure_user_with_password(name: str, email: str, phone: str, password: str) -> int:
    if not validate_email(email):
        raise ValueError("Podaj prawidłowy e-mail.")
    if not validate_phone(phone):
        raise ValueError("Telefon jest wymagany i musi mieć prawidłowy format (+ i 9–15 cyfr).")
    weak = validate_password_strength(password)
    if weak:
        raise ValueError(weak)

    phone_norm = phone.strip().replace(" ", "").replace("-", "")
    salt_hex, key_hex, meta = hash_password(password)

    with engine.begin() as conn:
        # czy e-mail istnieje?
        exists = conn.execute(select(users.c.id).where(users.c.email == email.strip().lower())).first()
        if exists:
            raise ValueError("Konto z tym adresem e-mail już istnieje.")
        res = conn.execute(
            insert(users).values(
                name=name.strip(),
                email=email.strip().lower(),
                phone=phone_norm,
                pwd_salt=salt_hex,
                pwd_hash=key_hex,
                pwd_meta=meta
            )
        )
        uid = int(res.inserted_primary_key[0])
        return uid

def update_user_password(user_id: int, new_password: str):
    weak = validate_password_strength(new_password)
    if weak:
        raise ValueError(weak)
    salt_hex, key_hex, meta = hash_password(new_password)
    with engine.begin() as conn:
        conn.execute(
            update(users).where(users.c.id == int(user_id)).values(
                pwd_salt=salt_hex, pwd_hash=key_hex, pwd_meta=meta
            )
        )

def _insert_membership(conn, user_id: int, group_id: int, role: str):
    if IS_PG:
        conn.execute(
            text(
            f"""
            INSERT INTO {T('memberships')} (user_id, group_id, role)
            VALUES (:u, :g, :r)
            ON CONFLICT (user_id, group_id) DO NOTHING;
            """),
            {"u": int(user_id), "g": int(group_id), "r": role},
        )
    else:
        conn.execute(
            text(
            f"""
            INSERT OR IGNORE INTO {T('memberships')} (user_id, group_id, role)
            VALUES (:u, :g, :r);
            """),
            {"u": int(user_id), "g": int(group_id), "r": role},
        )

def join_group(user_id: int, group_id: int):
    with engine.begin() as conn:
        _insert_membership(conn, int(user_id), int(group_id), "member")

def create_group(name: str, city: str, venue: str, weekday: int, start_time: str,
                 price_cents: int, blik_phone: str, created_by: int, duration_minutes: int = 60,
                 sport: str = "Piłka nożna (Hala)", postal_code: str = "") -> int:
    with engine.begin() as conn:
        res = conn.execute(
            insert(groups).values(
                name=name, city=city, venue=venue, weekday=weekday, start_time=start_time,
                price_cents=price_cents, duration_minutes=duration_minutes, blik_phone=blik_phone,
                sport=sport, created_by=created_by, postal_code=postal_code or None
            )
        )
        gid = int(res.inserted_primary_key[0])
        _insert_membership(conn, int(created_by), int(gid), "moderator")
        return gid

def delete_group(group_id: int):
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {T('groups')} WHERE id=:g"), {"g": int(group_id)})

def create_recurring_events(group_id: int, weekday: int, base_price_cents: int,
                            slots: List[Tuple[str, Optional[str]]], weeks_ahead: int = 12):
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
        if IS_PG:
            conn.execute(
                text(
                f"""
                INSERT INTO {T('event_signups')} (event_id, user_id, signed_at)
                VALUES (:e, :u, :t)
                ON CONFLICT (event_id, user_id) DO NOTHING;
                """),
                {"e": int(event_id), "u": int(user_id), "t": now},
            )
        else:
            conn.execute(
                text(
                f"""
                INSERT OR IGNORE INTO {T('event_signups')} (event_id, user_id, signed_at)
                VALUES (:e, :u, :t);
                """),
                {"e": int(event_id), "u": int(user_id), "t": now},
            )

        conn.execute(
            text(
            f"""
            INSERT INTO {T('payments')} (event_id, user_id, user_marked_paid, moderator_confirmed)
            SELECT :e, :u, 0, 0
            WHERE NOT EXISTS (
               SELECT 1 FROM {T('payments')} WHERE event_id=:e AND user_id=:u
            );
            """),
            {"e": int(event_id), "u": int(user_id)},
        )

        conn.execute(
            text(
            f"""
            INSERT OR IGNORE INTO {T('memberships')} (user_id, group_id, role)
            SELECT :u, e.group_id, 'member'
            FROM {T('events')} e
            WHERE e.id=:e;
            """),
            {"u": int(user_id), "e": int(event_id)},
        )

def withdraw(event_id: int, user_id: int):
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {T('payments')} WHERE event_id=:e AND user_id=:u"),
                     {"e": int(event_id), "u": int(user_id)})
        conn.execute(text(f"DELETE FROM {T('event_signups')} WHERE event_id=:e AND user_id=:u"),
                     {"e": int(event_id), "u": int(user_id)})

def payment_toggle(event_id: int, user_id: int, field: str, value: int):
    if field not in ("user_marked_paid", "moderator_confirmed"):
        return
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE {T('payments')} SET {field}=:v WHERE event_id=:e AND user_id=:u"),
                     {"v": bool(value), "e": int(event_id), "u": int(user_id)})

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
        conn.execute(text(f"DELETE FROM {T('goals')} WHERE id=:g"), {"g": int(goal_id)})

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
    with engine.begin() as _c:
        grp = _c.execute(select(groups.c.sport).where(groups.c.id == group_id)).first()
    team_mode = bool(grp and is_team_sport(grp.sport))

    if show_pay:
        df = pd.read_sql_query(
            text(
            f"""
            SELECT es.user_id, u.name,
                   COALESCE(p.user_marked_paid, 0) AS user_marked_paid,
                   COALESCE(p.moderator_confirmed, 0) AS moderator_confirmed,
                   COALESCE(SUM(CASE WHEN g.scorer_id=es.user_id THEN 1 ELSE 0 END),0) AS goals,
                   COALESCE(SUM(CASE WHEN g.assist_id=es.user_id THEN 1 ELSE 0 END),0) AS assists
            FROM {T('event_signups')} es
            JOIN {T('users')} u ON u.id=es.user_id
            LEFT JOIN {T('payments')} p ON p.event_id=es.event_id AND p.user_id=es.user_id
            LEFT JOIN {T('goals')} g ON g.event_id=es.event_id
            WHERE es.event_id=:eid
            GROUP BY es.user_id,u.name,p.user_marked_paid,p.moderator_confirmed
            ORDER BY user_marked_paid DESC, u.name
            """),
            engine, params={"eid": int(event_id)}
        )
    else:
        signups_df = cached_signups(event_id, DB_SCHEMA)
        if signups_df.empty:
            st.caption("Brak zapisanych.")
            return

        e = get_event(event_id)
        year = pd.to_datetime(e.starts_at).year

        if IS_PG:
            stats_sql = f"""
            SELECT u.id AS user_id,
                   u.name AS name_stat,
                   COALESCE(SUM(CASE WHEN g.scorer_id=u.id THEN 1 ELSE 0 END),0) AS goals,
                   COALESCE(SUM(CASE WHEN g.assist_id=u.id THEN 1 ELSE 0 END),0) AS assists
            FROM {T('users')} u
            JOIN {T('memberships')} m ON m.user_id=u.id AND m.group_id=:gid
            LEFT JOIN {T('events')} e ON e.group_id=m.group_id
            LEFT JOIN {T('goals')} g ON g.event_id=e.id
            WHERE e.id IS NOT NULL
              AND EXTRACT(YEAR FROM e.starts_at)=:yr
            GROUP BY u.id, u.name
            """
        else:
            stats_sql = f"""
            SELECT u.id AS user_id,
                   u.name AS name_stat,
                   COALESCE(SUM(CASE WHEN g.scorer_id=u.id THEN 1 ELSE 0 END),0) AS goals,
                   COALESCE(SUM(CASE WHEN g.assist_id=u.id THEN 1 ELSE 0 END),0) AS assists
            FROM {T('users')} u
            JOIN {T('memberships')} m ON m.user_id=u.id AND m.group_id=:gid
            LEFT JOIN {T('events')} e ON e.group_id=m.group_id
            LEFT JOIN {T('goals')} g ON g.event_id=e.id
            WHERE e.id IS NOT NULL
              AND CAST(strftime('%Y', e.starts_at) AS INTEGER)=:yr
            GROUP BY u.id, u.name
            """
        stats = pd.read_sql_query(text(stats_sql), engine, params={"gid": int(e.group_id), "yr": int(year)})

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
def get_event(event_id: int):
    with engine.begin() as conn:
        return conn.execute(
            select(events.c.id, events.c.group_id, events.c.starts_at, events.c.price_cents, events.c.locked, events.c.name)
            .where(events.c.id == event_id)
        ).first()

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

        # gole/asysty (jak było) ...

# ---------------------------
# AUTH UI (rejestracja, logowanie, reset)
# ---------------------------
def _rate_limit_ok() -> bool:
    key = "login_attempts"
    now = datetime.utcnow().timestamp()
    attempts = st.session_state.get(key, [])
    # zostaw tylko ostatnie 10 min
    attempts = [t for t in attempts if now - t < 600]
    st.session_state[key] = attempts
    if len(attempts) >= 8:
        return False
    return True

def _bump_attempt():
    key = "login_attempts"
    now = datetime.utcnow().timestamp()
    st.session_state.setdefault(key, []).append(now)

def sidebar_auth_and_filters():
    st.sidebar.header("Panel")

    # tryb resetu hasła z linka (query param)
    qp = st.query_params
    reset_token = None
    if "reset" in qp:
        vals = qp.get("reset")
        if isinstance(vals, list):
            reset_token = vals[0]
        else:
            reset_token = vals

    if reset_token:
        st.sidebar.subheader("Ustaw nowe hasło")
        new_pw = st.sidebar.text_input("Nowe hasło", type="password")
        new_pw2 = st.sidebar.text_input("Powtórz hasło", type="password")
        change = st.sidebar.button("Zmień hasło")
        if change:
            if new_pw != new_pw2:
                st.sidebar.error("Hasła nie są takie same.")
            else:
                try:
                    uid = consume_reset_token(reset_token)
                    if not uid:
                        st.sidebar.error("Link jest nieprawidłowy lub wygasł.")
                    else:
                        update_user_password(uid, new_pw)
                        st.sidebar.success("Hasło zmienione. Zaloguj się.")
                        # usuń reset z URL
                        st.query_params.clear()
                except Exception as e:
                    st.sidebar.error(f"Nie udało się zmienić hasła: {e}")
        st.sidebar.markdown("---")

    # Logowanie
    st.sidebar.subheader("Logowanie")
    email_login = st.sidebar.text_input("E-mail")
    pw_login = st.sidebar.text_input("Hasło", type="password")
    do_login = st.sidebar.button("Zaloguj")

    if do_login:
        if not _rate_limit_ok():
            st.sidebar.error("Zbyt wiele prób. Spróbuj ponownie za kilka minut.")
        else:
            with engine.begin() as conn:
                row = conn.execute(
                    select(users.c.id, users.c.name, users.c.email, users.c.phone, users.c.pwd_salt, users.c.pwd_hash, users.c.pwd_meta)
                    .where(users.c.email == (email_login or "").strip().lower())
                ).first()
            if not row:
                _bump_attempt()
                st.sidebar.error("Błędny e-mail lub hasło.")
            else:
                ok = verify_password(pw_login or "", row.pwd_salt or "", row.pwd_hash or "", row.pwd_meta or "")
                if not ok:
                    _bump_attempt()
                    st.sidebar.error("Błędny e-mail lub hasło.")
                else:
                    st.session_state["user_id"] = int(row.id)
                    st.session_state["user_name"] = row.name
                    st.session_state["user_email"] = row.email
                    st.sidebar.success(f"Witaj, {row.name}!")

    # Reset hasła (wysyłka linku)
    with st.sidebar.expander("Nie pamiętam hasła"):
        reset_email = st.text_input("Twój e-mail", key="reset_email")
        if st.button("Wyślij link resetu"):
            try:
                with engine.begin() as conn:
                    u = conn.execute(select(users.c.id, users.c.email, users.c.name).where(users.c.email == (reset_email or "").strip().lower())).first()
                if not u:
                    st.warning("Jeśli adres istnieje w systemie, wyślemy e-mail resetu.")  # nie zdradzaj istnienia konta
                else:
                    token = create_reset_token_for_user(int(u.id), minutes_valid=15)
                    link = f"{BASE_URL}?reset={token}"
                    html = f"""
                    <p>Cześć {u.name},</p>
                    <p>Otrzymaliśmy prośbę o reset hasła do Sport Manager.</p>
                    <p><a href="{link}">Kliknij tutaj, aby ustawić nowe hasło</a> (link ważny 15 minut).</p>
                    <p>Jeśli to nie Ty, zignoruj tę wiadomość.</p>
                    """
                    send_email(u.email, "Reset hasła — Sport Manager", html, text_body=f"Otwórz link (ważny 15 min): {link}")
                    st.success("Jeśli adres istnieje, wysłaliśmy link resetu.")
            except Exception as e:
                st.error(f"Nie udało się wysłać maila: {e}")

    st.sidebar.markdown("---")

    # Rejestracja (telefon wymagany, e-mail unikalny, hasło)
    st.sidebar.subheader("Rejestracja")
    reg_name = st.sidebar.text_input("Imię / nick", key="reg_name")
    reg_email = st.sidebar.text_input("E-mail", key="reg_email")
    reg_phone = st.sidebar.text_input("Telefon (wymagany)", key="reg_phone")
    reg_pw = st.sidebar.text_input("Hasło", type="password", key="reg_pw")
    reg_pw2 = st.sidebar.text_input("Powtórz hasło", type="password", key="reg_pw2")
    do_reg = st.sidebar.button("Utwórz konto")

    if do_reg:
        if reg_pw != reg_pw2:
            st.sidebar.error("Hasła nie są takie same.")
        else:
            try:
                uid = ensure_user_with_password(reg_name, reg_email, reg_phone, reg_pw)
                st.session_state["user_id"] = uid
                st.session_state["user_name"] = reg_name.strip()
                st.session_state["user_email"] = reg_email.strip().lower()
                st.sidebar.success("Konto utworzone i zalogowano!")
            except Exception as e:
                st.sidebar.error(str(e))

    st.sidebar.markdown("---")

    # Filtry (bez fitnesowych dyscyplin — jak ustalaliśmy)
    activity_type = st.sidebar.selectbox(
        "Typ aktywności",
        ["Wszystkie", "Sporty drużynowe", "Zajęcia fitness"],
        index=0
    )
    if activity_type == "Sporty drużynowe":
        discipline = st.sidebar.selectbox("Dyscyplina", ["Wszystkie"] + TEAM_SPORTS, index=0)
    else:
        discipline = "Wszystkie"

    city_filter = st.sidebar.text_input("Miejscowość (filtr)", value="")
    postal_filter = st.sidebar.text_input("Kod pocztowy (filtr)", value="")

    st.session_state["activity_type"] = activity_type
    st.session_state["discipline"] = discipline
    st.session_state["city_filter"] = city_filter.strip()
    st.session_state["postal_filter"] = postal_filter.strip()

    if "user_id" in st.session_state:
        st.sidebar.info(f"Zalogowano jako: {st.session_state.get('user_name','')} ({st.session_state.get('user_email','')})")
        if st.sidebar.button("Wyloguj"):
            for k in ["user_id","user_name","user_email","selected_group_id","selected_event_id","nav","go_panel","go_groups",
                      "activity_type","discipline","city_filter","postal_filter","login_attempts"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.sidebar.caption("Zaloguj się, aby zapisywać się i zarządzać wydarzeniami.")

# ---------------------------
# Strony (jak wcześniej)
# ---------------------------
def page_groups():
    st.header("Grupy")

    uid = st.session_state.get("user_id")
    activity_type = st.session_state.get("activity_type", "Wszystkie")
    discipline = st.session_state.get("discipline", "Wszystkie")
    city_filter = st.session_state.get("city_filter", "")
    postal_filter = st.session_state.get("postal_filter", "")

    # Twoje grupy
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

    # Tworzenie grupy — layout jak ustalaliśmy (3x3 + kolejne sekcje od lewej)
    st.markdown("---")
    with st.expander("➕ Utwórz nową grupę", expanded=False):
        with st.form("create_group_form", clear_on_submit=False):
            st.markdown("### Dane grupy")

            r1c1, r1c2, r1c3 = st.columns(3)
            name = r1c1.text_input("Nazwa grupy")
            city = r1c2.text_input("Miejscowość")
            postal_code = r1c3.text_input("Kod pocztowy (np. 00-001)")

            r2c1, r2c2, r2c3 = st.columns(3)
            venue = r2c1.text_input("Miejsce wydarzenia (hala/boisko/plaża)")
            weekday = r2c2.selectbox("Dzień tygodnia", list(range(7)),
                                     format_func=lambda i: ["Pon","Wt","Śr","Czw","Pt","Sob","Nd"][i])
            start_time = r2c3.text_input("Godzina bazowa (HH:MM)", value="21:00")

            r3c1, r3c2, r3c3 = st.columns(3)
            duration_minutes = r3c1.number_input("Czas gry / zajęć (min)", min_value=30, max_value=240, step=15, value=60)
            price = r3c2.number_input("Cena za obiekt/zajęcia (zł)", min_value=0.0, step=1.0)
            blik = r3c3.text_input("Numer BLIK/telefon do płatności")

            st.markdown("### Typ aktywności")
            r4c1, r4c2, r4c3 = st.columns(3)
            activity_type_f = r4c1.selectbox("Typ aktywności", ["Sporty drużynowe", "Zajęcia fitness"])
            if activity_type_f == "Sporty drużynowe":
                sport_sel = r4c2.selectbox("Dyscyplina", TEAM_SPORTS, index=0)
            else:
                sport_sel = r4c2.selectbox("Zajęcia", FITNESS_CLASSES, index=0)

            st.markdown("### Dodatkowe sloty (opcjonalnie)")
            st.caption("Po jednej linii: `HH:MM;Nazwa`. Przykład: `09:00;Pilates`")
            r5c1, r5c2, r5c3 = st.columns(3)
            extra_raw = r5c1.text_area("Lista slotów (godzina;nazwa)", height=120, key="extra_slots")

            submitted = st.form_submit_button("Utwórz grupę")

        if submitted:
            if "user_id" not in st.session_state:
                st.error("Zaloguj się, aby tworzyć grupy.")
            elif not all([name.strip(), city.strip(), venue.strip(), blik.strip()]):
                st.error("Uzupełnij wszystkie pola (w tym numer BLIK).")
            elif ":" not in start_time or len(start_time) != 5:
                st.error("Podaj **godzinę bazową** w formacie HH:MM (np. 21:00).")
            else:
                slots: List[Tuple[str, Optional[str]]] = [(start_time.strip(), None)]
                if extra_raw and extra_raw.strip():
                    for line in extra_raw.strip().splitlines():
                        if ";" in line:
                            hhmm, nm = line.split(";", 1)
                            hhmm = hhmm.strip()
                            nm = nm.strip()
                            if len(hhmm) == 5 and ":" in hhmm and nm:
                                slots.append((hhmm, nm))
                        else:
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
            return
        page_group_dashboard(int(gid))

if __name__ == "__main__":
    main()
