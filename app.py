import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta, time as dt_time
from typing import Optional, List, Tuple

DB_PATH = "futsal.db"
APP_TITLE = "Futsal Manager"

# ---------------------------
# DB helpers
# ---------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Users & Groups
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            is_admin INTEGER DEFAULT 0
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            venue TEXT NOT NULL,
            weekday INTEGER NOT NULL,    -- 0=Monday ... 6=Sunday
            start_time TEXT NOT NULL,    -- "HH:MM"
            price_cents INTEGER NOT NULL,
            blik_phone TEXT NOT NULL,
            created_by INTEGER NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memberships (
            user_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            role TEXT NOT NULL DEFAULT 'member', -- member | moderator
            PRIMARY KEY (user_id, group_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
        );
        """
    )

    # Events & Participation & Payments
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            starts_at TEXT NOT NULL,  -- ISO datetime
            price_cents INTEGER NOT NULL,
            generated INTEGER DEFAULT 1,
            locked INTEGER DEFAULT 0,  -- when finished
            FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_signups (
            event_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            signed_at TEXT NOT NULL,
            PRIMARY KEY (event_id, user_id),
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS payments (
            event_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            user_marked_paid INTEGER DEFAULT 0,
            moderator_confirmed INTEGER DEFAULT 0,
            PRIMARY KEY (event_id, user_id),
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    # Teams & Results
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            idx INTEGER NOT NULL, -- 1..3 order
            goals INTEGER DEFAULT 0,
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            PRIMARY KEY (team_id, user_id),
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            scorer_id INTEGER NOT NULL,
            assist_id INTEGER,
            team_id INTEGER,
            minute INTEGER,
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
            FOREIGN KEY (scorer_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (assist_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL
        );
        """
    )

    conn.commit()
    conn.close()


# ---------------------------
# Utilities
# ---------------------------

def cents_to_str(cents: int) -> str:
    return f"{cents/100:.2f} zł"


def time_label(weekday: int, hhmm: str) -> str:
    days = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Nd"]
    return f"{days[weekday]} {hhmm}"


def next_dates_for_weekday(start_from: date, weekday: int, count: int) -> List[date]:
    # find next occurrence of weekday (including today)
    days_ahead = (weekday - start_from.weekday()) % 7
    first = start_from + timedelta(days=days_ahead)
    return [first + timedelta(days=7*i) for i in range(count)]


def is_moderator(user_id: int, group_id: int) -> bool:
    conn = get_conn()
    r = conn.execute(
        "SELECT 1 FROM memberships WHERE user_id=? AND group_id=? AND role='moderator'", (user_id, group_id)
    ).fetchone()
    conn.close()
    return r is not None


# ---------------------------
# Auth (very lightweight)
# ---------------------------

def ensure_user(name: str, phone: str = "", email: str = "") -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name=? AND IFNULL(email,'')=IFNULL(?, '')", (name, email))
    row = cur.fetchone()
    if row:
        uid = row[0]
        cur.execute("UPDATE users SET phone=COALESCE(?, phone) WHERE id=?", (phone or None, uid))
    else:
        cur.execute("INSERT INTO users(name, phone, email) VALUES(?,?,?)", (name, phone, email))
        uid = cur.lastrowid
    conn.commit()
    conn.close()
    return uid


# ---------------------------
# Data access
# ---------------------------

def create_group(name: str, city: str, venue: str, weekday: int, start_time: str, price_cents: int, blik_phone: str, created_by: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO groups(name, city, venue, weekday, start_time, price_cents, blik_phone, created_by) VALUES(?,?,?,?,?,?,?,?)",
        (name, city, venue, weekday, start_time, price_cents, blik_phone, created_by),
    )
    gid = cur.lastrowid
    cur.execute("INSERT INTO memberships(user_id, group_id, role) VALUES(?,?, 'moderator')", (created_by, gid))
    conn.commit()
    conn.close()
    return gid


def upsert_events_for_group(group_id: int, weeks_ahead: int = 12):
    """Generate weekly events if missing for the next N weeks."""
    conn = get_conn()
    g = conn.execute("SELECT weekday, start_time, price_cents FROM groups WHERE id=?", (group_id,)).fetchone()
    if not g:
        conn.close(); return
    weekday, start_time, price_cents = g
    today = date.today()
    dates = next_dates_for_weekday(today, weekday, weeks_ahead)
    for d in dates:
        h, m = map(int, start_time.split(":"))
        starts_at = datetime.combine(d, dt_time(hour=h, minute=m))
        exists = conn.execute("SELECT 1 FROM events WHERE group_id=? AND starts_at=?", (group_id, starts_at.isoformat())).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO events(group_id, starts_at, price_cents, generated) VALUES(?,?,?,1)",
                (group_id, starts_at.isoformat(), price_cents),
            )
    conn.commit(); conn.close()


def list_groups_for_user(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT g.id, g.name, g.city, g.venue, g.weekday, g.start_time, g.price_cents, g.blik_phone,
               CASE WHEN m.role='moderator' THEN 1 ELSE 0 END AS is_mod
        FROM groups g
        JOIN memberships m ON m.group_id=g.id
        WHERE m.user_id=?
        ORDER BY g.city, g.name
        """,
        conn,
        params=(user_id,),
    )
    conn.close()
    return df


def join_group(user_id: int, group_id: int):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO memberships(user_id, group_id, role) VALUES(?,?,'member')", (user_id, group_id))
    conn.commit(); conn.close()


def get_group(group_id: int):
    conn = get_conn()
    row = conn.execute("SELECT id, name, city, venue, weekday, start_time, price_cents, blik_phone FROM groups WHERE id=?", (group_id,)).fetchone()
    conn.close()
    return row


def events_df(group_id: int, only_future: bool = True) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT id, starts_at, price_cents, locked FROM events WHERE group_id=?"
    params = [group_id]
    if only_future:
        q += " AND datetime(starts_at) >= datetime('now','-1 day')"
    q += " ORDER BY datetime(starts_at)"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close();
    if not df.empty:
        df["starts_at"] = pd.to_datetime(df["starts_at"]) 
    return df


def sign_up(event_id: int, user_id: int):
    conn = get_conn()
    now = datetime.now().isoformat()
    conn.execute("INSERT OR IGNORE INTO event_signups(event_id, user_id, signed_at) VALUES(?,?,?)", (event_id, user_id, now))
    conn.execute("INSERT OR IGNORE INTO payments(event_id, user_id, user_marked_paid, moderator_confirmed) VALUES(?,?,0,0)", (event_id, user_id))
    conn.commit(); conn.close()


def withdraw(event_id: int, user_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM event_signups WHERE event_id=? AND user_id=?", (event_id, user_id))
    conn.execute("DELETE FROM payments WHERE event_id=? AND user_id=?", (event_id, user_id))
    conn.commit(); conn.close()


def payment_toggle(event_id: int, user_id: int, field: str, value: int):
    conn = get_conn()
    conn.execute(f"UPDATE payments SET {field}=? WHERE event_id=? AND user_id=?", (value, event_id, user_id))
    conn.commit(); conn.close()


def get_event_context(event_id: int):
    conn = get_conn()
    e = conn.execute("SELECT id, group_id, starts_at, price_cents, locked FROM events WHERE id=?", (event_id,)).fetchone()
    signups = pd.read_sql_query(
        """
        SELECT es.user_id, u.name,
               COALESCE(p.user_marked_paid,0) AS user_marked_paid,
               COALESCE(p.moderator_confirmed,0) AS moderator_confirmed
        FROM event_signups es
        JOIN users u ON u.id=es.user_id
        LEFT JOIN payments p ON p.event_id=es.event_id AND p.user_id=es.user_id
        WHERE es.event_id=?
        ORDER BY u.name
        """,
        conn,
        params=(event_id,),
    )
    teams = pd.read_sql_query(
        "SELECT id, name, idx, goals FROM teams WHERE event_id=? ORDER BY idx", conn, params=(event_id,))
    conn.close()
    return e, signups, teams


def set_team_goals(team_id: int, goals: int):
    conn = get_conn(); conn.execute("UPDATE teams SET goals=? WHERE id=?", (goals, team_id)); conn.commit(); conn.close()


def create_team(event_id: int, name: str, idx: int) -> int:
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO teams(event_id, name, idx, goals) VALUES(?,?,?,0)", (event_id, name, idx))
    tid = cur.lastrowid
    conn.commit(); conn.close(); return tid


def add_member_to_team(team_id: int, user_id: int):
    conn = get_conn(); conn.execute("INSERT OR IGNORE INTO team_members(team_id, user_id) VALUES(?,?)", (team_id, user_id)); conn.commit(); conn.close()


def remove_member_from_team(team_id: int, user_id: int):
    conn = get_conn(); conn.execute("DELETE FROM team_members WHERE team_id=? AND user_id=?", (team_id, user_id)); conn.commit(); conn.close()


def list_team_members(team_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT tm.user_id, u.name
        FROM team_members tm JOIN users u ON u.id=tm.user_id
        WHERE tm.team_id=? ORDER BY u.name
        """,
        conn,
        params=(team_id,),
    )
    conn.close(); return df


def record_goal(event_id: int, scorer_id: int, assist_id: Optional[int], team_id: Optional[int], minute: Optional[int]):
    conn = get_conn(); conn.execute(
        "INSERT INTO goals(event_id, scorer_id, assist_id, team_id, minute) VALUES(?,?,?,?,?)",
        (event_id, scorer_id, assist_id, team_id, minute)
    ); conn.commit(); conn.close()


def goals_df(event_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT g.id, g.minute, s.name AS scorer, a.name AS assist, t.name AS team
        FROM goals g
        LEFT JOIN users s ON s.id=g.scorer_id
        LEFT JOIN users a ON a.id=g.assist_id
        LEFT JOIN teams t ON t.id=g.team_id
        WHERE g.event_id=?
        ORDER BY COALESCE(g.minute, 0), g.id
        """,
        conn,
        params=(event_id,),
    )
    conn.close(); return df


def total_team_goals(event_id: int) -> Tuple[int, List[int]]:
    conn = get_conn()
    rows = conn.execute("SELECT goals FROM teams WHERE event_id=? ORDER BY idx", (event_id,)).fetchall()
    conn.close()
    ls = [r[0] for r in rows]
    return sum(ls), ls


def computed_stats(group_id: int, year: int) -> pd.DataFrame:
    conn = get_conn()
    # Build per-user stats based on goals table + team results
    # Team results: team with max goals in an event wins; ties counted as draw
    df_users = pd.read_sql_query("SELECT id, name FROM users", conn)
    df_events = pd.read_sql_query("SELECT id FROM events WHERE group_id=? AND strftime('%Y', starts_at)=?", conn, params=(group_id, f"{year:04d}"))

    # Goals/assists
    df_g = pd.read_sql_query(
        """
        SELECT g.event_id, g.scorer_id, g.assist_id
        FROM goals g JOIN events e ON e.id=g.event_id
        WHERE e.group_id=? AND strftime('%Y', e.starts_at)=?
        """,
        conn,
        params=(group_id, f"{year:04d}"),
    )

    # Team membership per event
    df_tm = pd.read_sql_query(
        """
        SELECT t.event_id, t.id AS team_id, tm.user_id, t.goals
        FROM teams t JOIN team_members tm ON tm.team_id=t.id
        WHERE t.event_id IN (SELECT id FROM events WHERE group_id=?)
        """,
        conn,
        params=(group_id,),
    )

    # Determine winners per event
    df_tg = pd.read_sql_query("SELECT event_id, MAX(goals) AS maxg FROM teams GROUP BY event_id", conn)

    # Aggregate
    stats = {u: {"name": n, "goals": 0, "assists": 0, "wins": 0, "losses": 0, "draws": 0} for u, n in df_users.itertuples(index=False)}

    for _, row in df_g.iterrows():
        if pd.notna(row["scorer_id"]):
            stats[row["scorer_id"]]["goals"] += 1
        if pd.notna(row["assist_id"]):
            stats[row["assist_id"]]["assists"] += 1

    # Wins/Losses/Draws per user via team goals vs max goals in event
    if not df_tm.empty and not df_tg.empty:
        merged = df_tm.merge(df_tg, on="event_id", how="left")
        for _, r in merged.iterrows():
            if pd.isna(r["maxg"]):
                continue
            if r["goals"] == r["maxg"]:
                stats[r["user_id"]]["wins"] += 1
            elif r["goals"] < r["maxg"]:
                stats[r["user_id"]]["losses"] += 1

    rows = []
    for uid, s in stats.items():
        rows.append({"user_id": uid, **s})
    out = pd.DataFrame(rows)
    out["points"] = out["wins"]*3 + out["draws"]
    out = out.sort_values(["points","goals","assists"], ascending=False)
    conn.close(); return out


# ---------------------------
# UI
# ---------------------------

def sidebar_auth():
    st.sidebar.header("Logowanie")
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

    if "user_id" in st.session_state:
        st.sidebar.info(f"Zalogowano jako: {st.session_state['user_name']}")
        if st.sidebar.button("Wyloguj"):
            for k in ["user_id","user_name","selected_group_id","selected_event_id"]:
                st.session_state.pop(k, None)
            st.rerun()


def page_groups():
    st.header("Twoje grupy")
    uid = st.session_state.get("user_id")
    if not uid:
        st.info("Zaloguj się z lewego panelu.")
        return

    # Create group
    with st.expander("➕ Utwórz nową grupę", expanded=False):
        col1, col2 = st.columns(2)
        name = col1.text_input("Nazwa grupy")
        city = col1.text_input("Miejscowość")
        venue = col1.text_input("Miejsce wydarzenia (hala)")
        weekday = col2.selectbox("Dzień tygodnia", list(range(7)), format_func=lambda i: ["Pon","Wt","Śr","Czw","Pt","Sob","Nd"][i])
        start_time = col2.text_input("Godzina startu (HH:MM)", value="21:00")
        price = col2.number_input("Cena za halę (zł)", min_value=0.0, step=1.0)
        blik = col2.text_input("Numer BLIK/telefon do płatności")
        if st.button("Utwórz grupę"):
            if all([name, city, venue, blik]) and ":" in start_time:
                gid = create_group(name, city, venue, weekday, start_time, int(round(price*100)), blik, uid)
                upsert_events_for_group(gid)
                st.success("Grupa utworzona. Dodano nadchodzące wydarzenia na 12 tygodni.")
            else:
                st.error("Uzupełnij wszystkie pola.")

    # List groups
    df = list_groups_for_user(uid)
    if df.empty:
        st.info("Nie należysz jeszcze do żadnej grupy. Utwórz grupę powyżej lub poproś moderatora o dodanie.")
        return

    for _, g in df.iterrows():
        with st.container(border=True):
            cols = st.columns([2,2,2,2,1])
            cols[0].markdown(f"**{g['name']}**\n\n{g['city']} — {g['venue']}")
            cols[1].markdown(f"{time_label(int(g['weekday']), g['start_time'])}")
            cols[2].markdown(f"Cena: {cents_to_str(int(g['price_cents']))}")
            cols[3].markdown(f"Płatność BLIK: **{g['blik_phone']}**")
            if cols[4].button("Wejdź", key=f"enter_{g['id']}"):
                st.session_state["selected_group_id"] = int(g['id'])
                upsert_events_for_group(int(g['id']))
                st.rerun()


def page_group_dashboard(group_id: int):
    g = get_group(group_id)
    if not g:
        st.error("Grupa nie istnieje")
        return
    gid, name, city, venue, weekday, start_time, price_cents, blik_phone = g
    st.header(f"{name} — {city} · {venue}")
    st.caption(f"Gramy: {time_label(weekday, start_time)} · Cena hali: {cents_to_str(price_cents)} · BLIK: {blik_phone}")

    uid = st.session_state.get("user_id")
    mod = is_moderator(uid, gid)

    tabs = st.tabs(["Nadchodzące", "Płatności", "Drużyny & Wynik", "Statystyki" + (" (admin)" if mod else "")])

    # Nadchodzące
    with tabs[0]:
        df = events_df(gid, only_future=True)
        if df.empty:
            st.info("Brak nadchodzących wydarzeń")
        for _, ev in df.iterrows():
            e, signups, teams = get_event_context(int(ev["id"]))
            starts = pd.to_datetime(e[2])
            with st.container(border=True):
                st.subheader(starts.strftime("%d.%m.%Y %H:%M"))
                c1, c2, c3 = st.columns([2,2,2])
                # Participation
                is_signed = uid in set(signups["user_id"]) if not signups.empty else False
                if is_signed:
                    if c1.button("Wypisz się", key=f"wd_{e[0]}"):
                        withdraw(e[0], uid)
                        st.rerun()
                else:
                    # Business rule: pokaż możliwość zapisu dopiero po kliknięciu "Zapłacę BLIK" (user_marked_paid)
                    pay_rec = None
                    conn = get_conn()
                    pay_rec = conn.execute("SELECT user_marked_paid FROM payments WHERE event_id=? AND user_id=?", (e[0], uid)).fetchone()
                    conn.close()
                    can_signup = bool(pay_rec and pay_rec[0] == 1)
                    if not can_signup:
                        c1.info("Aby zapisać się, kliknij najpierw 'Zapłacę BLIK' i opłać udział.")
                    if can_signup and c1.button("Zapisz się", key=f"su_{e[0]}"):
                        sign_up(e[0], uid)
                        st.rerun()

                # Payment box
                with c2.expander("💳 Zapłacę BLIK"):
                    st.markdown(f"**Numer do BLIK / telefon:** {blik_phone}")
                    st.caption("Skopiuj numer, zapłać i oznacz poniżej.")
                    # mark self-paid
                    conn = get_conn()
                    row = conn.execute("SELECT user_marked_paid FROM payments WHERE event_id=? AND user_id=?", (e[0], uid)).fetchone()
                    conn.close()
                    cur_val = row[0] if row else 0
                    new_val = st.checkbox("Oznaczam: zapłacone", value=bool(cur_val), key=f"ump_{e[0]}")
                    if int(new_val) != int(bool(cur_val)):
                        payment_toggle(e[0], uid, 'user_marked_paid', int(new_val))
                        if new_val and not is_signed:
                            st.success("Znakomicie! Teraz możesz się zapisać na grę.")

                # People & split cost
                count = len(signups)
                per_head = ev["price_cents"] / 100 / max(1, count)
                c3.metric("Zapisani", f"{count}")
                c3.metric("Koszt na osobę", f"{per_head:.2f} zł")
                st.dataframe(signups.rename(columns={"name":"Uczestnik","user_marked_paid":"Zapłacone (użytkownik)","moderator_confirmed":"Potwierdzone (mod)"})[["Uczestnik","Zapłacone (użytkownik)","Potwierdzone (mod)"]], hide_index=True, use_container_width=True)

    # Płatności
    with tabs[1]:
        df = events_df(gid, only_future=False)
        if df.empty:
            st.info("Brak wydarzeń")
        else:
            pick = st.selectbox("Wybierz wydarzenie", list(df["id"]), format_func=lambda i: pd.to_datetime(df.loc[df["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M"))
            e, signups, _ = get_event_context(int(pick))
            st.subheader("Płatności — " + pd.to_datetime(e[2]).strftime("%d.%m.%Y %H:%M"))
            if is_moderator(uid, gid):
                for _, r in signups.iterrows():
                    cols = st.columns([3,1,1,1])
                    cols[0].markdown(f"**{r['name']}**")
                    cols[1].checkbox("Zapłacił", value=bool(r['user_marked_paid']), key=f"u_{pick}_{r['user_id']}", disabled=True)
                    new_conf = cols[2].checkbox("Potwierdź (mod)", value=bool(r['moderator_confirmed']), key=f"m_{pick}_{r['user_id']}")
                    if int(new_conf) != int(r['moderator_confirmed']):
                        payment_toggle(int(pick), int(r['user_id']), 'moderator_confirmed', int(new_conf))
                st.caption("Uwaga: bez potwierdzenia moderatora płatność nie jest finalna.")
            else:
                st.dataframe(signups.rename(columns={"name":"Uczestnik","user_marked_paid":"Zapłacone (użytkownik)","moderator_confirmed":"Potwierdzone (mod)"})[["Uczestnik","Zapłacone (użytkownik)","Potwierdzone (mod)"]], hide_index=True, use_container_width=True)

    # Teams & Result
    with tabs[2]:
        df = events_df(gid, only_future=False)
        if df.empty:
            st.info("Brak wydarzeń")
        else:
            pick = st.selectbox("Wydarzenie", list(df["id"]), key="ev_for_teams", format_func=lambda i: pd.to_datetime(df.loc[df["id"]==i, "starts_at"].values[0]).strftime("%d.%m.%Y %H:%M"))
            e, signups, teams = get_event_context(int(pick))

            if is_moderator(uid, gid):
                st.subheader("Zarządzanie drużynami (moderator)")
                cols = st.columns(3)
                for i in range(3):
                    with cols[i]:
                        label = f"Drużyna {i+1}"
                        existing = teams[teams["idx"]==i+1]
                        if existing.empty:
                            if st.button(f"Dodaj {label}", key=f"add_team_{i}"):
                                create_team(int(pick), label, i+1)
                                st.rerun()
                        else:
                            tid = int(existing.iloc[0]["id"]) 
                            st.markdown(f"**{existing.iloc[0]['name']}**")
                            cur_members = list_team_members(tid)
                            options = [(int(u), n) for u, n in signups[["user_id","name"]].itertuples(index=False)]
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
            # Everyone can record their own goal/assist
            # Pick team (optional)
            team_map = {int(r["id"]): f"{r['name']}" for _, r in teams.iterrows()}
            team_choice = st.selectbox("Drużyna (opcjonalnie)", options=[None] + list(team_map.keys()), format_func=lambda x: "—" if x is None else team_map[x])
            scorer_choice = st.selectbox("Strzelec", options=[(int(i), n) for i,n in signups[["user_id","name"]].itertuples(index=False)], format_func=lambda x: x[1])
            assist_choice = st.selectbox("Asysta (opcjonalnie)", options=[None] + [(int(i), n) for i,n in signups[["user_id","name"]].itertuples(index=False)], format_func=lambda x: "—" if x is None else x[1])
            minute = st.number_input("Minuta (opcjonalnie)", min_value=0, max_value=200, step=1, value=0)
            if st.button("Dodaj gola"):
                record_goal(int(pick), int(scorer_choice[0]), int(assist_choice[0]) if assist_choice else None, int(team_choice) if team_choice else None, int(minute) if minute else None)
                st.success("Dodano gola")

            st.markdown("### Lista goli")
            gdf = goals_df(int(pick))
            st.dataframe(gdf, hide_index=True, use_container_width=True)

            # Validation: sum goals equals total team goals
            total_g, team_g = total_team_goals(int(pick))
            if len(gdf) != total_g and total_g > 0:
                st.warning(f"Suma goli wprowadzonych ({len(gdf)}) ≠ suma bramek drużyn ({total_g}). Uzupełnij dane.")
            elif total_g == 0:
                st.info("Podaj gole drużyn, aby móc zweryfikować zgodność.")
            else:
                st.success("Liczba goli się zgadza ✅")

    # Stats
    with tabs[3]:
        year = st.selectbox("Rok", options=list(range(datetime.now().year, datetime.now().year-5, -1)))
        df_stats = computed_stats(gid, int(year))
        if df_stats.empty:
            st.info("Brak statystyk na wybrany rok")
        else:
            st.dataframe(df_stats.rename(columns={"name":"Zawodnik","goals":"Gole","assists":"Asysty","wins":"Wygrane","losses":"Przegrane","draws":"Remisy","points":"Punkty"})[["Zawodnik","Gole","Asysty","Wygrane","Przegrane","Remisy","Punkty"]], hide_index=True, use_container_width=True)

        if is_moderator(uid, gid):
            st.markdown("---")
            st.subheader("Narzędzia moderatora")
            if st.button("Wygeneruj 12 kolejnych wydarzeń"):
                upsert_events_for_group(gid, 12)
                st.success("Dodano brakujące wydarzenia w kalendarzu.")


# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    init_db()

    sidebar_auth()

    page = st.sidebar.radio("Nawigacja", ["Grupy", "Panel grupy"], label_visibility="collapsed")

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
