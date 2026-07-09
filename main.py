"""
wiebke_cloud — subscription proxy server.

Endpoints:
  POST /auth/register           Create account
  POST /auth/login              Get JWT token
  POST /auth/refresh            Refresh JWT token
  POST /auth/forgot-password    Request a password reset email
  GET  /auth/reset-password     Reset-password HTML form (link from the email)
  POST /auth/reset-password     Reset-password form handler
  GET  /account/status          Subscription status + AI usage this period
  POST /stripe/checkout         Start Stripe checkout
  POST /stripe/portal           Manage subscription
  POST /stripe/webhook          Stripe event handler
  POST /v1/chat                 Proxy to OpenAI (requires active sub + budget)
  POST /v1/transcribe           Transcribe audio via Whisper (requires active sub + budget)
  POST /v1/memory/summarize     Summarise a conversation turn (requires active sub)
  POST /v1/memory/filter        Decide whether to store a memory (requires active sub)
  POST /v1/memory/extract       Extract structured facts (requires active sub)
  POST /v1/memory/embed         Generate an embedding vector (requires active sub)
  POST /v1/memory/update        Merge a turn into long-term memory (requires active sub)
  POST /v1/memory/tags          Generate short tags for a turn (requires active sub)
  POST /v1/memory/rerank        Pick the most relevant candidate memories (requires active sub)
  POST /v1/memory/duplicate     Check a new memory against recent ones (requires active sub)
  POST /v1/goals/parse          Extract {name, category, deadline} from a chat message (requires active sub)

All OpenAI-touching endpoints meter their cost against the caller's monthly
usage budget (see "AI usage metering" below). /v1/chat and /v1/transcribe are
the only ones that actually refuse a call once hard-blocked — the memory
endpoints only ever fire as a side effect of a chat turn that already passed
that gate, so gating chat alone already caps total spend.

Deploy to Railway / Render / Fly.io:
  Required environment variables:
    OPENAI_API_KEY        your OpenAI key (never exposed to users)
    STRIPE_SECRET_KEY     sk_live_...
    STRIPE_WEBHOOK_SECRET whsec_...
    STRIPE_PRICE_ID       price_... (your £9.99/mo product)
    JWT_SECRET            any long random string

  Optional:
    APP_URL               https://yourapp.com  (for Stripe redirects + reset links)
    DATABASE_URL          postgresql://...     (Railway Postgres — strongly recommended)
    DB_PATH               /data/wiebke.db      (SQLite path when DATABASE_URL is not set)
    RESEND_API_KEY         re_...               (for forgot-password emails — see resend.com)
    RESEND_FROM_EMAIL      "Wiebke <noreply@yourdomain>" (defaults to Resend's sandbox sender)

  On Railway: add a PostgreSQL service — DATABASE_URL is injected automatically.
  Without DATABASE_URL the server falls back to SQLite, which is wiped on every
  redeploy. Use SQLite only for local development.

  Without RESEND_API_KEY set, /auth/forgot-password still responds normally
  (to avoid leaking which emails are registered) but no email is actually sent
  — a warning is printed server-side instead. The app does not fail to start.
"""

import hashlib
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import bcrypt
import requests
import stripe
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from jose import JWTError, jwt
from openai import OpenAI
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY     = os.environ["OPENAI_API_KEY"]
STRIPE_SECRET_KEY  = os.environ["STRIPE_SECRET_KEY"]
STRIPE_WEBHOOK_SEC = os.environ["STRIPE_WEBHOOK_SECRET"]
STRIPE_PRICE_ID    = os.environ["STRIPE_PRICE_ID"]
JWT_SECRET         = os.environ.get("JWT_SECRET", "change-me-in-production")
APP_URL            = os.environ.get("APP_URL", "https://wiebke.app")
DATABASE_URL       = os.environ.get("DATABASE_URL", "")
RESEND_API_KEY     = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM_EMAIL  = os.environ.get("RESEND_FROM_EMAIL", "Wiebke <onboarding@resend.dev>")
JWT_ALGORITHM      = "HS256"
ACCESS_TOKEN_TTL   = 60 * 60 * 24 * 30  # 30 days

stripe.api_key = STRIPE_SECRET_KEY
openai_client  = OpenAI(api_key=OPENAI_API_KEY)


# ──────────────────────────────────────────────────────────────────
# AI usage metering
# ──────────────────────────────────────────────────────────────────
#
# Static price table (USD per 1M tokens). OpenAI's published rates change
# occasionally — review periodically. Model choice directly affects how
# fast a subscriber's monthly budget depletes, which is the point: it's
# the same table used to validate /v1/chat's requested model.

ALLOWED_CHAT_MODELS = {
    # model name:    (input $ / 1M tokens, output $ / 1M tokens)
    "gpt-4.1-mini":  (0.40, 1.60),    # Economy — default
    "gpt-4o-mini":   (0.15, 0.60),    # Standard
    "gpt-4o":        (2.50, 10.00),   # Premium
}
DEFAULT_CHAT_MODEL     = "gpt-4.1-mini"
SUPPORT_MODEL          = "gpt-4.1-mini"  # tags/filter/extract/summarize/update/rerank/duplicate always use this, regardless of the user's chat model
EMBEDDING_PRICE_PER_1M = 0.02            # text-embedding-3-small
WHISPER_PRICE_PER_MIN  = 0.006

# £4.00/month ≈ $5.00 — a static, conservative conversion (not live FX).
# Retune these if GBP/USD moves a lot or OpenAI reprices.
MONTHLY_BUDGET_USD = 5.00
WARN_THRESHOLD_USD = 4.00   # 80% — desktop shows a warning banner
HARD_BLOCK_USD     = 7.50   # 150% — further /v1/chat and /v1/transcribe calls are refused
USAGE_PERIOD_DAYS  = 30


# ──────────────────────────────────────────────────────────────────
# Password hashing  (bcrypt direct — no passlib)
# ──────────────────────────────────────────────────────────────────
#
# bcrypt silently truncates passwords longer than 72 bytes, which is
# a silent security degradation. We pre-hash with SHA-256 (32 bytes)
# so every password, regardless of length, is treated consistently.
# The bcrypt layer still provides the slow, salted work factor.

def _hash_password(password: str) -> str:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.hashpw(digest, bcrypt.gensalt(rounds=12)).decode("utf-8")


def _verify_password(password: str, stored_hash: str) -> bool:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.checkpw(digest, stored_hash.encode("utf-8"))


# ──────────────────────────────────────────────────────────────────
# Database — PostgreSQL (preferred) or SQLite (local / fallback)
# ──────────────────────────────────────────────────────────────────

# Railway injects DATABASE_URL as  postgres://...
# psycopg2 requires the scheme to be postgresql://
_PG_URL      = DATABASE_URL.replace("postgres://", "postgresql://", 1) if DATABASE_URL else ""
USE_POSTGRES = _PG_URL.startswith("postgresql://")

# SQLite fallback — only used when DATABASE_URL is not set
DB_PATH = Path(os.environ.get("DB_PATH", "wiebke_users.db"))

# Paramstyle differs between the two drivers:  %s (psycopg2)  vs  ? (sqlite3)
PH = "%s" if USE_POSTGRES else "?"

# Eagerly import psycopg2 if Postgres is configured so a missing package
# surfaces at startup rather than on the first database call.
if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL points to PostgreSQL but psycopg2 is not installed. "
            "Add  psycopg2-binary>=2.9  to requirements.txt and redeploy."
        ) from exc


@contextmanager
def _db() -> Generator:
    """
    Open a database connection and yield its cursor.

    Commits automatically on clean exit; rolls back and re-raises on any
    exception; always closes the connection.

    Both backends return dict-like rows so column access by name works the
    same way for callers:
      - PostgreSQL  → psycopg2 RealDictRow   (row["email"])
      - SQLite      → sqlite3.Row            (row["email"])
    """
    if USE_POSTGRES:
        conn = psycopg2.connect(_PG_URL)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def _init_db() -> None:
    """Create the users table if it does not already exist."""
    with _db() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                id                   TEXT PRIMARY KEY,
                email                TEXT UNIQUE NOT NULL,
                password_hash        TEXT NOT NULL,
                stripe_customer_id   TEXT,
                subscription_status  TEXT DEFAULT 'inactive',
                subscription_end     TEXT,
                created_at           TEXT,
                reset_token_hash     TEXT,
                reset_token_expires  TEXT,
                usage_cost_usd       REAL DEFAULT 0,
                usage_period_start   TEXT
            )
        """)
    _migrate_columns()


def _migrate_columns() -> None:
    """
    Add columns that CREATE TABLE IF NOT EXISTS won't retroactively add to
    a users table that already existed before this deploy. Safe to run on
    every startup — each ALTER is wrapped individually so an already-present
    column (the normal case after the first run) is silently skipped rather
    than aborting the rest.
    """
    new_columns = [
        ("reset_token_hash",    "TEXT"),
        ("reset_token_expires", "TEXT"),
        ("usage_cost_usd",      "REAL DEFAULT 0"),
        ("usage_period_start",  "TEXT"),
    ]
    for col, decl in new_columns:
        try:
            with _db() as cur:
                cur.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
        except Exception:
            pass  # column already exists


# ── Row-level helpers ────────────────────────────────────────────

def _get_user(user_id: str):
    with _db() as cur:
        cur.execute(f"SELECT * FROM users WHERE id = {PH}", (user_id,))
        return cur.fetchone()


def _get_user_by_email(email: str):
    with _db() as cur:
        cur.execute(f"SELECT * FROM users WHERE email = {PH}", (email.lower(),))
        return cur.fetchone()


def _create_user(
    user_id: str,
    email: str,
    pw_hash: str,
    stripe_customer_id: str,
    created_at: str,
) -> None:
    with _db() as cur:
        cur.execute(
            f"""INSERT INTO users
                    (id, email, password_hash, stripe_customer_id, created_at)
                VALUES ({PH}, {PH}, {PH}, {PH}, {PH})""",
            (user_id, email, pw_hash, stripe_customer_id, created_at),
        )


def _update_subscription(
    stripe_customer_id: str,
    status: str,
    end_date: str | None = None,
) -> None:
    with _db() as cur:
        cur.execute(
            f"""UPDATE users
                   SET subscription_status = {PH},
                       subscription_end    = {PH}
                 WHERE stripe_customer_id  = {PH}""",
            (status, end_date, stripe_customer_id),
        )


def _set_password(user_id: str, pw_hash: str) -> None:
    with _db() as cur:
        cur.execute(
            f"UPDATE users SET password_hash = {PH} WHERE id = {PH}",
            (pw_hash, user_id),
        )


def _set_reset_token(user_id: str, token_hash: str, expires_iso: str) -> None:
    with _db() as cur:
        cur.execute(
            f"""UPDATE users
                   SET reset_token_hash    = {PH},
                       reset_token_expires = {PH}
                 WHERE id = {PH}""",
            (token_hash, expires_iso, user_id),
        )


def _clear_reset_token(user_id: str) -> None:
    with _db() as cur:
        cur.execute(
            f"""UPDATE users
                   SET reset_token_hash = NULL, reset_token_expires = NULL
                 WHERE id = {PH}""",
            (user_id,),
        )


# ──────────────────────────────────────────────────────────────────
# AI usage helpers
# ──────────────────────────────────────────────────────────────────

def _reset_usage(user_id: str, period_start: datetime) -> None:
    with _db() as cur:
        cur.execute(
            f"""UPDATE users
                   SET usage_cost_usd = 0, usage_period_start = {PH}
                 WHERE id = {PH}""",
            (period_start.isoformat(), user_id),
        )


def _get_usage(user_id: str) -> dict:
    """
    Returns {"cost_usd", "period_start"} for the user's current billing
    window, lazily rolling it forward (resetting the counter to 0) if more
    than USAGE_PERIOD_DAYS has elapsed since it started. No cron needed —
    this runs on every usage-touching request.
    """
    user = _get_user(user_id)
    now  = datetime.now(timezone.utc)
    raw_start = user["usage_period_start"] if user else None

    if not raw_start or (now - datetime.fromisoformat(raw_start)) > timedelta(days=USAGE_PERIOD_DAYS):
        _reset_usage(user_id, now)
        return {"cost_usd": 0.0, "period_start": now.isoformat()}

    return {"cost_usd": float(user["usage_cost_usd"] or 0), "period_start": raw_start}


def _add_usage(user_id: str, cost_usd: float) -> float:
    """Adds cost_usd to the user's running total (after any due reset) and returns the new total."""
    current   = _get_usage(user_id)
    new_total = current["cost_usd"] + cost_usd
    with _db() as cur:
        cur.execute(
            f"UPDATE users SET usage_cost_usd = {PH} WHERE id = {PH}",
            (new_total, user_id),
        )
    return new_total


def _chat_cost(model: str, usage) -> float:
    """Cost in USD for a chat/completions call given OpenAI's usage object."""
    input_price, output_price = ALLOWED_CHAT_MODELS.get(model, ALLOWED_CHAT_MODELS[DEFAULT_CHAT_MODEL])
    prompt_tokens     = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return (prompt_tokens / 1_000_000 * input_price) + (completion_tokens / 1_000_000 * output_price)


def _embedding_cost(usage) -> float:
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    return total_tokens / 1_000_000 * EMBEDDING_PRICE_PER_1M


def _whisper_cost(duration_seconds: float) -> float:
    return (duration_seconds / 60.0) * WHISPER_PRICE_PER_MIN


# ──────────────────────────────────────────────────────────────────
# Email (Resend)
# ──────────────────────────────────────────────────────────────────

def _send_email(to: str, subject: str, html: str) -> bool:
    """
    Sends an email via Resend's REST API. Soft-fails (logs and returns
    False) rather than raising — a missing/misconfigured RESEND_API_KEY
    degrades to "no email sent", not a crash. Callers that must not leak
    whether an account exists (forgot-password) return their generic
    response regardless of this result.
    """
    if not RESEND_API_KEY:
        print(f"[wiebke_cloud] RESEND_API_KEY not set — skipping email to {to}")
        return False
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={"from": RESEND_FROM_EMAIL, "to": [to], "subject": subject, "html": html},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"[wiebke_cloud] Failed to send email to {to}: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────
# JWT helpers
# ──────────────────────────────────────────────────────────────────

def _make_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + ACCESS_TOKEN_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _verify_token(token: str) -> str:
    """Decode a JWT and return the user_id, or raise HTTP 401."""
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return data["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def _bearer_user_id(authorization: str) -> str:
    """Extract and verify the user_id from an  Authorization: Bearer <token>  header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    return _verify_token(authorization.split(" ", 1)[1])


# ──────────────────────────────────────────────────────────────────
# Auth dependency  (used by gated endpoints)
# ──────────────────────────────────────────────────────────────────

def require_active_sub(authorization: str = Header(...)) -> str:
    """
    FastAPI dependency — validates the JWT and confirms an active subscription.
    Returns the user_id on success.
    """
    user_id = _bearer_user_id(authorization)
    user    = _get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Please log in again.")
    if user["subscription_status"] != "active":
        raise HTTPException(
            status_code=402,
            detail="No active subscription. Visit the Account page to subscribe.",
        )
    return user_id


def require_budget(user_id: str = Depends(require_active_sub)) -> str:
    """
    Additional gate for the two user-initiated, directly-expensive endpoints
    (/v1/chat, /v1/transcribe). The /v1/memory/* endpoints use plain
    require_active_sub and are metered but never blocked here — see the
    module docstring for why gating chat alone is sufficient.
    """
    usage = _get_usage(user_id)
    if usage["cost_usd"] >= HARD_BLOCK_USD:
        reset_date = (
            datetime.fromisoformat(usage["period_start"]) + timedelta(days=USAGE_PERIOD_DAYS)
        ).date().isoformat()
        raise HTTPException(
            status_code=429,
            detail=f"Usage limit reached for this billing period. Resets on {reset_date}.",
        )
    return user_id


# ──────────────────────────────────────────────────────────────────
# App + lifespan
# ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    if USE_POSTGRES:
        # Show host only — never log credentials
        host = _PG_URL.split("@")[-1] if "@" in _PG_URL else _PG_URL
        print(f"[wiebke_cloud] database: PostgreSQL  ({host})")
    else:
        print(f"[wiebke_cloud] database: SQLite  ({DB_PATH})  — not suitable for Railway")
    yield


app = FastAPI(title="Wiebke Cloud", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    str
    password: str


class LoginRequest(BaseModel):
    email:    str
    password: str


@app.post("/auth/register")
def register(req: RegisterRequest):
    email = req.email.strip().lower()

    if _get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered.")

    user_id     = str(uuid.uuid4())
    pw_hash     = _hash_password(req.password)
    stripe_cust = stripe.Customer.create(email=email)

    _create_user(
        user_id            = user_id,
        email              = email,
        pw_hash            = pw_hash,
        stripe_customer_id = stripe_cust.id,
        created_at         = datetime.now(timezone.utc).isoformat(),
    )
    return {"token": _make_token(user_id), "email": email}


@app.post("/auth/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    user  = _get_user_by_email(email)

    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    return {
        "token":               _make_token(user["id"]),
        "email":               user["email"],
        "subscription_status": user["subscription_status"],
    }


@app.post("/auth/refresh")
def refresh_token(authorization: str = Header(...)):
    """Issue a fresh token without requiring the password again."""
    user_id = _bearer_user_id(authorization)
    return {"token": _make_token(user_id)}


class ForgotPasswordRequest(BaseModel):
    email: str


@app.post("/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    """
    Always returns the same generic message, whether or not the email is
    registered, to avoid leaking which addresses have accounts. Only
    actually sends an email when the account exists.
    """
    email = req.email.strip().lower()
    user  = _get_user_by_email(email)

    if user:
        raw_token  = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        expires    = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        _set_reset_token(user["id"], token_hash, expires)

        reset_url = f"{APP_URL}/auth/reset-password?token={raw_token}&email={email}"
        _send_email(
            to=email,
            subject="Reset your Wiebke password",
            html=(
                "<p>Click the link below to choose a new password. "
                "This link expires in 30 minutes.</p>"
                f'<p><a href="{reset_url}">{reset_url}</a></p>'
                "<p>If you didn't request this, you can safely ignore this email.</p>"
            ),
        )

    return {"message": "If that email is registered, a reset link has been sent."}


@app.get("/auth/reset-password", response_class=HTMLResponse)
def reset_password_form(token: str = "", email: str = ""):
    """Renders the HTML form linked from the reset email."""
    body = f"""
<div class="card">
  <h1>Reset your password</h1>
  <p>Choose a new password for <span class="highlight">{email}</span>.</p>
  <form method="POST" action="/auth/reset-password" style="text-align:left; margin-top:24px;">
    <input type="hidden" name="token" value="{token}"/>
    <input type="hidden" name="email" value="{email}"/>
    <label style="font-size:13px; color:rgba(255,255,255,0.6);">New password (min 8 characters)</label>
    <input type="password" name="new_password" minlength="8" required
      style="width:100%; padding:12px; margin:8px 0 20px; border-radius:8px;
             border:1px solid rgba(255,255,255,0.15); background:#1A0700; color:#fff; font-size:14px;"/>
    <button type="submit"
      style="width:100%; padding:12px; border-radius:8px; border:none; background:#F97316;
             color:#fff; font-size:15px; font-weight:600; cursor:pointer;">Set new password</button>
  </form>
</div>"""
    return _PAGE_BASE.format(title="Reset password", body=body)


@app.post("/auth/reset-password", response_class=HTMLResponse)
async def reset_password_submit(request: Request):
    """Handles the form submission from reset_password_form."""
    form         = await request.form()
    token        = str(form.get("token", ""))
    email        = str(form.get("email", "")).strip().lower()
    new_password = str(form.get("new_password", ""))

    user       = _get_user_by_email(email)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    valid = bool(
        user
        and user["reset_token_hash"]
        and secrets.compare_digest(user["reset_token_hash"], token_hash)
        and user["reset_token_expires"]
        and datetime.fromisoformat(user["reset_token_expires"]) > datetime.now(timezone.utc)
    )

    if not valid or len(new_password) < 8:
        body = """
<div class="card">
  <div class="icon-circle cancel" style="font-size:28px; color:rgba(255,255,255,0.3)">×</div>
  <h1>Link expired or invalid</h1>
  <p>This password reset link is no longer valid, or the new password was too short.
  Request a new link from the Wiebke app.</p>
</div>"""
        return _PAGE_BASE.format(title="Reset failed", body=body)

    _set_password(user["id"], _hash_password(new_password))
    _clear_reset_token(user["id"])

    body = """
<div class="card">
  <div class="icon-circle success">✓</div>
  <h1>Password updated</h1>
  <p>Return to the Wiebke app and log in with your new password.</p>
</div>"""
    return _PAGE_BASE.format(title="Password updated", body=body)


# ──────────────────────────────────────────────────────────────────
# Account
# ──────────────────────────────────────────────────────────────────

@app.get("/account/status")
def account_status(authorization: str = Header(...)):
    user_id = _bearer_user_id(authorization)
    user    = _get_user(user_id)
    if not user:
        # Token is valid but user row is gone (e.g. account deleted).
        # Return 401 so the desktop clears its stored token instead of
        # looping on 404 and showing a confusing "Not Found" message.
        raise HTTPException(status_code=401, detail="User not found. Please log in again.")

    usage = _get_usage(user_id)
    return {
        "email":               user["email"],
        "subscription_status": user["subscription_status"],
        "subscription_end":    user["subscription_end"],
        "usage_cost_usd":      round(usage["cost_usd"], 4),
        "usage_budget_usd":    MONTHLY_BUDGET_USD,
        "usage_period_start":  usage["period_start"],
        "usage_warn":          usage["cost_usd"] >= WARN_THRESHOLD_USD,
    }


# ──────────────────────────────────────────────────────────────────
# Stripe
# ──────────────────────────────────────────────────────────────────

@app.post("/stripe/checkout")
def create_checkout(authorization: str = Header(...)):
    """Return a Stripe Checkout URL to start a new subscription."""
    user_id = _bearer_user_id(authorization)
    user    = _get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Please log in again.")

    session = stripe.checkout.Session.create(
        customer             = user["stripe_customer_id"],
        payment_method_types = ["card"],
        line_items           = [{"price": STRIPE_PRICE_ID, "quantity": 1}],
        mode                 = "subscription",
        success_url          = f"{APP_URL}/subscribe/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url           = f"{APP_URL}/subscribe/cancel",
        metadata             = {"user_id": user_id},
    )
    return {"checkout_url": session.url}


@app.post("/stripe/portal")
def customer_portal(authorization: str = Header(...)):
    """Return a Stripe Customer Portal URL for managing billing."""
    user_id = _bearer_user_id(authorization)
    user    = _get_user(user_id)
    if not user or not user["stripe_customer_id"]:
        raise HTTPException(status_code=404, detail="No billing account found.")

    session = stripe.billing_portal.Session.create(
        customer   = user["stripe_customer_id"],
        return_url = f"{APP_URL}/account",
    )
    return {"portal_url": session.url}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe events and update subscription status in the database."""
    payload = await request.body()
    sig     = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SEC)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature.")

    obj = event["data"]["object"]

    if event["type"] == "customer.subscription.created":
        _update_subscription(obj["customer"], "active")

    elif event["type"] == "customer.subscription.updated":
        status = "active" if obj["status"] == "active" else "inactive"
        end    = (
            datetime.fromtimestamp(
                obj["current_period_end"], tz=timezone.utc
            ).isoformat()
            if obj.get("current_period_end") else None
        )
        _update_subscription(obj["customer"], status, end)

    elif event["type"] in (
        "customer.subscription.deleted",
        "customer.subscription.paused",
    ):
        _update_subscription(obj["customer"], "cancelled")

    elif event["type"] == "invoice.payment_failed":
        _update_subscription(obj["customer"], "past_due")

    return {"received": True}


# ──────────────────────────────────────────────────────────────────
# Subscription landing pages
# ──────────────────────────────────────────────────────────────────

_PAGE_BASE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — Wiebke</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #1A0700;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .card {{
      background: #2A1400;
      border: 1px solid rgba(249,115,22,0.2);
      border-radius: 20px;
      padding: 56px 48px;
      max-width: 480px;
      width: 100%;
      text-align: center;
    }}
    .mark {{
      display: inline-block;
      margin-bottom: 32px;
    }}
    .icon-circle {{
      width: 72px;
      height: 72px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 28px;
      font-size: 32px;
    }}
    .icon-circle.success {{ background: rgba(249,115,22,0.15); }}
    .icon-circle.cancel  {{ background: rgba(255,255,255,0.06); }}
    h1 {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 12px;
      color: #fff;
    }}
    p {{
      font-size: 15px;
      color: rgba(255,255,255,0.6);
      line-height: 1.6;
      margin-bottom: 8px;
    }}
    .highlight {{ color: #F97316; font-weight: 600; }}
    .divider {{
      border: none;
      border-top: 1px solid rgba(255,255,255,0.08);
      margin: 32px 0;
    }}
    .step {{
      display: flex;
      align-items: flex-start;
      gap: 14px;
      text-align: left;
      margin-bottom: 16px;
    }}
    .step-num {{
      flex-shrink: 0;
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: rgba(249,115,22,0.2);
      color: #F97316;
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-top: 1px;
    }}
    .step p {{ margin: 0; }}
  </style>
</head>
<body>
  {body}
</body>
</html>"""

_SUCCESS_BODY = """
<div class="card">
  <div class="mark">
    <svg width="48" height="48" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="56" height="56" rx="14" fill="#1A0700"/>
      <path d="M 8 14 L 16 42 L 22 25 L 28.5 42 L 36 14"
        stroke="#F97316" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M 36 14 L 36 42" stroke="#F97316" stroke-width="4" stroke-linecap="round"/>
      <path d="M 36 28 L 47 14" stroke="#FCD34D" stroke-width="3.5" stroke-linecap="round"/>
      <path d="M 36 28 L 48 42" stroke="#FCD34D" stroke-width="3.5" stroke-linecap="round"/>
    </svg>
  </div>
  <div class="icon-circle success">✓</div>
  <h1>You're subscribed</h1>
  <p>Your <span class="highlight">Wiebke Pro</span> subscription is now active.</p>
  <hr class="divider"/>
  <div class="step">
    <div class="step-num">1</div>
    <p>Return to the Wiebke app on your desktop.</p>
  </div>
  <div class="step">
    <div class="step-num">2</div>
    <p>Go to <strong style="color:#fff">Account</strong> and click <strong style="color:#fff">Refresh status</strong>.</p>
  </div>
  <div class="step">
    <div class="step-num">3</div>
    <p>Your plan will show <span class="highlight">Active</span> — you can now use Chat.</p>
  </div>
  <hr class="divider"/>
  <p style="font-size:13px">You can close this tab.</p>
</div>"""

_CANCEL_BODY = """
<div class="card">
  <div class="mark">
    <svg width="48" height="48" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="56" height="56" rx="14" fill="#1A0700"/>
      <path d="M 8 14 L 16 42 L 22 25 L 28.5 42 L 36 14"
        stroke="#F97316" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M 36 14 L 36 42" stroke="#F97316" stroke-width="4" stroke-linecap="round"/>
      <path d="M 36 28 L 47 14" stroke="#FCD34D" stroke-width="3.5" stroke-linecap="round"/>
      <path d="M 36 28 L 48 42" stroke="#FCD34D" stroke-width="3.5" stroke-linecap="round"/>
    </svg>
  </div>
  <div class="icon-circle cancel" style="font-size:28px; color:rgba(255,255,255,0.3)">×</div>
  <h1>Payment cancelled</h1>
  <p>No charge was made. You can subscribe any time from the Account page in the Wiebke app.</p>
  <hr class="divider"/>
  <p style="font-size:13px">You can close this tab.</p>
</div>"""


@app.get("/subscribe/success", response_class=HTMLResponse)
def subscribe_success():
    """Stripe redirects here after a successful checkout."""
    return _PAGE_BASE.format(title="Subscription active", body=_SUCCESS_BODY)


@app.get("/subscribe/cancel", response_class=HTMLResponse)
def subscribe_cancel():
    """Stripe redirects here if the user closes the checkout."""
    return _PAGE_BASE.format(title="Payment cancelled", body=_CANCEL_BODY)


# ──────────────────────────────────────────────────────────────────
# OpenAI proxy  (requires an active subscription)
# ──────────────────────────────────────────────────────────────────

class ProxyChatRequest(BaseModel):
    messages:    list
    model:       str   = DEFAULT_CHAT_MODEL
    temperature: float = 0.85
    max_tokens:  int   = 1024


@app.post("/v1/chat")
def proxy_chat(
    req:     ProxyChatRequest,
    user_id: str = Depends(require_budget),
):
    """Forward a messages array to OpenAI on behalf of an active subscriber."""
    if req.model not in ALLOWED_CHAT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req.model}'. Allowed: {', '.join(ALLOWED_CHAT_MODELS)}.",
        )
    try:
        response = openai_client.chat.completions.create(
            model       = req.model,
            messages    = req.messages,
            temperature = req.temperature,
            max_tokens  = req.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {exc}")

    new_total = _add_usage(user_id, _chat_cost(req.model, response.usage))
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "cost_usd":   round(new_total, 4),
            "budget_usd": MONTHLY_BUDGET_USD,
            "warn":       new_total >= WARN_THRESHOLD_USD,
        },
    }


class ProxyTranscribeRequest(BaseModel):
    audio_base64: str
    filename:     str = "audio.wav"


@app.post("/v1/transcribe")
def proxy_transcribe(
    req:     ProxyTranscribeRequest,
    user_id: str = Depends(require_budget),
):
    """Transcribe audio via Whisper on behalf of an active subscriber."""
    import base64
    import io

    try:
        audio_bytes      = base64.b64decode(req.audio_base64)
        audio_file       = io.BytesIO(audio_bytes)
        audio_file.name  = req.filename
        result = openai_client.audio.transcriptions.create(
            model           = "whisper-1",
            file            = audio_file,
            response_format = "verbose_json",  # includes `duration`, needed to cost the call
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Whisper error: {exc}")

    _add_usage(user_id, _whisper_cost(getattr(result, "duration", 0) or 0))
    return {"text": result.text}


# ──────────────────────────────────────────────────────────────────
# Memory proxy  (requires an active subscription)
#
# Routes all memory operations (summarise, filter, extract, embed)
# through the cloud so users never need their own OpenAI API key.
# ──────────────────────────────────────────────────────────────────

import json as _json


class MemorySummarizeRequest(BaseModel):
    question: str
    answer:   str


class MemoryFilterRequest(BaseModel):
    question: str
    answer:   str


class MemoryExtractRequest(BaseModel):
    question:  str
    answer:    str
    user_name: str = "User"


class MemoryEmbedRequest(BaseModel):
    text: str


@app.post("/v1/memory/summarize")
def memory_summarize(
    req:     MemorySummarizeRequest,
    user_id: str = Depends(require_active_sub),
):
    """Summarise a conversation turn for long-term memory storage."""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": (
                "Summarize this interaction.\n\n"
                "Rules:\n- 1 to 3 sentences maximum.\n- Remove greetings.\n"
                "- Remove jokes.\n- Preserve important facts.\n- Be concise.\n\n"
                f"Question:\n{req.question}\n\nAnswer:\n{req.answer}"
            )}],
            temperature=0.3,
            max_tokens=200,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarize error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))
    return {"summary": resp.choices[0].message.content.strip()}


@app.post("/v1/memory/filter")
def memory_filter(
    req:     MemoryFilterRequest,
    user_id: str = Depends(require_active_sub),
):
    """Decide whether an interaction is worth storing as a long-term memory."""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": (
                "Determine whether this interaction should become a long-term memory.\n\n"
                "Store: identity, preferences, projects, goals, relationships, pets, "
                "hobbies, interests, vehicles, important life facts.\n\n"
                "Do NOT store: time, date, weather, sports fixtures, temporary events, "
                "news, one-off factual questions.\n\n"
                f"Question:\n{req.question}\n\nAnswer:\n{req.answer}\n\n"
                "Return ONLY True or False."
            )}],
            temperature=0.0,
            max_tokens=10,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Filter error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))
    result = resp.choices[0].message.content.strip().lower()
    return {"should_store": result == "true"}


@app.post("/v1/memory/extract")
def memory_extract(
    req:     MemoryExtractRequest,
    user_id: str = Depends(require_active_sub),
):
    """Extract structured facts from a conversation turn."""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": (
                "You are a memory extraction system.\n\n"
                f"USER:\n{req.question}\n\nASSISTANT:\n{req.answer}\n\n"
                "Extract only facts useful long-term.\n"
                "Categories: user_name, favorite_color, favorite_food, "
                "favorite_drink, interests, important_people, pets, projects.\n\n"
                "Return JSON only. Example:\n"
                f'{{"user_name":"{req.user_name}","interests":["coding"]}}\n'
                "If nothing to remember return: {}"
            )}],
            temperature=0.0,
            max_tokens=500,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extract error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))
    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        memories = _json.loads(text)
    except Exception:
        memories = {}
    return {"memories": memories}


@app.post("/v1/memory/embed")
def memory_embed(
    req:     MemoryEmbedRequest,
    user_id: str = Depends(require_active_sub),
):
    """Generate a text embedding vector for semantic memory search."""
    try:
        result = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=req.text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embed error: {exc}")

    _add_usage(user_id, _embedding_cost(result.usage))
    return {"embedding": result.data[0].embedding}


class MemoryUpdateRequest(BaseModel):
    current_memory: dict
    question:       str
    answer:         str


@app.post("/v1/memory/update")
def memory_update(
    req:     MemoryUpdateRequest,
    user_id: str = Depends(require_active_sub),
):
    """Merge the latest interaction into the long-term memory JSON blob."""
    prompt = f"""You maintain Wiebke's long-term memory.

Current memory:

{_json.dumps(req.current_memory, indent=2)}

Latest interaction:

USER:

{req.question}

ASSISTANT:

{req.answer}

Update the memory.

Rules:

1. Preserve memories unless explicitly contradicted.
2. New information replaces older information.
3. Correct obvious spelling mistakes only.
4. Preserve names exactly as written.
5. Merge duplicates.
6. Keep pets separate from people.
7. Learn user preferences.
8. Never invent facts.
9. Never assume someone or a pet is deceased unless explicitly stated.
10. Preserve deceased people or pets if explicitly mentioned.
11. Never store temporary information.

Temporary information includes:

- time
- date
- weather
- news
- sports fixtures
- current events

12. Never create keys such as:

current_time
current_date
weather
news

13. Preserve pronunciations if explicitly stated.
14. Return COMPLETE JSON ONLY.

Preference examples:

User:
"Don't describe the image every time."

Memory:

"preferences":
{{
    "describe_images_by_default": false
}}

User:
"Keep answers short."

Memory:

"preferences":
{{
    "response_length": "short"
}}

User:
"Explain things in more detail."

Memory:

"preferences":
{{
    "technical_level": "high"
}}

User:
"Be more humorous."

Memory:

"preferences":
{{
    "humor_level": "high"
}}

Pet rules:

- Preserve pet names exactly.
- Preserve pronunciation information.
- Preserve whether a pet is alive or deceased if explicitly stated.
- Never change a pet's status unless explicitly told.

Required structure:

{{
    "user_name": "",
    "preferred_form_of_address": "",

    "favorite_color": "",
    "favorite_food": "",
    "favorite_drink": "",

    "user_is_person_in_images": true,

    "user_appearance": "",

    "projects": [],

    "interests": [],

    "important_people": [],

    "pets":
    [
        {{
            "name": "",
            "pronunciation": "",
            "status": "alive"
        }}
    ],

    "preferences":
    {{
        "describe_images_by_default": true,

        "response_length": "medium",

        "technical_level": "high",

        "humor_level": "low"
    }}
}}
"""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory update error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))

    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        updated_memory = _json.loads(text)
        for key in ("current_time", "current_date", "weather", "news"):
            updated_memory.pop(key, None)
        return {"memory": updated_memory}
    except Exception:
        return {"memory": req.current_memory}


class MemoryTagsRequest(BaseModel):
    question: str
    answer:   str


@app.post("/v1/memory/tags")
def memory_tags(
    req:     MemoryTagsRequest,
    user_id: str = Depends(require_active_sub),
):
    """Generate 3-8 short tags describing a conversation turn."""
    prompt = f"""Generate 3 to 8 short tags describing this interaction.

Question:

{req.question}

Answer:

{req.answer}

Return ONLY a JSON array.

Example:

[
    "AI",
    "computer vision",
    "smart glasses",
    "hardware"
]
"""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tag generation error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))

    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        tags = _json.loads(text)
        return {"tags": tags if isinstance(tags, list) else []}
    except Exception:
        return {"tags": []}


class MemoryRerankRequest(BaseModel):
    question: str
    memories: list


@app.post("/v1/memory/rerank")
def memory_rerank(
    req:     MemoryRerankRequest,
    user_id: str = Depends(require_active_sub),
):
    """Pick the 1-3 candidate memories most relevant to the question."""
    if not req.memories:
        return {"selected": []}

    memory_block = ""
    for i, memory in enumerate(req.memories):
        memory_block += f"\nMEMORY {i}\n\n{memory}\n\n--------------------\n"

    prompt = f"""You are selecting memories for an AI assistant.

Question:

{req.question}

Candidate memories:

{memory_block}

Choose the memories that are genuinely useful for answering the question.

Rules:

- Return between 1 and 3 memory numbers.
- Ignore unrelated memories.
- Prefer memories containing facts.
- Prefer memories that help continuity.
- Return ONLY JSON.

Example:

[0,2]
"""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rerank error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))

    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        indexes  = _json.loads(text)
        selected = [req.memories[i] for i in indexes if isinstance(i, int) and 0 <= i < len(req.memories)]
        return {"selected": selected}
    except Exception:
        return {"selected": req.memories[:3]}


class MemoryDuplicateRequest(BaseModel):
    new_memory:        str
    existing_memories: list


@app.post("/v1/memory/duplicate")
def memory_duplicate(
    req:     MemoryDuplicateRequest,
    user_id: str = Depends(require_active_sub),
):
    """Determine whether new_memory duplicates one of existing_memories."""
    if not req.existing_memories:
        return {"is_duplicate": False}

    memories_text = "\n\n".join(req.existing_memories)

    prompt = f"""Determine whether the NEW MEMORY is essentially the same information as one of the EXISTING MEMORIES.

Treat different wording as duplicates.

Examples:

NEW MEMORY:

User enjoys hiking.

EXISTING MEMORY:

User's hobby is hiking.

Result:

True


NEW MEMORY:

User likes science fiction.

EXISTING MEMORY:

User enjoys sci-fi books.

Result:

True


NEW MEMORY:

User owns a bicycle.

EXISTING MEMORY:

User enjoys hiking.

Result:

False


EXISTING MEMORIES:

{memories_text}


NEW MEMORY:

{req.new_memory}


Return ONLY:

True

or

False
"""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Duplicate check error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))
    result = resp.choices[0].message.content.strip().lower()
    return {"is_duplicate": result.startswith("true")}


class GoalParseRequest(BaseModel):
    message: str


@app.post("/v1/goals/parse")
def goals_parse(
    req:     GoalParseRequest,
    user_id: str = Depends(require_active_sub),
):
    """Extract a goal name/category/deadline from a chat message like
    'add a goal to run a 5k by December'. Powers chat-driven goal
    creation — the desktop dispatches straight to this from a matched
    GOAL_CREATION intent, without a full companion-brain round trip."""
    today = datetime.now(timezone.utc).date().isoformat()

    prompt = f"""Extract a goal from this message.

Message:

{req.message}

Today's date is {today}.

Categories (pick the single best match): career, financial, fitness_wellbeing, personal, relationships, learning, other.

Return ONLY JSON in this exact shape:
{{"name": "short goal name", "category": "one of the categories above", "deadline": "YYYY-MM-DD or null"}}

If a relative date is mentioned (e.g. "by December", "in 3 months", "next year"), resolve it to an actual YYYY-MM-DD date based on today's date. If no deadline is mentioned at all, use null for deadline.
"""
    try:
        resp = openai_client.chat.completions.create(
            model=SUPPORT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Goal parse error: {exc}")

    _add_usage(user_id, _chat_cost(SUPPORT_MODEL, resp.usage))

    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = _json.loads(text)
        return {
            "name":     parsed.get("name") or req.message.strip()[:80],
            "category": parsed.get("category") or "other",
            "deadline": parsed.get("deadline"),
        }
    except Exception:
        return {"name": req.message.strip()[:80], "category": "other", "deadline": None}


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
