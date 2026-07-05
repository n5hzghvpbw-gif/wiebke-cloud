"""
wiebke_cloud — subscription proxy server.

Endpoints:
  POST /auth/register          Create account
  POST /auth/login             Get JWT token
  POST /auth/refresh           Refresh JWT token
  GET  /account/status         Subscription status
  POST /stripe/checkout        Start Stripe checkout
  POST /stripe/portal          Manage subscription
  POST /stripe/webhook         Stripe event handler
  POST /v1/chat                Proxy to OpenAI (requires active sub)
  POST /v1/transcribe          Transcribe audio via Whisper (requires active sub)

Deploy to Railway / Render / Fly.io:
  Required environment variables:
    OPENAI_API_KEY        your OpenAI key (never exposed to users)
    STRIPE_SECRET_KEY     sk_live_...
    STRIPE_WEBHOOK_SECRET whsec_...
    STRIPE_PRICE_ID       price_... (your £9.99/mo product)
    JWT_SECRET            any long random string

  Optional:
    APP_URL               https://yourapp.com  (for Stripe redirects)
    DATABASE_URL          postgresql://...     (Railway Postgres — strongly recommended)
    DB_PATH               /data/wiebke.db      (SQLite path when DATABASE_URL is not set)

  On Railway: add a PostgreSQL service — DATABASE_URL is injected automatically.
  Without DATABASE_URL the server falls back to SQLite, which is wiped on every
  redeploy. Use SQLite only for local development.
"""

import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import stripe
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
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
JWT_ALGORITHM      = "HS256"
ACCESS_TOKEN_TTL   = 60 * 60 * 24 * 30  # 30 days

stripe.api_key = STRIPE_SECRET_KEY
openai_client  = OpenAI(api_key=OPENAI_API_KEY)
pwd_ctx        = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
                id                  TEXT PRIMARY KEY,
                email               TEXT UNIQUE NOT NULL,
                password_hash       TEXT NOT NULL,
                stripe_customer_id  TEXT,
                subscription_status TEXT DEFAULT 'inactive',
                subscription_end    TEXT,
                created_at          TEXT
            )
        """)


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
    pw_hash     = pwd_ctx.hash(req.password)
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

    if not user or not pwd_ctx.verify(req.password, user["password_hash"]):
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
    return {
        "email":               user["email"],
        "subscription_status": user["subscription_status"],
        "subscription_end":    user["subscription_end"],
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
# OpenAI proxy  (requires an active subscription)
# ──────────────────────────────────────────────────────────────────

class ProxyChatRequest(BaseModel):
    messages:    list
    model:       str   = "gpt-4.1-mini"
    temperature: float = 0.85
    max_tokens:  int   = 1024


@app.post("/v1/chat")
def proxy_chat(
    req:     ProxyChatRequest,
    user_id: str = Depends(require_active_sub),
):
    """Forward a messages array to OpenAI on behalf of an active subscriber."""
    try:
        response = openai_client.chat.completions.create(
            model       = req.model,
            messages    = req.messages,
            temperature = req.temperature,
            max_tokens  = req.max_tokens,
        )
        return {"content": response.choices[0].message.content}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {exc}")


class ProxyTranscribeRequest(BaseModel):
    audio_base64: str
    filename:     str = "audio.wav"


@app.post("/v1/transcribe")
def proxy_transcribe(
    req:     ProxyTranscribeRequest,
    user_id: str = Depends(require_active_sub),
):
    """Transcribe audio via Whisper on behalf of an active subscriber."""
    import base64
    import io

    try:
        audio_bytes      = base64.b64decode(req.audio_base64)
        audio_file       = io.BytesIO(audio_bytes)
        audio_file.name  = req.filename
        result = openai_client.audio.transcriptions.create(
            model = "whisper-1",
            file  = audio_file,
        )
        return {"text": result.text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Whisper error: {exc}")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
