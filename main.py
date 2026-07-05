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

Deploy to Railway / Render / Fly.io:
  Set environment variables:
    OPENAI_API_KEY       your OpenAI key (never exposed to users)
    STRIPE_SECRET_KEY    sk_live_...
    STRIPE_WEBHOOK_SECRET whsec_...
    STRIPE_PRICE_ID      price_... (your £9.99/mo product)
    JWT_SECRET           any long random string
    APP_URL              https://yourapp.com (for Stripe redirects)
    DATABASE_URL         (optional) postgresql://...  defaults to SQLite
"""

import hashlib
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import stripe
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
STRIPE_SECRET_KEY   = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SEC  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID     = os.environ.get("STRIPE_PRICE_ID", "")
JWT_SECRET          = os.environ.get("JWT_SECRET", "change-me-in-production")
APP_URL             = os.environ.get("APP_URL", "https://wiebke.app")
JWT_ALGORITHM       = "HS256"
ACCESS_TOKEN_TTL    = 60 * 60 * 24 * 30   # 30 days

stripe.api_key = STRIPE_SECRET_KEY
openai_client  = OpenAI(api_key=OPENAI_API_KEY)
pwd_ctx        = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ──────────────────────────────────────────────
# Database (SQLite; swap for Postgres in prod)
# ──────────────────────────────────────────────

DB_PATH = Path(os.environ.get("DB_PATH", "wiebke_users.db"))

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _db() as c:
        c.execute("""
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
        c.commit()


# ──────────────────────────────────────────────
# JWT helpers
# ──────────────────────────────────────────────

def _make_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + ACCESS_TOKEN_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _verify_token(token: str) -> str:
    """Returns user_id or raises HTTPException."""
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return data["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def _get_user(user_id: str) -> sqlite3.Row | None:
    with _db() as c:
        return c.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()


def _get_user_by_email(email: str) -> sqlite3.Row | None:
    with _db() as c:
        return c.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()


# ──────────────────────────────────────────────
# Auth dependency
# ──────────────────────────────────────────────

def require_active_sub(authorization: str = Header(...)):
    """FastAPI dependency — validates JWT and checks subscription is active."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token   = authorization.split(" ", 1)[1]
    user_id = _verify_token(token)
    user    = _get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    if user["subscription_status"] != "active":
        raise HTTPException(
            status_code=402,
            detail="No active subscription. Visit the Account page to subscribe."
        )
    return user_id


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield

app = FastAPI(title="Wiebke Cloud", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register")
def register(req: RegisterRequest):
    email = req.email.strip().lower()
    if _get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered.")

    user_id = str(uuid.uuid4())
    pw_hash = pwd_ctx.hash(req.password)

   # Create a Stripe customer (optional — skipped if Stripe not configured yet)
    stripe_customer_id = None
    if STRIPE_SECRET_KEY and not STRIPE_SECRET_KEY.startswith("sk_test_placeholder"):
        try:
            stripe_customer = stripe.Customer.create(email=email)
            stripe_customer_id = stripe_customer.id
        except Exception:
            pass

    with _db() as c:
        c.execute(
            """INSERT INTO users
               (id, email, password_hash, stripe_customer_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, email, pw_hash,
             stripe_customer_id,
             datetime.now(timezone.utc).isoformat())
        )
        c.commit()

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
def refresh_token(user_id: str = Depends(lambda auth=Header(...): _verify_token(auth.split(" ", 1)[-1]))):
    return {"token": _make_token(user_id)}


# ──────────────────────────────────────────────
# Account
# ──────────────────────────────────────────────

@app.get("/account/status")
def account_status(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    user_id = _verify_token(authorization.split(" ", 1)[1])
    user    = _get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "email":               user["email"],
        "subscription_status": user["subscription_status"],
        "subscription_end":    user["subscription_end"],
    }


# ──────────────────────────────────────────────
# Stripe
# ──────────────────────────────────────────────

@app.post("/stripe/checkout")
def create_checkout(authorization: str = Header(...)):
    """Returns a Stripe Checkout URL to start a subscription."""
    user_id = _verify_token(authorization.split(" ", 1)[1])
    user    = _get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    session = stripe.checkout.Session.create(
        customer=user["stripe_customer_id"],
        payment_method_types=["card"],
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        mode="subscription",
        success_url=f"{APP_URL}/subscribe/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{APP_URL}/subscribe/cancel",
        metadata={"user_id": user_id},
    )
    return {"checkout_url": session.url}


@app.post("/stripe/portal")
def customer_portal(authorization: str = Header(...)):
    """Returns a Stripe Customer Portal URL for managing billing."""
    user_id = _verify_token(authorization.split(" ", 1)[1])
    user    = _get_user(user_id)
    if not user or not user["stripe_customer_id"]:
        raise HTTPException(status_code=404, detail="No billing account found.")

    session = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{APP_URL}/account",
    )
    return {"portal_url": session.url}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe events (subscription activated, cancelled, etc.)."""
    payload = await request.body()
    sig     = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SEC)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature.")

    def _update(customer_id: str, status: str, end_date: str | None = None):
        with _db() as c:
            c.execute(
                """UPDATE users
                   SET subscription_status=?, subscription_end=?
                   WHERE stripe_customer_id=?""",
                (status, end_date, customer_id)
            )
            c.commit()

    if event["type"] == "customer.subscription.created":
        sub = event["data"]["object"]
        _update(sub["customer"], "active")

    elif event["type"] == "customer.subscription.updated":
        sub    = event["data"]["object"]
        status = "active" if sub["status"] == "active" else "inactive"
        end    = datetime.fromtimestamp(
            sub["current_period_end"], tz=timezone.utc
        ).isoformat() if sub.get("current_period_end") else None
        _update(sub["customer"], status, end)

    elif event["type"] in (
        "customer.subscription.deleted",
        "customer.subscription.paused",
    ):
        sub = event["data"]["object"]
        _update(sub["customer"], "cancelled")

    elif event["type"] == "invoice.payment_failed":
        inv = event["data"]["object"]
        _update(inv["customer"], "past_due")

    return {"received": True}


# ──────────────────────────────────────────────
# OpenAI proxy — the core value
# ──────────────────────────────────────────────

class ProxyChatRequest(BaseModel):
    messages: list
    model:    str = "gpt-4.1-mini"
    temperature: float = 0.85
    max_tokens:  int = 1024


@app.post("/v1/chat")
def proxy_chat(req: ProxyChatRequest, user_id: str = Depends(require_active_sub)):
    """Forward a prepared messages array to OpenAI on behalf of a subscriber."""
    try:
        response = openai_client.chat.completions.create(
            model=req.model,
            messages=req.messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return {"content": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")


class ProxyTranscribeRequest(BaseModel):
    audio_base64: str
    filename:     str = "audio.wav"


@app.post("/v1/transcribe")
def proxy_transcribe(req: ProxyTranscribeRequest, user_id: str = Depends(require_active_sub)):
    """Transcribe audio via Whisper on behalf of a subscriber."""
    import base64, io
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        audio_file  = io.BytesIO(audio_bytes)
        audio_file.name = req.filename
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return {"text": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
