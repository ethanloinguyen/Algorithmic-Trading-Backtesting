# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.routers import stocks, indices

settings = get_settings()

app = FastAPI(
    title="LagLens API",
    description="Stock OHLCV data API backed by Google Cloud BigQuery.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Content-Type"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(stocks.router)
app.include_router(indices.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )