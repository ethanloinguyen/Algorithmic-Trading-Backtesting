# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.routers.model import router as model_router
from app.routers import stocks, indices, portfolio, pairs, montecarlo

settings = get_settings()

app = FastAPI(
    title="LagLens API",
    description="Stock OHLCV data API backed by Google Cloud BigQuery.",
    version="1.0.0",
)

# Safely combine settings origins with local development origins
custom_origins = getattr(settings, "origins_list", [])
if isinstance(custom_origins, str):
    custom_origins = [custom_origins]

origins = list(set(custom_origins + [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]))

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Matches all Vercel preview deployments for this project automatically
    allow_origin_regex=r"https://algorithmic-trading-backtesting[^.]*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(stocks.router)
app.include_router(indices.router)
app.include_router(model_router)
app.include_router(portfolio.router)
app.include_router(pairs.router)
app.include_router(montecarlo.router)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})