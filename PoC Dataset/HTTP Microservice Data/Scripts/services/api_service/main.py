
import os
import time
import asyncio
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from services.common.metrics import start_system_metrics_poller

SERVICE_NAME = os.getenv("SERVICE_NAME", "api-service")
TELEMETRY_URL = os.getenv("TELEMETRY_URL", "http://telemetry-service:8000")
ANALYTICS_URL = os.getenv("ANALYTICS_URL", "http://analytics-service:8000")

app = FastAPI(title=SERVICE_NAME)

# App-specific KPIs (Table 2)
API_LATENCY = Histogram(
    "api_latency_seconds",
    "End-to-end API request latency",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)
REQ_TOTAL = Counter("http_requests_total", "Total HTTP requests", ["route", "method", "code"])
REQ_BYTES_IN = Counter("http_request_bytes_total", "Total request bytes", ["route"])
RESP_BYTES_OUT = Counter("http_response_bytes_total", "Total response bytes", ["route"])
ERROR_TOTAL = Counter("http_errors_total", "Total server-side errors", ["route", "type"])

# Fault knobs
INJECT_DELAY_MS = Gauge("fault_inject_delay_ms", "Injected delay (ms) added to API handler")
ERROR_RATE = Gauge("fault_error_rate", "Probability of forced 500 errors (0..1) in API handler")

# Health / availability proxy (SRI can be derived from success ratio)
SUCCESS_TOTAL = Counter("http_success_total", "Successful HTTP requests", ["route"])

@app.on_event("startup")
async def on_startup():
    start_system_metrics_poller()
    INJECT_DELAY_MS.set(0.0)
    ERROR_RATE.set(0.0)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": SERVICE_NAME}

@app.post("/fault")
async def fault(kind: str, level: float = 0.0, duration_s: int = 0):
    """
    Simple fault injector for drift scenarios.
    kind=delay_ms or error_rate
    - delay_ms: level is ms added to each request
    - error_rate: level in [0,1] probability to return 500
    duration_s > 0 will revert to normal automatically.
    """
    kind = kind.lower()
    if kind == "delay_ms":
        INJECT_DELAY_MS.set(max(0.0, float(level)))
        if duration_s > 0:
            asyncio.create_task(_revert_after(INJECT_DELAY_MS, duration_s, 0.0))
    elif kind == "error_rate":
        ERROR_RATE.set(max(0.0, min(1.0, float(level))))
        if duration_s > 0:
            asyncio.create_task(_revert_after(ERROR_RATE, duration_s, 0.0))
    else:
        return {"ok": False, "error": f"unknown kind={kind}"}
    return {"ok": True, "kind": kind, "level": level, "duration_s": duration_s}

async def _revert_after(gauge: Gauge, duration_s: int, value: float):
    await asyncio.sleep(duration_s)
    gauge.set(value)

@app.post("/api")
async def api_endpoint(request: Request):
    route = "/api"
    method = request.method
    body = await request.body()
    REQ_BYTES_IN.labels(route=route).inc(len(body))

    # Inject faults
    delay_ms = float(INJECT_DELAY_MS._value.get())  # type: ignore[attr-defined]
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000.0)

    import random
    if random.random() < float(ERROR_RATE._value.get()):  # type: ignore[attr-defined]
        ERROR_TOTAL.labels(route=route, type="forced").inc()
        REQ_TOTAL.labels(route=route, method=method, code="500").inc()
        return Response(content="forced error", status_code=500)

    t0 = time.perf_counter()
    code = "200"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 1) send telemetry (fire-and-forget best effort)
            try:
                await client.post(f"{TELEMETRY_URL}/ingest", content=body)
            except Exception:
                ERROR_TOTAL.labels(route=route, type="telemetry_down").inc()

            # 2) ask analytics to process something related
            # (this couples intents to allow co-drift propagation)
            try:
                r = await client.post(f"{ANALYTICS_URL}/process", content=body)
                code = str(r.status_code)
                resp_bytes = len(r.content)
            except Exception:
                ERROR_TOTAL.labels(route=route, type="analytics_down").inc()
                code = "502"
                resp_bytes = 0

        latency = time.perf_counter() - t0
        API_LATENCY.observe(latency)

        if code.startswith("2"):
            SUCCESS_TOTAL.labels(route=route).inc()

        REQ_TOTAL.labels(route=route, method=method, code=code).inc()
        RESP_BYTES_OUT.labels(route=route).inc(resp_bytes)
        return Response(content=b"ok", status_code=int(code) if code.isdigit() else 200)
    except Exception:
        latency = time.perf_counter() - t0
        API_LATENCY.observe(latency)
        ERROR_TOTAL.labels(route=route, type="unhandled").inc()
        REQ_TOTAL.labels(route=route, method=method, code="500").inc()
        return Response(content="server error", status_code=500)
