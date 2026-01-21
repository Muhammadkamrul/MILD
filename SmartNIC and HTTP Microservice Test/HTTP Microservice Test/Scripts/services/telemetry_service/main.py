
import os
import time
import asyncio
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from services.common.metrics import start_system_metrics_poller

SERVICE_NAME = os.getenv("SERVICE_NAME", "telemetry-service")
ANALYTICS_URL = os.getenv("ANALYTICS_URL", "http://analytics-service:8000")

app = FastAPI(title=SERVICE_NAME)

# Application-specific KPI: telemetry_queue length
TELEMETRY_QUEUE_LEN = Gauge("telemetry_queue", "Length of telemetry ingestion queue")
TELEMETRY_INGEST_TOTAL = Counter("telemetry_ingest_total", "Total telemetry items ingested")
TELEMETRY_DROPPED_TOTAL = Counter("telemetry_dropped_total", "Total telemetry items dropped")
TELEMETRY_INGEST_LAT = Histogram(
    "telemetry_ingest_latency_seconds",
    "Latency to enqueue telemetry item",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)

# Fault knobs
MAX_QUEUE = Gauge("fault_max_queue", "Max queue size before dropping (0 means unlimited)")
FORWARD_PAUSE = Gauge("fault_forward_pause", "If 1, pause forwarding to analytics")

queue: asyncio.Queue[bytes] = asyncio.Queue()

@app.on_event("startup")
async def on_startup():
    start_system_metrics_poller()
    MAX_QUEUE.set(5000.0)
    FORWARD_PAUSE.set(0.0)
    asyncio.create_task(_forward_loop())

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": SERVICE_NAME}

@app.post("/fault")
async def fault(kind: str, level: float = 0.0, duration_s: int = 0):
    """
    kind=max_queue or forward_pause
    - max_queue: level is max queue length before dropping (set 0 for unlimited)
    - forward_pause: level 1 pauses forwarding, 0 resumes (creates queue build-up)
    """
    kind = kind.lower()
    if kind == "max_queue":
        MAX_QUEUE.set(max(0.0, float(level)))
        if duration_s > 0:
            asyncio.create_task(_revert_after(MAX_QUEUE, duration_s, 5000.0))
    elif kind == "forward_pause":
        FORWARD_PAUSE.set(1.0 if float(level) >= 0.5 else 0.0)
        if duration_s > 0:
            asyncio.create_task(_revert_after(FORWARD_PAUSE, duration_s, 0.0))
    else:
        return {"ok": False, "error": f"unknown kind={kind}"}
    return {"ok": True, "kind": kind, "level": level, "duration_s": duration_s}

async def _revert_after(gauge: Gauge, duration_s: int, value: float):
    await asyncio.sleep(duration_s)
    gauge.set(value)

@app.post("/ingest")
async def ingest(request: Request):
    body = await request.body()
    t0 = time.perf_counter()

    max_q = int(float(MAX_QUEUE._value.get()))  # type: ignore[attr-defined]
    if max_q > 0 and queue.qsize() >= max_q:
        TELEMETRY_DROPPED_TOTAL.inc()
        TELEMETRY_QUEUE_LEN.set(queue.qsize())
        return Response(content="dropped", status_code=429)

    await queue.put(body)
    TELEMETRY_INGEST_TOTAL.inc()
    TELEMETRY_QUEUE_LEN.set(queue.qsize())
    TELEMETRY_INGEST_LAT.observe(time.perf_counter() - t0)
    return {"ok": True, "queued": True, "qsize": queue.qsize()}

async def _forward_loop():
    """
    Forwards telemetry to analytics. Pausing this creates backpressure and telemetry_queue drift.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                if float(FORWARD_PAUSE._value.get()) >= 0.5:  # type: ignore[attr-defined]
                    await asyncio.sleep(0.5)
                    TELEMETRY_QUEUE_LEN.set(queue.qsize())
                    continue

                item = await queue.get()
                TELEMETRY_QUEUE_LEN.set(queue.qsize())
                try:
                    await client.post(f"{ANALYTICS_URL}/process", content=item)
                except Exception:
                    # if analytics is down, requeue a bit (simulates retry buildup)
                    try:
                        await queue.put(item)
                    except Exception:
                        TELEMETRY_DROPPED_TOTAL.inc()
                    await asyncio.sleep(0.2)
            except Exception:
                await asyncio.sleep(0.2)
