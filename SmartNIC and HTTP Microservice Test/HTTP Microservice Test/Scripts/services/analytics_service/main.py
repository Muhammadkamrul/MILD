
import os
import time
import asyncio
import hashlib
from typing import Optional

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from services.common.metrics import start_system_metrics_poller

SERVICE_NAME = os.getenv("SERVICE_NAME", "analytics-service")
app = FastAPI(title=SERVICE_NAME)

# Application-specific KPI: analytics throughput
ANALYTICS_TPUT_TOTAL = Counter("analytics_processed_total", "Total processed items")
ANALYTICS_PROC_LAT = Histogram(
    "analytics_processing_latency_seconds",
    "Time to process an item",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

# Fault knobs
CPU_BURN = Gauge("fault_cpu_burn", "CPU burn intensity (0..1) per request")
SLEEP_MS = Gauge("fault_sleep_ms", "Sleep added to processing path (ms)")
MEM_LEAK_MB_PER_REQ = Gauge("fault_mem_leak_mb_per_req", "MB allocated and kept per request")
ERROR_RATE = Gauge("fault_error_rate_analytics", "Probability of forced 500 errors (0..1)")

_leak_store = []

@app.on_event("startup")
async def on_startup():
    start_system_metrics_poller()
    CPU_BURN.set(0.0)
    SLEEP_MS.set(0.0)
    MEM_LEAK_MB_PER_REQ.set(0.0)
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
    kind=cpu_burn|sleep_ms|mem_leak|error_rate
    - cpu_burn: level in [0,1], burns CPU with busy-loop
    - sleep_ms: level ms sleep
    - mem_leak: level MB allocated and kept per request
    - error_rate: level in [0,1]
    duration_s reverts to normal automatically.
    """
    kind = kind.lower()
    if kind == "cpu_burn":
        CPU_BURN.set(max(0.0, min(1.0, float(level))))
        if duration_s > 0:
            asyncio.create_task(_revert_after(CPU_BURN, duration_s, 0.0))
    elif kind == "sleep_ms":
        SLEEP_MS.set(max(0.0, float(level)))
        if duration_s > 0:
            asyncio.create_task(_revert_after(SLEEP_MS, duration_s, 0.0))
    elif kind == "mem_leak":
        MEM_LEAK_MB_PER_REQ.set(max(0.0, float(level)))
        if duration_s > 0:
            asyncio.create_task(_revert_after(MEM_LEAK_MB_PER_REQ, duration_s, 0.0))
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

def _burn_cpu(intensity: float, base_ms: int = 20):
    """
    Busy-loop for roughly base_ms*intensity milliseconds.
    """
    if intensity <= 0:
        return
    burn_s = (base_ms * intensity) / 1000.0
    end = time.perf_counter() + burn_s
    x = 0
    while time.perf_counter() < end:
        x ^= 0xABCDEF
        x = (x << 1) & 0xFFFFFFFF

@app.post("/process")
async def process(request: Request):
    import random
    if random.random() < float(ERROR_RATE._value.get()):  # type: ignore[attr-defined]
        return Response(content="forced error", status_code=500)

    body = await request.body()
    t0 = time.perf_counter()

    # processing: hash payload (cheap) + optional CPU burn
    _ = hashlib.sha256(body).hexdigest()
    _burn_cpu(float(CPU_BURN._value.get()))  # type: ignore[attr-defined]

    sleep_ms = float(SLEEP_MS._value.get())  # type: ignore[attr-defined]
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000.0)

    leak_mb = float(MEM_LEAK_MB_PER_REQ._value.get())  # type: ignore[attr-defined]
    if leak_mb > 0:
        # allocate and keep
        _leak_store.append(bytearray(int(leak_mb * 1024 * 1024)))

    ANALYTICS_TPUT_TOTAL.inc()
    ANALYTICS_PROC_LAT.observe(time.perf_counter() - t0)
    return {"ok": True}
