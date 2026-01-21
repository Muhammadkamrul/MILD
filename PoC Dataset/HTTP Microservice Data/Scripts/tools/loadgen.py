
import argparse
import asyncio
import os
import random
import time
from dataclasses import dataclass

import httpx

def now_ms() -> int:
    return int(time.time() * 1000)

@dataclass
class Profile:
    rps: float
    payload_bytes: int
    concurrency: int

async def worker(client: httpx.AsyncClient, url: str, payload_bytes: int, stop_at: float, stats: dict):
    payload = os.urandom(payload_bytes)
    while time.time() < stop_at:
        try:
            t0 = time.perf_counter()
            r = await client.post(url, content=payload)
            dt = time.perf_counter() - t0
            stats["count"] += 1
            stats["lat_sum"] += dt
            if r.status_code >= 400:
                stats["err"] += 1
        except Exception:
            stats["err"] += 1

async def run(profile: Profile, duration_s: int, url: str):
    stats = {"count": 0, "err": 0, "lat_sum": 0.0}
    # token bucket pacing across workers
    interval = 1.0 / max(0.1, profile.rps)
    stop_at = time.time() + duration_s

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for _ in range(profile.concurrency):
            tasks.append(asyncio.create_task(worker(client, url, profile.payload_bytes, stop_at, stats)))

        # pace overall by sleeping; workers loop fast but http latency limits them.
        while time.time() < stop_at:
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks, return_exceptions=True)

    avg_lat = (stats["lat_sum"] / stats["count"]) if stats["count"] else 0.0
    print(f"sent={stats['count']} err={stats['err']} avg_lat_ms={avg_lat*1000:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/api")
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--mode", choices=["normal", "surge", "heavy_payload"], default="normal")
    args = ap.parse_args()

    if args.mode == "normal":
        profile = Profile(rps=20, payload_bytes=256, concurrency=20)
    elif args.mode == "surge":
        profile = Profile(rps=200, payload_bytes=256, concurrency=200)
    else:  # heavy_payload
        profile = Profile(rps=20, payload_bytes=64 * 1024, concurrency=40)

    asyncio.run(run(profile, args.duration, args.url))

if __name__ == "__main__":
    main()
