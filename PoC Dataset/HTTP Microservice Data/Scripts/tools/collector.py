
import argparse
import datetime as dt
import time
from typing import Dict, Any, List

import pandas as pd
import requests

PROM = "http://localhost:9090"

# PromQL helpers (Table 2 KPIs)
QUERIES = {
    # System
    "cpu_pct": 'avg(cpu_pct{job="api"})',   # you can change job=... or avg across jobs
    "ram_pct": 'avg(ram_pct{job="api"})',
    "storage_pct": 'avg(storage_pct{job="api"})',
    # Service network throughput (bytes/sec), using Prometheus rate
    "snet_bps": '8 * (rate(net_bytes_sent_total{job="api"}[30s]) + rate(net_bytes_recv_total{job="api"}[30s]))',
    # Availability proxy (SRI): success ratio over requests (rate-based)
    "sri": '(rate(http_success_total{job="api"}[1m]) / clamp_min(rate(http_requests_total{job="api"}[1m]), 1e-9))',
    # App
    "api_latency_p95_s": 'histogram_quantile(0.95, sum(rate(api_latency_seconds_bucket{job="api"}[1m])) by (le))',
    "telemetry_queue": 'avg(telemetry_queue{job="telemetry"})',
    "analytics_tput_rps": 'rate(analytics_processed_total{job="analytics"}[1m])',
}

def qrange(query: str, start: dt.datetime, end: dt.datetime, step_s: int) -> pd.DataFrame:
    url = f"{PROM}/api/v1/query_range"
    params = {
        "query": query,
        "start": start.timestamp(),
        "end": end.timestamp(),
        "step": step_s,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data["status"] != "success":
        raise RuntimeError(data)
    result = data["data"]["result"]
    if not result:
        return pd.DataFrame(columns=["ts", "value"])
    # take first series by default (or you can aggregate in PromQL)
    values = result[0]["values"]
    df = pd.DataFrame(values, columns=["ts", "value"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=20, help="How far back to pull")
    ap.add_argument("--step", type=int, default=60, help="Sampling step in seconds (dataset interval)")
    ap.add_argument("--out", default="dataset.parquet")
    ap.add_argument("--label", default="normal", help="label for this window (e.g., normal, drift_cpu, drift_queue)")
    args = ap.parse_args()

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(minutes=args.minutes)

    series = []
    for name, q in QUERIES.items():
        df = qrange(q, start, end, args.step).rename(columns={"value": name})
        series.append(df[["ts", name]])

    # merge on timestamp
    merged = series[0]
    for df in series[1:]:
        merged = pd.merge_asof(
            merged.sort_values("ts"),
            df.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=args.step//2 + 1),
        )

    merged["label"] = args.label
    merged["is_abnormal"] = (merged["label"] != "normal").astype(int)

    # convenience conversions
    merged["api_latency_p95_ms"] = merged["api_latency_p95_s"] * 1000.0

    out_path = args.out
    if out_path.endswith(".csv"):
        merged.to_csv(out_path, index=False)
    else:
        merged.to_parquet(out_path, index=False)

    print(f"Wrote {len(merged)} rows to {out_path}")
    print("Columns:", list(merged.columns))

if __name__ == "__main__":
    main()
