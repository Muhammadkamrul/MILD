
# HTTP KPI Dataset Generator (Table-2 style KPIs)

- 3 HTTP microservices that mimic our paper’s three intents:
  - **api-service** (API intent)
  - **telemetry-service** (Telemetry intent)
  - **analytics-service** (Analytics intent)
- Prometheus scraping for KPI time series
- A load generator + a scenario runner to produce **normal** and **abnormal/drift** windows
- A dataset exporter that pulls **Table-2 KPIs** from Prometheus and writes **CSV/Parquet**

## 1) Start the stack
```bash
docker compose up --build
```

Ports:
- API: http://localhost:8000
- Telemetry: http://localhost:8001
- Analytics: http://localhost:8002
- Prometheus: http://localhost:9090

## 2) Generate traffic
Normal traffic:
```bash
python tools/loadgen.py --mode normal --duration 300 --url http://localhost:8000/api
```

Run an end-to-end labeled scenario (normal -> fault -> cooldown):
```bash
python tools/run_scenario.py --fault analytics_cpu --normal_s 300 --fault_s 300 --cooldown_s 120
```

Faults we inject:
- `api_delay` (increases api_latency)
- `api_error` (availability drop -> SRI drift)
- `telemetry_pause` (queue build-up -> telemetry_queue drift)
- `analytics_cpu` / `analytics_sleep` / `analytics_memleak` (throughput + CPU/RAM drifts, cascades to API/Telemetry)

## 3) Export dataset (Table-2 KPIs)
Pull last 20 minutes at 1-minute sampling:
```bash
python tools/collector.py --minutes 20 --step 60 --out dataset.parquet --label drift_cpu
```

## Notes on KPIs mapping (Table 2)
Base KPIs are produced as Prometheus metrics and then queried:
- `cpu_pct`, `ram_pct`, `storage_pct`: exported by each service via `psutil`
- `snet_bps`: derived from byte counters (`net_bytes_*`) using PromQL `rate()`
- `sri`: availability proxy = success-rate / request-rate
- `api_latency_p95_ms`: p95 from `api_latency_seconds` histogram
- `telemetry_queue`: queue gauge from telemetry-service
- `analytics_tput_rps`: processing throughput from analytics counter rate

This can be used to:
- compute rolling-window features (mean/std over 5/15 min) later in Python,
- store separate labels for **fault type**, **root cause**, etc.
