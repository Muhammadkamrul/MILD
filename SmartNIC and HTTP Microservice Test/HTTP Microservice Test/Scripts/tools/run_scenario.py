
import argparse
import subprocess
import time
import requests

def post(url, **params):
    r = requests.post(url, params=params, timeout=5)
    r.raise_for_status()
    return r.json()

def run_load(mode, duration):
    cmd = ["python", "tools/loadgen.py", "--mode", mode, "--duration", str(duration), "--url", "http://localhost:8000/api"]
    subprocess.run(cmd, check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_s", type=int, default=300)
    ap.add_argument("--fault_s", type=int, default=300)
    ap.add_argument("--cooldown_s", type=int, default=120)
    ap.add_argument("--fault", choices=["api_delay", "api_error", "telemetry_pause", "analytics_cpu", "analytics_sleep", "analytics_memleak"], default="analytics_cpu")
    args = ap.parse_args()

    print("Phase A: normal traffic")
    run_load("normal", args.normal_s)

    print(f"Phase B: inject fault={args.fault}")
    if args.fault == "api_delay":
        post("http://localhost:8000/fault", kind="delay_ms", level=150, duration_s=args.fault_s)
    elif args.fault == "api_error":
        post("http://localhost:8000/fault", kind="error_rate", level=0.2, duration_s=args.fault_s)
    elif args.fault == "telemetry_pause":
        post("http://localhost:8001/fault", kind="forward_pause", level=1, duration_s=args.fault_s)
    elif args.fault == "analytics_cpu":
        post("http://localhost:8002/fault", kind="cpu_burn", level=1.0, duration_s=args.fault_s)
    elif args.fault == "analytics_sleep":
        post("http://localhost:8002/fault", kind="sleep_ms", level=120, duration_s=args.fault_s)
    elif args.fault == "analytics_memleak":
        post("http://localhost:8002/fault", kind="mem_leak", level=5, duration_s=args.fault_s)

    # while fault active, use same workload (or use surge for co-drift)
    run_load("normal", args.fault_s)

    print("Phase C: cooldown (normal)")
    run_load("normal", args.cooldown_s)

    print("Done. Now run tools/collector.py to export a dataset from Prometheus.")

if __name__ == "__main__":
    main()
