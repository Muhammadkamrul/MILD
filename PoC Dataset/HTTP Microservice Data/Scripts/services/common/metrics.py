
import time
import threading
from typing import Optional

import psutil
from prometheus_client import Gauge

# System-level KPIs (Table 2 style)
CPU_PCT = Gauge("cpu_pct", "CPU utilization percentage")
RAM_PCT = Gauge("ram_pct", "RAM utilization percentage")
STORAGE_PCT = Gauge("storage_pct", "Storage utilization percentage", ["path"])
NET_BYTES_SENT = Gauge("net_bytes_sent_total", "Total bytes sent (host interface aggregation)")
NET_BYTES_RECV = Gauge("net_bytes_recv_total", "Total bytes received (host interface aggregation)")

def _safe_pct(val: float) -> float:
    if val is None:
        return float("nan")
    return max(0.0, min(100.0, float(val)))

def start_system_metrics_poller(interval_s: float = 2.0, disk_path: str = "/") -> None:
    """
    Periodically updates CPU/RAM/Storage and Network byte counters.
    Expose rates in Prometheus via irate()/rate() in queries.
    """
    def loop():
        # psutil.cpu_percent needs a first call; prime it:
        psutil.cpu_percent(interval=None)
        while True:
            try:
                CPU_PCT.set(_safe_pct(psutil.cpu_percent(interval=None)))
                RAM_PCT.set(_safe_pct(psutil.virtual_memory().percent))
                try:
                    du = psutil.disk_usage(disk_path)
                    STORAGE_PCT.labels(path=disk_path).set(_safe_pct(du.percent))
                except Exception:
                    # container may not have access; skip
                    pass

                try:
                    nio = psutil.net_io_counters(pernic=False)
                    NET_BYTES_SENT.set(float(nio.bytes_sent))
                    NET_BYTES_RECV.set(float(nio.bytes_recv))
                except Exception:
                    pass
            except Exception:
                # never kill the app due to polling errors
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
