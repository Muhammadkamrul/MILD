#!/usr/bin/env python3
import json
import os
import sys
import time
import subprocess
import threading
import re
import csv
import signal
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ----------------
TOPO_FILE = "/home/smartnic/Desktop/IPv6_Security/GitHub_version_last_NEW_exp/P4_Defense/topo/topo_hw.json"

OUTPUT_CSV = "rebuttal_data_subtracted.csv"
OUTPUT_PLOT = "rebuttal_plot_subtracted.png"
OUTPUT_EVENTS = "events_by_intent.json"
PCAP_PATH = f"/tmp/h0_rebuttal_poc_{int(time.time())}.pcap"

# Timings
PHASE_1_NORMAL_DUR = 60
PHASE_2_ATTACK_DUR = 60

TOTAL_DUR = PHASE_1_NORMAL_DUR + PHASE_2_ATTACK_DUR
ATTACK_START_T = PHASE_1_NORMAL_DUR

# Intent thresholds (tune if needed)
API_LATENCY_THRESHOLD_MS = 525.0      # default; set to what makes sense for your environment
API_LOSS_THRESHOLD_PCT = 20.0        # default
SEC_DROP_THRESHOLD_PER_S = 120       # your existing threshold
ANALYTICS_TPUT_MIN_PPS = 50          # your existing threshold for "perf intent" (optional)

# SmartNIC Config
RTECLI = "/opt/netronome/p4/bin/rtecli"
RTE_PORT = "20206"
DROP_REG = "drop_counter_reg"
DROP_IDX = 0

# Hosts
VICTIM_HOST = "h0"
LOAD_HOSTS = ["h1", "h2", "h3", "h4", "h5"]
PROBE_HOST = "h6"
ATTACK_HOSTS = ["h81", "h82", "h83"]

VICTIM_IFACE = "eth0"  # adjust if needed

# ---------------- OPTIONAL SYSTEM METRICS ----------------
try:
    import psutil  # type: ignore
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False

def get_system_stats():
    """
    Returns (cpu_pct, ram_pct, storage_pct). If not available, returns zeros.
    """
    if not HAVE_PSUTIL:
        return 0.0, 0.0, 0.0
    try:
        cpu = float(psutil.cpu_percent(interval=None))
        ram = float(psutil.virtual_memory().percent)
        disk = float(psutil.disk_usage("/").percent)
        return cpu, ram, disk
    except Exception:
        return 0.0, 0.0, 0.0

# ---------------- HELPERS ----------------
def sh(cmd, check=True):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"CMD failed: {' '.join(cmd)}\nRC={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p

def get_mininet_pid(host_name):
    cmd = f"pgrep -f \"mininet:{host_name}\\b\" | head -n 1 || true"
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return out if out else None

def parse_ipv6_from_topo(topo, host):
    cmds = topo["hosts"][host].get("commands", [])
    for c in cmds:
        m = re.search(r"ip\s+-6\s+addr\s+add\s+([0-9a-fA-F:]+)/", c)
        if m:
            return m.group(1)
    return None

def read_smartnic_drops_raw():
    """
    Read raw drop register value. Returns 0 if read fails.
    """
    try:
        cmd = [RTECLI, "-p", RTE_PORT, "-j", "registers", "get", "-r", DROP_REG]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        arr = json.loads(out)
        return int(arr[DROP_IDX], 16)
    except Exception:
        return 0

def counter_delta(curr, prev, width_bits=16):
    """
    Robust delta with wrap-around.
    """
    mod = 1 << width_bits
    if curr >= prev:
        return curr - prev
    return (curr + mod) - prev

# ---------------- TRAFFIC GENERATION ----------------
def write_udp_loader(path="/tmp/_udp_loader.py"):
    code = r"""#!/usr/bin/env python3
import time
import argparse
from scapy.all import Ether, IPv6, UDP, sendp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-mac", required=True)
    parser.add_argument("--dst-mac", required=True)
    parser.add_argument("--src-ip", required=True)
    parser.add_argument("--dst-ip", required=True)
    args = parser.parse_args()

    p = (Ether(src=args.src_mac, dst=args.dst_mac) /
         IPv6(src=args.src_ip, dst=args.dst_ip, hlim=64) /
         UDP(dport=8080, sport=1234) /
         ("MILD_PAYLOAD"*5))

    while True:
        sendp(p, iface="eth0", verbose=0, count=5)
        time.sleep(0.05)

if __name__ == "__main__":
    main()
"""
    Path(path).write_text(code)
    os.chmod(path, 0o755)
    return path

def write_single_probe(path="/tmp/_send_probe.py"):
    # IMPORTANT: seq must be 0..65535
    code = r"""#!/usr/bin/env python3
import argparse
from scapy.all import Ether, IPv6, sendp
from scapy.layers.inet6 import ICMPv6EchoRequest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-mac", required=True)
    parser.add_argument("--dst-mac", required=True)
    parser.add_argument("--src-ip", required=True)
    parser.add_argument("--dst-ip", required=True)
    parser.add_argument("--seq", type=int, required=True)
    args = parser.parse_args()

    seq16 = args.seq & 0xFFFF
    p = (Ether(src=args.src_mac, dst=args.dst_mac) /
         IPv6(src=args.src_ip, dst=args.dst_ip, hlim=64) /
         ICMPv6EchoRequest(id=0xBEEF, seq=seq16))

    sendp(p, iface="eth0", verbose=0)

if __name__ == "__main__":
    main()
"""
    Path(path).write_text(code)
    os.chmod(path, 0o755)
    return path

def write_attacker_flooder(path="/tmp/_attacker.py"):
    code = r"""#!/usr/bin/env python3
import time
import random
import argparse
from scapy.all import Ether, IPv6, sendp
from scapy.layers.inet6 import ICMPv6EchoRequest

SPOOF_IPS = ["2001:db8::1", "2001:db8:3::99", "2001:db8:2::55"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst-mac", required=True)
    parser.add_argument("--dst-ip", required=True)
    args = parser.parse_args()

    while True:
        src_ip = random.choice(SPOOF_IPS)
        p = (Ether(dst=args.dst_mac) /
             IPv6(src=src_ip, dst=args.dst_ip, hlim=64) /
             ICMPv6EchoRequest(id=0x0BAD, seq=1))
        sendp(p, iface="eth0", verbose=0, count=10)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
"""
    Path(path).write_text(code)
    os.chmod(path, 0o755)
    return path

# ---------------- METRIC COLLECTION (threads) ----------------
rx_throughput_lock = threading.Lock()
rx_throughput_count = 0

sent_probes = {}
sent_probes_lock = threading.Lock()

latency_results = []
latency_results_lock = threading.Lock()

stop_event = threading.Event()

def sniffer_throughput(victim_pid: str):
    """
    Counts UDP packets arriving at victim on port 8080.
    """
    global rx_throughput_count
    cmd = ["mnexec", "-a", victim_pid, "tcpdump", "-i", VICTIM_IFACE, "-n", "-l", "udp port 8080"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        while not stop_event.is_set():
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                break
            with rx_throughput_lock:
                rx_throughput_count += 1
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

def sniffer_latency(victim_pid: str):
    """
    Captures ONLY probe ICMP Echo Requests by filtering on ICMP ID=0xBEEF (48879).
    """
    # BPF: echo request (type 128) and ICMP id == 0xBEEF
    bpf = "icmp6 and ip6[40] == 128 and icmp6[4:2] == 0xbeef"
    cmd = ["mnexec", "-a", victim_pid, "tcpdump", "-i", VICTIM_IFACE, "-n", "-l", "-tt", bpf]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    # tcpdump line looks like:
    # 1768914011.123456 IP6 ... ICMP6, echo request, id 48879, seq 42, length 8
    regex = re.compile(r"^(\d+\.\d+).*echo request, id\s+(\d+),\s+seq\s+(\d+),")

    try:
        while not stop_event.is_set():
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                break
            m = regex.search(line)
            if not m:
                continue

            recv_ts = float(m.group(1))
            icmp_id = int(m.group(2))
            seq = int(m.group(3)) & 0xFFFF

            # safety: only our probe id
            if icmp_id != 48879:
                continue

            with sent_probes_lock:
                if seq in sent_probes:
                    send_ts = sent_probes.pop(seq)
                else:
                    continue

            delay = (recv_ts - send_ts) * 1000.0
            if 0 < delay < 2000:
                with latency_results_lock:
                    latency_results.append(delay)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

def start_pcap_capture(victim_pid: str, duration_s: int, pcap_path: str):
    """
    Records a pcap for evidence and pcap_kb tracking.
    """
    bpf = "udp port 8080 or (icmp6 and ip6[40]==128)"
    cmd = ["mnexec", "-a", victim_pid, "timeout", str(duration_s + 5),
           "tcpdump", "-i", VICTIM_IFACE, "-n", "-w", pcap_path, bpf]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------------- JSON (events) ----------------
def build_events(records):
    """
    Build events_by_intent.json compatible with MILD pipeline:
      { "security":[{start,failure_time,end,type},...], "api":[...], "analytics":[...] }
    """
    events = {"security": [], "api": [], "analytics": []}

    # security: first time bin_security==1
    sec_times = [r["t"] for r in records if r["bin_security"] == 1]
    if sec_times:
        fs = int(sec_times[0])
        events["security"].append({
            "start": int(ATTACK_START_T),
            "failure_time": fs,
            "end": int(TOTAL_DUR - 1),
            "type": "security_spoofing"
        })

    # api: first time bin_api==1
    api_times = [r["t"] for r in records if r["bin_api"] == 1]
    if api_times:
        fa = int(api_times[0])
        events["api"].append({
            "start": max(0, fa - 5),
            "failure_time": fa,
            "end": int(TOTAL_DUR - 1),
            "type": "api_latency_or_loss"
        })

    # analytics: optional; based on analytics_tput threshold (not required by your requested CSV)
    an_times = [r["t"] for r in records if r["analytics_tput"] < ANALYTICS_TPUT_MIN_PPS]
    if an_times:
        fan = int(an_times[0])
        events["analytics"].append({
            "start": max(0, fan - 5),
            "failure_time": fan,
            "end": int(TOTAL_DUR - 1),
            "type": "analytics_throughput_drop"
        })

    return events

def compute_ttf(records, key_bin, default_fail_t):
    """
    Adds TTF fields given a binary column.
    """
    fail_times = [r["t"] for r in records if r[key_bin] == 1]
    first_fail = int(fail_times[0]) if fail_times else int(default_fail_t)
    return first_fail

# ---------------- MAIN ----------------
def main():
    if os.geteuid() != 0:
        sys.exit("Run as sudo.")

    print(f"[*] Loading {TOPO_FILE}...")
    with open(TOPO_FILE) as f:
        topo = json.load(f)

    victim_ip = parse_ipv6_from_topo(topo, VICTIM_HOST)
    victim_mac = topo["hosts"][VICTIM_HOST]["mac"]
    victim_pid = get_mininet_pid(VICTIM_HOST)

    if not victim_pid:
        sys.exit(f"[ERROR] Could not find PID for {VICTIM_HOST}. Is Mininet running?")
    if not victim_ip or not victim_mac:
        sys.exit("[ERROR] Missing victim IPv6 or MAC in topo.")

    print(f"[*] Victim: {VICTIM_HOST} PID={victim_pid} IP={victim_ip} IFACE={VICTIM_IFACE}")
    print(f"[*] Output CSV: {OUTPUT_CSV}")
    print(f"[*] Output plot: {OUTPUT_PLOT}")
    print(f"[*] Output events: {OUTPUT_EVENTS}")
    print(f"[*] PCAP: {PCAP_PATH}")

    procs = []  # track background processes for cleanup

    def cleanup():
        stop_event.set()
        # terminate background procs we started
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(0.2)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass

    def sig_handler(signum, frame):
        print("\n[!] Caught signal, stopping...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Start UDP Sink on victim
    print(f"[*] Starting UDP Sink (Netcat) on {VICTIM_HOST}...")
    p_nc = subprocess.Popen(
        ["mnexec", "-a", victim_pid, "nc", "-u", "-l", "-k", "8080"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    procs.append(p_nc)

    # Prepare Scripts
    udp_script = write_udp_loader()
    probe_script = write_single_probe()
    flood_script = write_attacker_flooder()

    # Start PCAP recorder
    p_pcap = start_pcap_capture(victim_pid, TOTAL_DUR, PCAP_PATH)
    procs.append(p_pcap)

    # Start sniffers (threads)
    t_tput = threading.Thread(target=sniffer_throughput, args=(victim_pid,), daemon=True)
    t_lat = threading.Thread(target=sniffer_latency, args=(victim_pid,), daemon=True)
    t_tput.start()
    t_lat.start()

    # Start Load
    print(f"[*] Starting Load on {len(LOAD_HOSTS)} hosts...")
    for h in LOAD_HOSTS:
        pid = get_mininet_pid(h)
        if not pid:
            print(f"[WARN] Missing PID for {h}, skipping loader.")
            continue
        src_ip = parse_ipv6_from_topo(topo, h)
        src_mac = topo["hosts"][h]["mac"]
        if not src_ip or not src_mac:
            print(f"[WARN] Missing IP/MAC for {h}, skipping loader.")
            continue
        p = subprocess.Popen(
            ["mnexec", "-a", pid, "python3", udp_script,
             "--src-mac", src_mac, "--dst-mac", victim_mac,
             "--src-ip", src_ip, "--dst-ip", victim_ip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append(p)

    # Probe setup
    probe_pid = get_mininet_pid(PROBE_HOST)
    if not probe_pid:
        cleanup()
        sys.exit(f"[ERROR] Cannot find PID for probe host {PROBE_HOST}.")
    probe_ip = parse_ipv6_from_topo(topo, PROBE_HOST)
    probe_mac = topo["hosts"][PROBE_HOST]["mac"]
    if not probe_ip or not probe_mac:
        cleanup()
        sys.exit("[ERROR] Missing probe IP/MAC in topo.")

    # --- CALIBRATION ---
    print("[*] Calibrating Noise Floor (5s)...")
    last_drops_raw = read_smartnic_drops_raw()
    noise_samples = []
    for _ in range(5):
        time.sleep(1)
        curr = read_smartnic_drops_raw()
        delta = counter_delta(curr, last_drops_raw)
        noise_samples.append(delta)
        last_drops_raw = curr

    noise_floor = max(noise_samples) if noise_samples else 0
    print(f"[*] Calibration Done. Noise floor set to {noise_floor} drops/sec.\n")

    print("[ Phase 1 ] Normal Operation...")

    # ---- MAIN LOOP ----
    records = []
    start_time = time.time()

    last_rx = 0
    attack_launched = False

    probe_seq = 1
    benign_send_fails_total = 0
    attack_send_fails_total = 0

    prev_cpu = 0.0

    try:
        for i in range(TOTAL_DUR):
            loop_start = time.time()
            rel_time = int(i)  # integer seconds as "t"

            phase = "normal"
            if i >= PHASE_1_NORMAL_DUR:
                phase = "attack"
                if not attack_launched:
                    print("\n[!!!] LAUNCHING ATTACK [!!!]")
                    attack_launched = True
                    for h in ATTACK_HOSTS:
                        pid = get_mininet_pid(h)
                        if not pid:
                            print(f"[WARN] Missing PID for attacker {h}, skipping.")
                            continue
                        p = subprocess.Popen(
                            ["mnexec", "-a", pid, "python3", flood_script,
                             "--dst-mac", victim_mac, "--dst-ip", victim_ip],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        procs.append(p)

            # 1) Probe send (benign_tx = 1)
            benign_tx = 1
            benign_send_fails = 0

            seq16 = probe_seq & 0xFFFF
            t_send = time.time()
            with sent_probes_lock:
                sent_probes[seq16] = t_send

            pr = subprocess.run(
                ["mnexec", "-a", probe_pid, "python3", probe_script,
                 "--src-mac", probe_mac, "--dst-mac", victim_mac,
                 "--src-ip", probe_ip, "--dst-ip", victim_ip,
                 "--seq", str(seq16)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if pr.returncode != 0:
                benign_send_fails = 1
                benign_send_fails_total += 1

            probe_seq = (probe_seq + 1) & 0xFFFF
            if probe_seq == 0:
                probe_seq = 1

            # 2) SmartNIC drop counter (raw and delta)
            curr_drops_raw = read_smartnic_drops_raw()
            delta_drops_raw = counter_delta(curr_drops_raw, last_drops_raw, width_bits=16)
            last_drops_raw = curr_drops_raw

            # subtract noise floor
            sri_delta = max(0, int(delta_drops_raw) - int(noise_floor))
            sri = int(curr_drops_raw)

            # 3) Throughput (pps)
            with rx_throughput_lock:
                curr_rx = rx_throughput_count
            delta_rx = int(curr_rx - last_rx)
            last_rx = curr_rx

            snet = delta_rx  # keep it non-zero and meaningful
            analytics_tput = delta_rx  # analytics throughput proxy (pps)

            # 4) wait a bit for latency sniffer to catch the probe
            time.sleep(0.35)

            # 5) latency and benign_rx/loss
            api_latency = 0.0
            benign_rx = 0
            with latency_results_lock:
                if latency_results:
                    benign_rx = len(latency_results)
                    api_latency = float(np.mean(latency_results))
                    latency_results.clear()
                else:
                    benign_rx = 0
                    api_latency = records[-1]["api_latency"] if records else 0.0

            loss_pct = 0.0
            if benign_tx > 0:
                loss_pct = max(0.0, (benign_tx - min(benign_rx, benign_tx)) * 100.0 / benign_tx)

            # 6) cpu/ram/storage + deltas
            cpu_pct, ram_pct, storage_pct = get_system_stats()
            cpu_delta = float(cpu_pct - prev_cpu) if records else 0.0
            prev_cpu = cpu_pct

            telemetry_queue = 0.0  # not collected here (set to 0 per your requirement)

            # pcap_kb
            try:
                pcap_kb = int(Path(PCAP_PATH).stat().st_size / 1024)
            except Exception:
                pcap_kb = 0

            # 7) Labels
            bin_security = 1 if (sri_delta > SEC_DROP_THRESHOLD_PER_S) else 0

            # API violation: latency or loss
            bin_api = 1 if (api_latency > API_LATENCY_THRESHOLD_MS or loss_pct > API_LOSS_THRESHOLD_PCT) else 0

            # Attack tx: you launch background flooders; exact pps unknown here
            attack_tx = len(ATTACK_HOSTS) if phase == "attack" else 0
            attack_send_fails = 0  # not tracked for flooders (set 0)
            cause = "security_spoofing" if phase == "attack" else "none"

            epoch_s = int(time.time())

            # progress line
            if i % 1 == 0:
                print(
                    f"T={rel_time:4d}s [{phase.upper():6}] "
                    f"Lat={api_latency:7.2f}ms | RXpps={delta_rx:5d} | "
                    f"sri_delta={sri_delta:6d} | loss={loss_pct:5.1f}% | pcap={pcap_kb:4d}KB"
                )

            records.append({
                # REQUIRED COLUMN NAMES (exact):
                "t": rel_time,
                "cpu_pct": float(cpu_pct) if cpu_pct is not None else 0.0,
                "ram_pct": float(ram_pct) if ram_pct is not None else 0.0,
                "storage_pct": float(storage_pct) if storage_pct is not None else 0.0,
                "snet": float(snet),
                "sri": float(sri),
                "api_latency": float(api_latency),
                "analytics_tput": float(analytics_tput),
                "telemetry_queue": float(telemetry_queue),
                "cpu_delta": float(cpu_delta),
                "sri_delta": float(sri_delta),

                # extras requested:
                "epoch_s": epoch_s,
                "phase": phase,
                "benign_tx": int(benign_tx),
                "benign_rx": int(min(benign_rx, benign_tx)),  # cap to sent count
                "loss_pct": float(loss_pct),
                "benign_send_fails": int(benign_send_fails),
                "attack_tx": int(attack_tx),
                "attack_send_fails": int(attack_send_fails),
                "pcap_kb": int(pcap_kb),
                "bin_api": int(bin_api),
                "bin_security": int(bin_security),
                "cause": cause,
                # ttf filled later
                "ttf_api_s": 0,
                "ttf_security_s": 0,
            })

            # keep 1 Hz loop
            loop_dur = time.time() - loop_start
            if loop_dur < 1.0:
                time.sleep(1.0 - loop_dur)

    except KeyboardInterrupt:
        print("\nStopping...")

    # ---- Post-process TTF ----
    first_security_fail = compute_ttf(records, "bin_security", ATTACK_START_T)
    first_api_fail = compute_ttf(records, "bin_api", ATTACK_START_T)

    for r in records:
        r["ttf_security_s"] = int(max(0, first_security_fail - r["t"]))
        r["ttf_api_s"] = int(max(0, first_api_fail - r["t"]))

    # ---- Save CSV with exact column order ----
    cols = [
        "t","cpu_pct","ram_pct","storage_pct","snet","sri","api_latency","analytics_tput","telemetry_queue",
        "cpu_delta","sri_delta","epoch_s","phase","benign_tx","benign_rx","loss_pct","benign_send_fails",
        "attack_tx","attack_send_fails","pcap_kb","bin_api","bin_security","cause","ttf_api_s","ttf_security_s"
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            # ensure any missing is 0
            for c in cols:
                if c not in r or r[c] is None:
                    r[c] = 0
            w.writerow({c: r[c] for c in cols})

    # ---- Save events_by_intent.json ----
    events = build_events(records)
    with open(OUTPUT_EVENTS, "w") as f:
        json.dump(events, f, indent=2)

    # ---- Plot (same style as your script) ----
    ts = [r["t"] for r in records]
    lat = [r["api_latency"] for r in records]
    tput = [r["analytics_tput"] for r in records]
    drops = [r["sri_delta"] for r in records]  # subtracted drops/sec

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(ts, lat, color="green", label="API Latency")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Intent 1: API Performance")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, tput, color="blue", label="Throughput")
    ax2.set_ylabel("PPS")
    ax2.set_title("Intent 2: Analytics Performance")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax3.plot(ts, drops, color="red", label="Security Drops (Subtracted)")
    ax3.set_ylabel("Drops / sec")
    ax3.set_title("Intent 3: Security (Spoofing Drops)")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Time (s)")

    for ax in (ax1, ax2, ax3):
        ax.axvline(x=PHASE_1_NORMAL_DUR, color="black", linestyle="--", label="Attack Start")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)

    print(f"\n[OK] Data saved to {OUTPUT_CSV}")
    print(f"[OK] Plot saved to {OUTPUT_PLOT}")
    print(f"[OK] Events saved to {OUTPUT_EVENTS}")
    print(f"[OK] PCAP saved to {PCAP_PATH}")

    # Cleanup background processes
    stop_event.set()
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(0.2)
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass

if __name__ == "__main__":
    main()
