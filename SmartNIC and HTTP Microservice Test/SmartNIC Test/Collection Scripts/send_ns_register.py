#!/usr/bin/env python3
import os
import re
import subprocess
import textwrap
import time

INTERNAL = ["h101", "h102", "h103", "h104", "h105", "h106"]

def sh(cmd, check=True):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(f"CMD failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()

def find_pid(host: str) -> str:
    # Mininet host namespaces show up as "mininet:h101" etc.
    pid = sh(["bash", "-lc", rf"pgrep -f 'mininet:{host}\b' | head -n 1 || true"], check=False).strip()
    if not pid:
        raise RuntimeError(f"Cannot find PID for {host}. Is Mininet running?")
    return pid

def run_in_ns(pid: str, code: str) -> str:
    # Run python inside the host namespace
    return sh(["sudo", "mnexec", "-a", pid, "python3", "-c", code], check=True)

def main():
    if os.geteuid() != 0:
        print("[!] Run as root: sudo python3 send_ns_register_all.py")
        raise SystemExit(1)

    print("== NS/DAD Registration (learning) for internal hosts ==")
    print("Hosts:", INTERNAL)
    print("Note: This must be run AFTER Mininet is up.\n")

    # Resolve PIDs first
    pids = {}
    for h in INTERNAL:
        pids[h] = find_pid(h)

    # Embedded sender (runs inside each namespace)
    # - auto-detect eth0 MAC + global IPv6
    # - computes solicited-node multicast from last 24 bits
    # - sends NS a few times
    embedded = textwrap.dedent(r"""
        import ipaddress, re, subprocess, time
        from scapy.all import Ether, IPv6, sendp
        from scapy.layers.inet6 import ICMPv6ND_NS, ICMPv6NDOptSrcLLAddr

        IFACE = "eth0"

        def get_mac(iface):
            with open(f"/sys/class/net/{iface}/address", "r") as f:
                return f.read().strip()

        def get_global_ipv6(iface):
            out = subprocess.check_output(["ip", "-6", "addr", "show", "dev", iface], text=True)
            # Prefer global (2001:... etc)
            globals_ = re.findall(r"\s+inet6\s+([0-9a-fA-F:]+)/\d+\s+scope\s+global", out)
            if globals_:
                return globals_[0]
            # Fallback: any inet6 (avoid link-local if possible)
            any_ = re.findall(r"\s+inet6\s+([0-9a-fA-F:]+)/\d+", out)
            if not any_:
                raise RuntimeError(f"No IPv6 address found on {iface}")
            # choose first non-fe80 if exists
            for a in any_:
                if not a.lower().startswith("fe80:"):
                    return a
            return any_[0]

        def solicited_node(ipv6_str):
            addr_int = int(ipaddress.IPv6Address(ipv6_str))
            low24 = addr_int & 0xFFFFFF
            b1 = (low24 >> 16) & 0xFF
            b2 = (low24 >> 8) & 0xFF
            b3 = low24 & 0xFF
            sn_mac = f"33:33:ff:{b1:02x}:{b2:02x}:{b3:02x}"
            # ff02::1:ffXX:XXXX where XX:XXXX are low24 bits
            sn_ip_raw = f"ff02::1:ff{b1:02x}:{(b2<<8 | b3):04x}"
            sn_ip = str(ipaddress.IPv6Address(sn_ip_raw))  # compress nicely
            return sn_ip, sn_mac

        src_ip = get_global_ipv6(IFACE)
        src_mac = get_mac(IFACE)
        sn_ip, sn_mac = solicited_node(src_ip)

        # Add fl=0x12345 (safe even if you don't use it; helps your debug gating if present)
        p = (
            Ether(src=src_mac, dst=sn_mac)
            / IPv6(src=src_ip, dst=sn_ip, hlim=255, fl=0x12345)
            / ICMPv6ND_NS(tgt=src_ip)
            / ICMPv6NDOptSrcLLAddr(lladdr=src_mac)
        )

        # Send multiple times for robustness
        sendp(p, iface=IFACE, count=3, inter=0.05, verbose=False)

        print(f"NS_SENT iface={IFACE} src={src_ip} mac={src_mac} -> {sn_ip} {sn_mac} hlim=255 count=3 fl=0x12345")
    """)

    ok = []
    bad = []

    for h in INTERNAL:
        pid = pids[h]
        try:
            out = run_in_ns(pid, embedded)
            print(f"[OK]  {h} (pid {pid}): {out}")
            ok.append(h)
        except Exception as e:
            print(f"[ERR] {h} (pid {pid}): {e}")
            bad.append(h)
        time.sleep(0.1)

    print("\n== SUMMARY ==")
    print("OK :", ok)
    print("BAD:", bad)
    if bad:
        print("\n[!] Some hosts failed NS registration. Fix those first, then re-run this script.")
        raise SystemExit(2)

    print("\nDone. Now run your enforcement test:")
    print("  sudo python3 test_spoof_then_legit.py --timeout 2.0 --hlim 63 --h0-iface eth0")

if __name__ == "__main__":
    main()
