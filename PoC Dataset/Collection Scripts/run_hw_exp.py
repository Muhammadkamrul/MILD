#!/usr/bin/env python3

import sys
import os
import time
from mininet.link import Intf
import argparse
import inspect

# 1. Add the path to the original run_exercise.py
# based on your previous messages, it seems to be in the tutorials folder
sys.path.append('/home/smartnic/tutorials/utils/')

# 2. Import the original class
try:
    from run_exercise import ExerciseRunner
except ImportError:
    print("Error: Could not find run_exercise.py. Check the sys.path.append line.")
    sys.exit(1)

class SmartNICRunner(ExerciseRunner):
    def run_exercise(self):

        # --- FIX: avoid gRPC port conflict with SmartNIC RTE (uses 50051) ---
        try:
            from p4runtime_switch import P4RuntimeSwitch
            if hasattr(P4RuntimeSwitch, "next_grpc_port"):
                P4RuntimeSwitch.next_grpc_port = 51051
                print("Set BMv2 P4Runtime gRPC base port to 51051 (avoid 50051 conflict).")
        except Exception as e:
            print(f"WARNING: could not set BMv2 gRPC base port: {e}")
        # --------------------------------------------------------------------

        # A) Create nodes from topo_hw.json (but do NOT start yet)
        self.create_network()

        # --- FIX: create "unlinked" hosts (h0, h101-h106) so we can attach VFs ---
        for hn, props in self.hosts.items():
            if hn not in self.net.nameToNode:
                print(f"Adding unlinked host {hn} (will attach VF later)...")
                self.net.addHost(hn, ip=props.get('ip'), mac=props.get('mac'))
        # -------------------------------------------------------------------------

        print("ACTIVATING HARDWARE IN THE LOOP")

        host_map = {
            'h0':   'enp6s0v0',
            'h101': 'enp6s0v1',
            'h102': 'enp6s0v2',
            'h103': 'enp6s0v3',
            'h104': 'enp6s0v4',
            'h105': 'enp6s0v5',
            'h106': 'enp6s0v6'
        }

        # B) Bind host VFs BEFORE net.start()
        for host_name, iface in host_map.items():
            if host_name not in self.net.nameToNode:
                print(f"Warning: {host_name} not found in topology!")
                continue

            host = self.net.get(host_name)
            print(f"Binding {host_name} to {iface}...")

            # --- SAFETY FLUSH (User Tweak A) ---
            # Ensure no stale IPs exist before moving to namespace
            os.system(f"ip addr flush dev {iface} || true")
            os.system(f"ip -6 addr flush dev {iface} || true")
            os.system(f"ip link set {iface} down")
            # -----------------------------------

            intf = Intf(iface, node=host)      # moves VF into host netns

            # rename inside host netns
            host.cmd(f'ip link set {iface} name eth0')
            host.cmd('ip link set eth0 up')

            # update Mininet’s bookkeeping (important)
            try:
                host.nameToIntf.pop(iface, None)
                intf.name = 'eth0'
                host.nameToIntf['eth0'] = intf
                #host.defaultIntf = intf
            except Exception:
                pass

            host.cmd('ethtool --offload eth0 rx off tx off || true')

        # C) Bind VF7 to s3 port 1 BEFORE net.start()
        if 's3' not in self.net.nameToNode:
            raise RuntimeError("Switch s3 not found in topology!")

        print("Binding s3 port 1 to enp6s0v7 (VF7)...")
        s3 = self.net.get('s3')

        # --- SAFETY FLUSH (User Tweak B - CRITICAL) ---
        # VF7 stays in root namespace, so we MUST remove IP to prevent Kernel interference
        os.system("ip addr flush dev enp6s0v7 || true")
        os.system("ip -6 addr flush dev enp6s0v7 || true")
        os.system("ip link set enp6s0v7 down")
        # ----------------------------------------------

        #Intf('enp6s0v7', node=s3, port=1)

        # Bring it back up (L2 only, no IP)
        # Note: We run this command in the root shell because s3 is in root
        #os.system("ip link set enp6s0v7 up")

        Intf('enp6s0v7', node=s3, port=1)
        s3.cmd("ip link set enp6s0v7 up || true")

        # D) Now start the network (BMv2 sees port1 at launch)
        self.net.start()
        time.sleep(1)

        # E) Program (as usual)
        self.program_hosts()
        self.program_switches()

        self.do_net_cli()
        self.net.stop()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--topo', default='topo/topo_hw.json')
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--pcap-dir', default='pcaps')
    # Note: Using underscore for switch_json matches the parent class variable naming style
    ap.add_argument('-j', '--switch_json', default='build/multicast.json')
    # ap.add_argument('-j', '--switch_json', default='build/defense.json')
    # CRITICAL ADDITION: Handle the behavioral executable flag passed by Makefile
    ap.add_argument('-b', '--behavioral-exe', default='simple_switch', help='Path to P4 switch executable')
    ap.add_argument('-q', '--quiet', action='store_true', help='Suppress log messages')
    
    args = ap.parse_args()

    # Pass ALL arguments to the constructor so the parent class (ExerciseRunner) sets up correctly
    # runner = SmartNICRunner(
    #     topo_file=args.topo, 
    #     log_dir=args.log_dir, 
    #     pcap_dir=args.pcap_dir, 
    #     switch_json=args.switch_json, 
    #     bmv2_exe=args.behavioral_exe, 
    #     quiet=args.quiet
    # )
    
    # runner.run_exercise()

    kwargs = dict(
        topo_file=args.topo,
        log_dir=args.log_dir,
        pcap_dir=args.pcap_dir,
        switch_json=args.switch_json,
        bmv2_exe=args.behavioral_exe,
        quiet=args.quiet
    )

    sig = inspect.signature(ExerciseRunner.__init__)
    safe_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    runner = SmartNICRunner(**safe_kwargs)
    runner.run_exercise()