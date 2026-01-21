#!/bin/bash
set -x

# h0 -> VF0
sudo ip link set enp6s0v0 down
sudo ip link set enp6s0v0 address 02:01:01:01:00:01
sudo ip addr flush dev enp6s0v0
sudo ip -6 neigh flush dev enp6s0v0
sudo ip link set enp6s0v0 up

# External link -> VF7 (keep MAC if your SmartNIC expects it)
sudo ip link set enp6s0v7 down
sudo ip link set enp6s0v7 address 02:01:01:01:00:6c
sudo ip addr flush dev enp6s0v7
sudo ip -6 neigh flush dev enp6s0v7
sudo ip link set enp6s0v7 up

# h101..h106 -> VF1..VF6 MUST be 66..6b
sudo ip link set enp6s0v1 down
sudo ip link set enp6s0v1 address 02:01:01:01:00:66
sudo ip addr flush dev enp6s0v1
sudo ip -6 neigh flush dev enp6s0v1
sudo ip link set enp6s0v1 up

sudo ip link set enp6s0v2 down
sudo ip link set enp6s0v2 address 02:01:01:01:00:67
sudo ip addr flush dev enp6s0v2
sudo ip -6 neigh flush dev enp6s0v2
sudo ip link set enp6s0v2 up

sudo ip link set enp6s0v3 down
sudo ip link set enp6s0v3 address 02:01:01:01:00:68
sudo ip addr flush dev enp6s0v3
sudo ip -6 neigh flush dev enp6s0v3
sudo ip link set enp6s0v3 up

sudo ip link set enp6s0v4 down
sudo ip link set enp6s0v4 address 02:01:01:01:00:69
sudo ip addr flush dev enp6s0v4
sudo ip -6 neigh flush dev enp6s0v4
sudo ip link set enp6s0v4 up

sudo ip link set enp6s0v5 down
sudo ip link set enp6s0v5 address 02:01:01:01:00:6a
sudo ip addr flush dev enp6s0v5
sudo ip -6 neigh flush dev enp6s0v5
sudo ip link set enp6s0v5 up

sudo ip link set enp6s0v6 down
sudo ip link set enp6s0v6 address 02:01:01:01:00:6b
sudo ip addr flush dev enp6s0v6
sudo ip -6 neigh flush dev enp6s0v6
sudo ip link set enp6s0v6 up

echo "VFs Configured with Manual MACs!"