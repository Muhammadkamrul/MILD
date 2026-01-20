"""
This script generates a JSON file that describes the topology of the network.
"""
import json

OUTPUT_FILE = "./topo/topo.json"
NUM_HOSTS = 107
NUM_CONSUMER_SWITCHES = 6
CONSUMER_SWITCH_NUMBERS = (1, 2, 8, 9, 10, 11)
NUM_ISP_SWITCHES = 5
ISP_SWITCH_NUMBERS = (3, 4, 5, 6, 7)
NUM_HOSTS_PER_SWITCH = 20
VICTIM_NETWORK = ("2001:db8:1::", 48)
NETWORKS = (
    ("2001:db8:2::", 48),
    ("2001:db9:1::", 48),
    ("2001:dba:0:100::", 56),
    ("2001:dbb:0:100::", 56),
    ("2001:dbc:0:1::", 64),
)

def generate_hosts(num_hosts, networks):
    hosts = {}
    for host_number in range(num_hosts):
        # Set network based on host number
        # index 0 -> host 1-20, index 1 -> host 21-40, etc.
        # host 0, 101-106 are in the victim network
        if host_number == 0 or (101 <= host_number <= 106):
            network_address, network_prefix = VICTIM_NETWORK
            in_network_host_number = host_number % 100 + 1
        else:
            network_address, network_prefix = networks[(host_number - 1) // NUM_HOSTS_PER_SWITCH]
            in_network_host_number = (host_number - 1) % NUM_HOSTS_PER_SWITCH + 1
            
        ipv6 = f"{network_address}{in_network_host_number:02x}"
        ipv4 = f"10.0.{host_number // 256}.{host_number % 256}"
        mac = f"02:01:01:01:{host_number // 256:02x}:{host_number % 256 + 1:02x}"
        hosts[f"h{host_number}"] = {
            "ip": ipv4,
            "mac": mac,
            "commands": [
                "sudo sysctl -w net.ipv6.conf.all.disable_ipv6=0", # Enable IPv6
                "sudo sysctl -w net.ipv6.conf.all.accept_ra=0", # Disable IPv6 router advertisements
                "sudo sysctl -w net.ipv6.conf.default.accept_ra=0",
                "sudo sysctl -w net.ipv6.conf.all.router_solicitations=0",
                "sudo sysctl -w net.ipv6.conf.default.router_solicitations=0",
                f"sudo ip -6 addr add {ipv6}/{network_prefix} dev eth0", # Assign IP address
            ]
        }

        # for hosts 101-106, assign tentative ipv6 address using nodad
        # add a normal ipv6 address for other hosts
        # if 101 <= host_number <= 106:
        #     hosts[f"h{host_number}"]["commands"].append(f"sudo ip -6 addr add {ipv6}/{network_prefix} dev eth0 nodad")
        # else:
        #     hosts[f"h{host_number}"]["commands"].append(f"sudo ip -6 addr add {ipv6}/{network_prefix} dev eth0")

        if host_number != 0:
            # for hosts other than h0, add a neighbor entry to host 0
            hosts[f"h{host_number}"]["commands"].append(f"sudo ip -6 neigh add 2001:db8:1::1 lladdr 02:01:01:01:00:01 dev eth0")
            # for host h0, add a neighbor entry to the to all other hosts
            hosts["h0"]["commands"].append(f"sudo ip -6 neigh add {ipv6} lladdr {mac} dev eth0")

    return hosts

def generate_switches(num_switches):
    switches = {}
    for i in range(1, num_switches + 1):
        switches[f"s{i}"] = {
            "runtime_json": f"topo/s{i}-runtime.json"
        }
    return switches

def generate_links(num_hosts_per_switch):
    # Generate links from hosts to consumer switches except for the victim network (switch 1)
    # p1 to p5 will be reserved for switch-to-switch links, so p6 to p25 will be used for host-to-switch links
    links = []
    host_number = 0
    for switch_number in CONSUMER_SWITCH_NUMBERS:
        # Links for the victim network
        if switch_number == 1:
            links.append([f"h0", f"s1-p6"])
            links.append([f"h101", f"s1-p7"])
            links.append([f"h102", f"s1-p8"])
            links.append([f"h103", f"s1-p9"])
            links.append([f"h104", f"s1-p10"])
            links.append([f"h105", f"s1-p11"])
            links.append([f"h106", f"s1-p12"])
            continue
        # Links for other networks
        for i in range(1, num_hosts_per_switch + 1):
            host_number += 1
            links.append([f"h{host_number}", f"s{switch_number}-p{i + 5}"])

    # Generate links from consumer switches to ISP switches
    consumer_to_isp_map = {
        1: [3],
        2: [3],
        8: [4],
        9: [5],
        10: [6],
        11: [7],
    }

    used_ports = {switch: 1 for switch in CONSUMER_SWITCH_NUMBERS + ISP_SWITCH_NUMBERS}
    for consumer_switch, isp_switches in consumer_to_isp_map.items():
        for isp_switch in isp_switches:
            consumer_port = used_ports[consumer_switch]
            isp_port = used_ports[isp_switch]
            links.append([f"s{consumer_switch}-p{consumer_port}", f"s{isp_switch}-p{isp_port}"])
            used_ports[consumer_switch] += 1
            used_ports[isp_switch] += 1

    # Generate links between ISP switches
    isp_to_isp_map = {
        3: [4],
        4: [5],
        5: [6],
        6: [7],
    }

    for isp_switch1, connected_isp_switches in isp_to_isp_map.items():
        for isp_switch2 in connected_isp_switches:
            isp_port_1 = used_ports[isp_switch1]
            isp_port_2 = used_ports[isp_switch2]
            links.append([f"s{isp_switch1}-p{isp_port_1}", f"s{isp_switch2}-p{isp_port_2}"])
            used_ports[isp_switch1] += 1
            used_ports[isp_switch2] += 1

    return links


def main():
    topo = {
        "hosts": generate_hosts(NUM_HOSTS, NETWORKS),
        "switches": generate_switches(NUM_CONSUMER_SWITCHES + NUM_ISP_SWITCHES),
        "links": generate_links(NUM_HOSTS_PER_SWITCH)
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(topo, f, indent=4)

if __name__ == "__main__":
    main()