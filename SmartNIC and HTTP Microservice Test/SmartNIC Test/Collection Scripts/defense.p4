/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

/*************************************************************************
************************* P A R A M E T E R S  ***************************
*************************************************************************/

#define BUCKET_COUNT 1024


#define TIME_THRESHOLD 1000000 // if the inter-packet interval is less than this threshold, increment the CMS (1 second in microseconds)
#define UNICAST_COUNT_THRESHOLD 50
#define MULTICAST_COUNT_THRESHOLD 10
#define EXTERNAL_FLOODING_THRESHOLD 110 // if the CMS value for the prefix is greater than this threshold, block the prefix (100 packets per 5 seconds per prefix)
#define TIME_WINDOW 5000000 // the duration after which the CMS and window start time are reset (5 seconds in microseconds)

typedef bit<17> duration_t; // duration type (17-bit)
typedef bit<48> timestamp_t; // timestamp type (48-bit)
typedef bit<32> counter_t; // counter type for CMS (32-bit)
typedef bit<16> drop_counter_t; // drop counter type (16-bit)

// define parameters for the experiments
#define ATTACKER_MAC 0x020101010001

#define TYPE_IPV6 0x86DD
#define ICMPV6 0x3A
#define ICMPV6_RS 133
#define ICMPV6_RA 134
#define ICMPV6_NS 135
#define ICMPV6_NA 136

#define TYPE_TCP 6

typedef bit<10> idx_t;   // because BUCKET_COUNT = 1024


/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<16>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<128> ip6Addr_t;
typedef bit<32> ip4Addr_t;

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv6_t {
    bit<4> version;
    bit<8> trafficClass;
    bit<20> flowLabel;
    bit<16> payloadLen;
    bit<8> nextHdr;
    bit<8> hopLimit;
    ip6Addr_t srcAddr;
    ip6Addr_t dstAddr;
}

header icmpv6_t {
    bit<8> type;
    bit<8> code;
    bit<16> checksum;
    bit<32> body;
}

header icmpv6_ns_t {
    bit<128> targetAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4> dataOffset;
    bit<6> reserved;
    bit<6> flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header tcp_option_end_t {
    bit<8> kind;
}

header tcp_option_ts_t {
    bit<8> kind;
    bit<8> length;
    bit<32> tsVal;
    bit<32> tsEcr;
}

header tcp_option_padding_t {
    bit<8> padding;
}

struct network_t {
    ip6Addr_t network_addr;
    bit<8> prefix_len;
}

struct metadata {
    bit<1> dbg_pkt;
    bit<1> dbg_clear;
    bool should_drop;
    network_t current_network;
}

struct headers {
    ethernet_t   ethernet;
    ipv6_t ipv6;
    icmpv6_t icmpv6;
    icmpv6_ns_t icmpv6_ns;
    tcp_t tcp;
    tcp_option_end_t tcp_option_end;
    tcp_option_ts_t tcp_option_ts;
    tcp_option_padding_t tcp_option_padding;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV6 : parse_ipv6;
            default : accept;
        }
    }

    state parse_ipv6 {
        packet.extract(hdr.ipv6);
        transition select(hdr.ipv6.nextHdr) {
            ICMPV6 : parse_icmpv6;
            TYPE_TCP : parse_tcp;
            default : accept;
        }
    }

    state parse_icmpv6 {
        packet.extract(hdr.icmpv6);
        transition select(hdr.icmpv6.type) {
            ICMPV6_NS : parse_icmpv6_ns;
            default : accept;
        }
    }

    state parse_icmpv6_ns {
        packet.extract(hdr.icmpv6_ns);
        transition accept;
    }

    // Note: this parser only parses the TCP header and the options that are used in the experiments
    // it will not work in production environments
    state parse_tcp {
        packet.extract(hdr.tcp);
        packet.extract(hdr.tcp_option_ts);
        packet.extract(hdr.tcp_option_end);
        packet.extract(hdr.tcp_option_padding);
        transition accept;
    }
}


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

register<bit<32>>(1) debug_hash1_reg;
register<bit<32>>(1) debug_hash2_reg;
register<bit<32>>(1) debug_hash3_reg;

register<bit<32>>(1) debug_bf1_reg;
register<bit<32>>(1) debug_bf2_reg;
register<bit<32>>(1) debug_bf3_reg;

register<bit<32>>(1) debug_portval_reg;
register<bit<32>>(1) debug_ingress_reg;
register<bit<32>>(1) debug_status_reg;

register<bit<32>>(1) debug_flowlabel_reg;


// ---- Global drop reason + external flooding evidence ----
register<bit<32>>(1) debug_drop_reason_reg;

register<bit<32>>(1) debug_ext_ingress_reg;
register<bit<32>>(1) debug_ext_skip_reg;

register<bit<32>>(1) debug_ext_plen_reg;
register<bit<32>>(1) debug_ext_net_hi_reg;   // top 32 bits of network_addr
register<bit<32>>(1) debug_ext_net_lo_reg;   // low 32 bits of network_addr

register<bit<32>>(1) debug_ext_hash1_reg;
register<bit<32>>(1) debug_ext_hash2_reg;
register<bit<32>>(1) debug_ext_hash3_reg;

register<bit<32>>(1) debug_ext_fw1_reg;
register<bit<32>>(1) debug_ext_fw2_reg;
register<bit<32>>(1) debug_ext_fw3_reg;

register<bit<32>>(1) debug_ext_winreset_reg;

register<bit<32>>(1) debug_ext_wstart1_lo_reg;
register<bit<32>>(1) debug_ext_wstart2_lo_reg;
register<bit<32>>(1) debug_ext_wstart3_lo_reg;

register<bit<32>>(1) debug_ext_last1_lo_reg;
register<bit<32>>(1) debug_ext_last2_lo_reg;
register<bit<32>>(1) debug_ext_last3_lo_reg;

register<bit<32>>(1) debug_ext_dt1_reg;
register<bit<32>>(1) debug_ext_dt2_reg;
register<bit<32>>(1) debug_ext_dt3_reg;

register<bit<32>>(1) debug_ext_cms1_reg;
register<bit<32>>(1) debug_ext_cms2_reg;
register<bit<32>>(1) debug_ext_cms3_reg;
register<bit<32>>(1) debug_ext_min_reg;

register<bit<32>>(1) debug_ext_tsval_reg;  // tcp tsval if present, else 0
register<bit<32>>(1) debug_ext_stage_reg;  // simple breadcrumbs

register<bit<32>>(1) debug_ext_now_lo_reg;
register<bit<32>>(1) debug_ext_now_hi_reg; // upper 16 bits are enough but store 32 for convenience

register<bit<32>>(1) debug_ext_wstart1_hi_reg;
register<bit<32>>(1) debug_ext_last1_hi_reg;





// counter registers (Not used in the current implementation)
register<bit<32>>(1) icmpv6_counter_reg;
register<bit<32>>(1) ns_counter_reg;
register<bit<32>>(1) na_counter_reg;
register<bit<32>>(1) ra_counter_reg;
register<bit<32>>(1) rs_counter_reg;

// data structures for defense mechanism
register<counter_t>(BUCKET_COUNT) cms0;
register<counter_t>(BUCKET_COUNT) cms1;
register<counter_t>(BUCKET_COUNT) cms2;
register<timestamp_t>(BUCKET_COUNT) timestamp0;
register<timestamp_t>(BUCKET_COUNT) timestamp1;
register<timestamp_t>(BUCKET_COUNT) timestamp2;
register<timestamp_t>(BUCKET_COUNT) window_start_time0;
register<timestamp_t>(BUCKET_COUNT) window_start_time1;
register<timestamp_t>(BUCKET_COUNT) window_start_time2;
//register<bit<1>>(BUCKET_COUNT) multicast_reg; // Maps hashes to a bit (1 if multicast, 0 if not)
//register<bit<1>>(BUCKET_COUNT) first_window_set; // Indicates if the first time window has been set (1 if set, 0 if not)

//register<bit<1>>(BUCKET_COUNT) multicast_reg; // Maps hashes to a bit (1 if multicast, 0 if not)
register<bit<32>>(BUCKET_COUNT) multicast_reg;      // store 0/1
//register<bit<1>>(BUCKET_COUNT) first_window_set; // Indicates if the first time window has been set (1 if set, 0 if not)
register<bit<32>>(BUCKET_COUNT) first_window_set;   // store 0/1

register<duration_t>(85000) ingress_packet_processing_durations_reg;
register<bit<32>>(1) last_duration_index_reg;

// drop counters: 0 for other TCP hop count (external), 1 for other TCP suspicious counter (external),
// 2 for src addr port pair drop (internal), 3 for other TCP external flooding (external),
// 4 for NS internal unicast flooding (internal), 5 for internal multicast flooding (internal)
// 6 for other internal unicast flooding (internal), 7 for benign TCP external flooding (external)
// 8 for benign TCP internal unicast flooding (internal), 9 for benign TCP hop count (external)
// 10 for benign TCP suspicious counter (external)
register<drop_counter_t>(11) drop_counter_reg;

//register<bit<9>>(BUCKET_COUNT) ip_port_reg;
//register<bit<1>>(BUCKET_COUNT) ns_tgt_addr_bf_reg;

register<bit<32>>(BUCKET_COUNT) ip_port_reg;        // store ingress_port as 32-bit
register<bit<32>>(BUCKET_COUNT) ns_tgt_addr_bf_reg; // store 0/1


action increment_cms(in bit<32> hash1, in bit<32> hash2, in bit<32> hash3) {
    bit<32> val1;
    bit<32> val2;
    bit<32> val3;

    cms0.read(val1, hash1);
    cms1.read(val2, hash2);
    cms2.read(val3, hash3);

    cms0.write(hash1, val1 + 1);
    cms1.write(hash2, val2 + 1);
    cms2.write(hash3, val3 + 1);
}

action update_timestamps(in bit<32> hash1, in bit<32> hash2, in bit<32> hash3, in timestamp_t current_time) {
    timestamp0.write(hash1, current_time);
    timestamp1.write(hash2, current_time);
    timestamp2.write(hash3, current_time);
}

action increment_icmpv6_counter(inout metadata meta) {
    bit<32> icmpv6_counter_val;
    icmpv6_counter_reg.read(icmpv6_counter_val, 0);
    icmpv6_counter_val = icmpv6_counter_val + 1;
    icmpv6_counter_reg.write(0, icmpv6_counter_val);
}

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    action multicast() {
        standard_metadata.mcast_grp = 1;
        hdr.ipv6.hopLimit = hdr.ipv6.hopLimit - 1;
    }

    action mac_forward(egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ipv6.hopLimit = hdr.ipv6.hopLimit - 1;
    }

    table mac_lookup {
        key = {
            hdr.ethernet.dstAddr : exact;
        }
        actions = {
            multicast;
            mac_forward;
        }
        size = 1024;
        default_action = multicast;
    }


    action jenkins_hash(in bit<32> key, out bit<32> hash_result) {
            bit<32> tmp;
            
            // 5 arguments: 
            // 1. Output (tmp)
            // 2. Algorithm (crc32)
            // 3. Base/Offset (0)
            // 4. Input Data ({key})
            // 5. Max Value (BUCKET_COUNT) -> The hash will automatically be modulo this value
            hash(tmp, HashAlgorithm.crc32, 32w0, { key }, (bit<32>)BUCKET_COUNT);
            //hash(tmp, HashAlgorithm.csum16, 32w0, { key }, (bit<32>)BUCKET_COUNT);
            // Cast the result to your index type (10-bit)
            // No manual '& (BUCKET_COUNT - 1)' is needed because the hash() primitive 
            // already restricted the value to be < BUCKET_COUNT.
            hash_result = tmp; 
        }
    

    action get_hash_inputs_addr(in bit<128> src_addr, out bit<32> input1, out bit<32> input2, out bit<32> input3) {
        bit<32> chunk1 = (bit<32>)src_addr; // first (rightmost) 32 bits of src_addr
        bit<32> chunk2 = (bit<32>)(src_addr >> 32); // the next 32 bits of src_addr
        bit<32> chunk3 = (bit<32>)(src_addr >> 64); // the next 32 bits of src_addr
        bit<32> chunk4 = (bit<32>)(src_addr >> 96); // the last (leftmost) 32 bits of src_addr

        input1 = chunk1 ^ chunk2 ^ chunk3 ^ chunk4;
        input2 = chunk2 ^ (chunk3 << 8) ^ (chunk4 >> 8);
        input3 = chunk3 ^ (chunk4 << 16) ^ (chunk1 >> 16);
    }

    action get_hash_inputs_flow(in bit<128> src_addr, in bit<128> dst_addr, out bit<32> input1, out bit<32> input2, out bit<32> input3) {
        bit<32> chunk1 = (bit<32>)src_addr; // first (rightmost) 32 bits of src_addr
        bit<32> chunk2 = (bit<32>)(src_addr >> 32); // the next 32 bits of src_addr
        bit<32> chunk3 = (bit<32>)(dst_addr); // first (rightmost) 32 bits of dst_addr
        bit<32> chunk4 = (bit<32>)(dst_addr >> 32); // the next 32 bits of dst_addr

        input1 = chunk1 ^ chunk2 ^ chunk3 ^ chunk4;
        input2 = chunk2 ^ (chunk3 << 8) ^ (chunk4 >> 8);
        input3 = chunk3 ^ (chunk4 << 16) ^ (chunk1 >> 16);
    }

    action get_hash_inputs_prefix(in bit<128> network_addr, in bit<8> prefix_len, out bit<32> input1, out bit<32> input2, out bit<32> input3) {
        bit<32> chunk1 = (bit<32>)network_addr; // first (rightmost) 32 bits of network_addr
        bit<32> chunk2 = (bit<32>)(network_addr >> 32); // the next 32 bits of network_addr
        bit<32> chunk3 = (bit<32>)(network_addr >> 64); // the next 32 bits of network_addr
        bit<32> chunk4 = (bit<32>)(network_addr >> 96); // the last (leftmost) 32 bits of network_addr

        input1 = chunk1 ^ chunk2 ^ chunk3 ^ chunk4 ^ (bit<32>)prefix_len;
        input2 = chunk2 ^ (chunk3 << 8) ^ (chunk4 >> 8) ^ (bit<32>)prefix_len;
        input3 = chunk3 ^ (chunk4 << 16) ^ (chunk1 >> 16) ^ (bit<32>)prefix_len;
    }

    action hop_count_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata); 
        meta.should_drop = true;
    }

    action hop_count_allow() {
        // Explicit permit action for SmartNIC (instead of NoAction)
        meta.should_drop = false;
    }


    action hop_count_benign_drop() {
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)9);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)9, drop_counter_value);
    }

    action hop_count_other_drop() {
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)0);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)0, drop_counter_value);
    }

    action internal_port_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)2);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)2, drop_counter_value);
    }

    action external_flood_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)3);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)3, drop_counter_value);
    }

    action external_benign_flood_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)7);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)7, drop_counter_value);
    }

    action internal_other_unicast_flood_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)6);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)6, drop_counter_value);
    }

    action internal_ns_unicast_flood_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)4);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)4, drop_counter_value);
    }

    action internal_tcp_benign_unicast_flood_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)8);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)8, drop_counter_value);
    }

    action internal_multicast_flood_drop() {
        // simulate the drop action by sending to port 200
        //standard_metadata.egress_spec = 200;
        // mark_to_drop(standard_metadata);
        meta.should_drop = true;
        drop_counter_t drop_counter_value;
        drop_counter_reg.read(drop_counter_value, (bit<32>)5);
        drop_counter_value = drop_counter_value + 1;
        drop_counter_reg.write((bit<32>)5, drop_counter_value);
    }

    action extract_network(ip6Addr_t network_addr, bit<8> prefix_len) {
        meta.current_network.network_addr = network_addr;
        meta.current_network.prefix_len = prefix_len;
    }

    action existing_networks_default() {
        // Explicit default action for SmartNIC (instead of NoAction)
        meta.current_network.network_addr = 0;
        meta.current_network.prefix_len = 0;
    }
    
    /* ===== New: split by prefix length ===== */

    /* /64 table */
    table existing_networks_64 {
        key = {
            hdr.ipv6.srcAddr: ternary;
        }
        actions = {
            extract_network;
            existing_networks_default;
        }
        size = 256;
        default_action = existing_networks_default;
    }

    /* /56 table */
    table existing_networks_56 {
        key = {
            hdr.ipv6.srcAddr: ternary;
        }
        actions = {
            extract_network;
            existing_networks_default;
        }
        size = 256;
        default_action = existing_networks_default;
    }

    /* /48 table */
    table existing_networks_48 {
        key = {
            hdr.ipv6.srcAddr: ternary;
        }
        actions = {
            extract_network;
            existing_networks_default;
        }
        size = 256;
        default_action = existing_networks_default;
    }

    /* Keep existing table as the /32 fallback */
    table existing_networks_32 {
        key = {
            hdr.ipv6.srcAddr: ternary;
        }
        actions = {
            extract_network;
            existing_networks_default;
        }
        size = 256;
        default_action = existing_networks_default;
    }


    table hop_count {
        key = {
            meta.current_network.network_addr: exact;
            meta.current_network.prefix_len: exact;
            hdr.ipv6.hopLimit : exact;
        }
        actions = {
            hop_count_drop;
            hop_count_allow;
        }
        size = 256;
        default_action = hop_count_drop;
    }

    apply {

        //// Preparation ////

        
        
        meta.should_drop = false;

        bit<32> hash_input1 = 0;
        bit<32> hash_input2 = 0;
        bit<32> hash_input3 = 0;

        bit<32> hash1_i = 0;
        bit<32> hash2_i = 0;
        bit<32> hash3_i = 0;

        // note: above here, hash* is 10 bit while hash*_i is 32 bit

        bool skip_external_spoofing = false;
        bool skip_internal_spoofing = false;
        bool skip_external_flooding = false;
        bool skip_internal_flooding = false;
        
        // if the source ip does not start with 2001 or is not :: + comes from an external port, i.e., port <= 5
        if (hdr.ipv6.isValid() && (bit<16>)(hdr.ipv6.srcAddr >> 112) != 0x2001 && hdr.ipv6.srcAddr != 0 && (standard_metadata.ingress_port <= 5 || standard_metadata.ingress_port == 775)) {
            // Mark the packet for drop and exit the processing pipeline
            mark_to_drop();
            return;
        }
        
        meta.dbg_pkt = 0;
        meta.dbg_clear = 0;

        if (hdr.ipv6.isValid()) {
            if (hdr.ipv6.flowLabel == 20w0x12345) { meta.dbg_pkt = 1; }
            if (hdr.ipv6.flowLabel == 20w0x54321) { meta.dbg_pkt = 1; meta.dbg_clear = 1; }
        }


        if (meta.dbg_pkt == 1) {
            debug_flowlabel_reg.write(0, (bit<32>) hdr.ipv6.flowLabel);
            debug_status_reg.write(0, 0xF1);   // optional: "flowlabel matched"
            debug_drop_reason_reg.write(0, 0);
            debug_ext_stage_reg.write(0, 0xE000); // "dbg pkt entered apply"
        }     


        //***************************** External Spoofing Module ******************************
        //*************************************************************************************

        // check if the ingress port is internal port, i.e., any port > 5
        if (standard_metadata.ingress_port > 5 && standard_metadata.ingress_port != 775) {
            // Skip the external spoofing check
            skip_external_spoofing = true;
        }

        // check if the packet is IPv6
        if (hdr.ipv6.isValid() && !skip_external_spoofing) {
            //***** Hop Count (Metadata) Table Check *****
            // Check if the source address is in the existing networks table and apply the hop count table
            // to drop spoofed packets
            //existing_networks.apply(); // set the current_network metadata
            //hop_count.apply(); // drop packets with invalid network hop count pairs

            // Reset once per packet BEFORE the chain
            meta.current_network.network_addr = 0;
            meta.current_network.prefix_len  = 0;

            // Apply in descending specificity, but DO NOT use .hit
            existing_networks_64.apply();
            if (meta.current_network.prefix_len == 0) {
                existing_networks_56.apply();
            }
            if (meta.current_network.prefix_len == 0) {
                existing_networks_48.apply();
            }
            if (meta.current_network.prefix_len == 0) {
                existing_networks_32.apply(); 
            }

            hop_count.apply();

            // check if the packet is dropped by the hop count table and if the srcAddr starts with 2001
            // Note: the srcAddr check is done in order to avoid counting stray RS packets coming from other networks as spoofed
            if (meta.should_drop && (bit<16>)(hdr.ipv6.srcAddr >> 112) == 0x2001) {
                // check if the packet is benign (timestamp tsval == 1000)
                if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
                    // mark the packet for drop
                    hop_count_benign_drop();
                } else { // if the packet is not benign
                    // mark the packet for drop
                    hop_count_other_drop();
                }
                // exit the processing pipeline
                if (meta.dbg_pkt == 1) {
                    debug_drop_reason_reg.write(0, 0xE999);
                }

                return;
            }
            //*******************************************
        }

        
        //***************************** Internal Spoofing Module ******************************
        //*************************************************************************************
        
        bool skip_port_addr_check = false;

        // Check if the packet comes from an external port, i.e., any port <= 5
        if (standard_metadata.ingress_port <= 5 || standard_metadata.ingress_port == 775) {
            // Skip the internal spoofing check
            skip_internal_spoofing = true;
        }

        // Check if the packet is IPv6
        if (!meta.should_drop && !skip_internal_spoofing && hdr.ipv6.isValid()) {
            // DEBUG: Mark 1 - Entered IPv6 Block
            //debug_status_reg.write(0, 1);
            //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
            //bit<1> dbg_pkt = (hdr.ipv6.flowLabel == (bit<20>)0x12345);

            if (meta.dbg_pkt == 1) {
                debug_status_reg.write(0, 0xA1);
            }
            // Check if the packet is an ICMPv6 NS
            if (hdr.icmpv6.isValid() && hdr.icmpv6.type == ICMPV6_NS) {
                //// Check if the target address was not seen before using the bloom filter

                // DEBUG: Mark 2 - DAD Packet Processed
                //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);

                if (meta.dbg_pkt == 1) {
                    debug_status_reg.write(0, 2);
                }
                // get the hash inputs for the ns target address
                get_hash_inputs_addr(hdr.icmpv6_ns.targetAddr, hash_input1, hash_input2, hash_input3);

                // compute the hash values
                jenkins_hash(hash_input1, hash1_i);
                jenkins_hash(hash_input2, hash2_i);
                jenkins_hash(hash_input3, hash3_i);

                // after jenkins_hash(...) gives hash1_i/hash2_i/hash3_i
                //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                if (meta.dbg_pkt == 1) {
                    debug_hash1_reg.write(0, hash1_i);
                    debug_hash2_reg.write(0, hash2_i);
                    debug_hash3_reg.write(0, hash3_i);
                }
                //hash1_i = (bit<32>) hash1;
                //hash2_i = (bit<32>) hash2;
                //hash3_i = (bit<32>) hash3;

                //bit<1> bf_val1;
                //bit<1> bf_val2;
                //bit<1> bf_val3;

                bit<32> bf_val1; 
                bit<32> bf_val2; 
                bit<32> bf_val3;

                //ns_tgt_addr_bf_reg.read(bf_val1, hash1);
                //ns_tgt_addr_bf_reg.read(bf_val2, hash2);
                //ns_tgt_addr_bf_reg.read(bf_val3, hash3);

                ns_tgt_addr_bf_reg.read(bf_val1, hash1_i);
                ns_tgt_addr_bf_reg.read(bf_val2, hash2_i);
                ns_tgt_addr_bf_reg.read(bf_val3, hash3_i);

                //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                if (meta.dbg_pkt == 1) {

                    // after bf_reg.read(...) gives bf_val1/2/3
                    debug_bf1_reg.write(0, bf_val1);
                    debug_bf2_reg.write(0, bf_val2);
                    debug_bf3_reg.write(0, bf_val3);
                }

                if (bf_val1 == 0 || bf_val2 == 0 || bf_val3 == 0) {

                    //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                    if (meta.dbg_pkt == 1) {

                        debug_status_reg.write(0, 0x22); // "learning + write ip_port_reg"
                    }
                    // Add the source address to the bloom filter
                    //ns_tgt_addr_bf_reg.write(hash1, 1);
                    //ns_tgt_addr_bf_reg.write(hash2, 1);
                    //ns_tgt_addr_bf_reg.write(hash3, 1);

                    ns_tgt_addr_bf_reg.write(hash1_i, 1);
                    ns_tgt_addr_bf_reg.write(hash2_i, 1);
                    ns_tgt_addr_bf_reg.write(hash3_i, 1);

                    // Add the source address to the ip_port register
                    //ip_port_reg.write(hash1, standard_metadata.ingress_port);
                    ip_port_reg.write(hash1_i, (bit<32>) standard_metadata.ingress_port);

                    skip_port_addr_check = true;
                }
            }

            // Check if the source address port pair is invalid
            if (!skip_port_addr_check) {

                // calculate the hash values for the source address
                get_hash_inputs_addr(hdr.ipv6.srcAddr, hash_input1, hash_input2, hash_input3);

                // compute the first hash value
                jenkins_hash(hash_input1, hash1_i);

                //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                if (meta.dbg_pkt == 1) {

                    // DEBUG: Mark 3 - About to check Port
                    debug_status_reg.write(0, 3);

                    debug_hash1_reg.write(0, hash1_i);
                    debug_ingress_reg.write(0, (bit<32>)standard_metadata.ingress_port);
                }

                bit<32> port_val;
                //ip_port_reg.read(port_val, hash1);
                ip_port_reg.read(port_val, hash1_i);

                //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                if (meta.dbg_pkt == 1) {

                    debug_portval_reg.write(0, port_val);
                }
                //if (port_val != standard_metadata.ingress_port) {
                if (port_val != (bit<32>) standard_metadata.ingress_port) {
                    //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                    if (meta.dbg_pkt == 1) {
                        debug_status_reg.write(0, 0xD4);
                    }
                    // Drop the packet
                    internal_port_drop();

                    if (meta.dbg_pkt == 1) {
                    debug_drop_reason_reg.write(0, 0xE888);
                    }

                    // DEBUG: Mark 4 - DROPPED by Spoof Check
                    //debug_status_reg.write(0, 4);
                
                    
                    // exit the processing pipeline
                    return;
                }

                // DEBUG: Mark 5 - PASSED Spoof Check
                //debug_status_reg.write(0, 5);
                //bool dbg_pkt = hdr.ipv6.isValid() && (hdr.ipv6.flowLabel == 0x12345);
                if (meta.dbg_pkt == 1) {
                    debug_status_reg.write(0, 0xB5);
                }
            }
        }

        /*
        //***************************** External Flooding Module ******************************
        //*************************************************************************************
        // Get current time
        timestamp_t current_time = standard_metadata.ingress_global_timestamp;

        // Check if the packet comes from an internal port, i.e., any port > 5
        if (standard_metadata.ingress_port > 5 && standard_metadata.ingress_port != 775) {
            // Skip the external flooding check
            skip_external_flooding = true;
        }

        
        if (hdr.ipv6.isValid() && !skip_external_flooding) {

            //bit<1> did_init = 0;
            bit<1> extf_skip_rest = 0;

            if (meta.dbg_pkt == 1) {

                debug_drop_reason_reg.write(0, 0);

                debug_ext_stage_reg.write(0, 0xE100);
                debug_ext_ingress_reg.write(0, (bit<32>) standard_metadata.ingress_port);
                debug_ext_skip_reg.write(0, skip_external_flooding ? 32w1 : 32w0);

                // NEW: log current_time (48b) split into hi/lo
                debug_ext_now_lo_reg.write(0, (bit<32>) current_time);
                debug_ext_now_hi_reg.write(0, (bit<32>) (current_time >> 32));

                debug_ext_plen_reg.write(0, (bit<32>) meta.current_network.prefix_len);
                debug_ext_net_hi_reg.write(0, (bit<32>) (meta.current_network.network_addr >> 96));
                debug_ext_net_lo_reg.write(0, (bit<32>) meta.current_network.network_addr);
            }


            // Note: the network was extracted in the hop count table (external spoofing module)

            // get the hash inputs for the prefix
            get_hash_inputs_prefix(meta.current_network.network_addr, meta.current_network.prefix_len, hash_input1, hash_input2, hash_input3);

            // compute the hash values
            jenkins_hash(hash_input1, hash1_i);
            jenkins_hash(hash_input2, hash2_i);
            jenkins_hash(hash_input3, hash3_i);

            if (meta.dbg_pkt == 1) {
                debug_ext_stage_reg.write(0, 0xE110); // hashes computed
                debug_ext_hash1_reg.write(0, hash1_i);
                debug_ext_hash2_reg.write(0, hash2_i);
                debug_ext_hash3_reg.write(0, hash3_i);
            }

            if (meta.dbg_pkt == 1 && meta.dbg_clear == 1) {
                // Clear ONLY the 3 buckets for this prefix
                cms0.write(hash1_i, 0);  cms1.write(hash2_i, 0);  cms2.write(hash3_i, 0);
                timestamp0.write(hash1_i, current_time);
                timestamp1.write(hash2_i, current_time);
                timestamp2.write(hash3_i, current_time);
                window_start_time0.write(hash1_i, current_time);
                window_start_time1.write(hash2_i, current_time);
                window_start_time2.write(hash3_i, current_time);
                first_window_set.write(hash1_i, 0);
                first_window_set.write(hash2_i, 0);
                first_window_set.write(hash3_i, 0);

                debug_ext_stage_reg.write(0, 0xE115); // prefix buckets cleared
                
                
                extf_skip_rest = 1;
            }


            // check if this is the first time the prefix is seen so that we can set the window start time (bloom filter)
            //bit<1> first_window_set_value1;
            //bit<1> first_window_set_value2;
            //bit<1> first_window_set_value3;

            if (extf_skip_rest == 0) {
            
                bit<32> first_window_set_value1;
                bit<32> first_window_set_value2;
                bit<32> first_window_set_value3;

                first_window_set.read(first_window_set_value1, hash1_i);
                first_window_set.read(first_window_set_value2, hash2_i);
                first_window_set.read(first_window_set_value3, hash3_i);
                
                if (meta.dbg_pkt == 1) {
                    debug_ext_stage_reg.write(0, 0xE120); // first_window_set read
                    debug_ext_fw1_reg.write(0, first_window_set_value1);
                    debug_ext_fw2_reg.write(0, first_window_set_value2);
                    debug_ext_fw3_reg.write(0, first_window_set_value3);
                }


                if (first_window_set_value1 == 0 || first_window_set_value2 == 0 || first_window_set_value3 == 0) {
                    
                    //did_init = 1;
                    
                    cms0.write(hash1_i, 0);
                    cms1.write(hash2_i, 0);
                    cms2.write(hash3_i, 0);

                    timestamp0.write(hash1_i, current_time);
                    timestamp1.write(hash2_i, current_time);
                    timestamp2.write(hash3_i, current_time);

                    // set the window start time for the prefix
                    window_start_time0.write(hash1_i, current_time);
                    window_start_time1.write(hash2_i, current_time);
                    window_start_time2.write(hash3_i, current_time);
                    // set the first_window_set flag for the prefix (bloom filter)
                    first_window_set.write(hash1_i, 1);
                    first_window_set.write(hash2_i, 1);
                    first_window_set.write(hash3_i, 1);

                    extf_skip_rest = 1;

                    if (meta.dbg_pkt == 1) {
                        debug_ext_stage_reg.write(0, 0xE125); // init done; skipping ext flood checks for this packet
                    }

                }
            
            }

            if (extf_skip_rest == 0) {

                // read the window start time for the prefix
                timestamp_t window_start_time_value1;
                timestamp_t window_start_time_value2;
                timestamp_t window_start_time_value3;
                window_start_time0.read(window_start_time_value1, hash1_i);
                window_start_time1.read(window_start_time_value2, hash2_i);
                window_start_time2.read(window_start_time_value3, hash3_i);

                if (meta.dbg_pkt == 1) {
                    debug_ext_stage_reg.write(0, 0xE130); // window_start read
                    debug_ext_wstart1_lo_reg.write(0, (bit<32>) window_start_time_value1);
                    debug_ext_wstart2_lo_reg.write(0, (bit<32>) window_start_time_value2);
                    debug_ext_wstart3_lo_reg.write(0, (bit<32>) window_start_time_value3);
                    debug_ext_wstart1_hi_reg.write(0, (bit<32>)(window_start_time_value1 >> 32));
                
                }


                bit<1> window_reset = 0;

                // check if the duration between now and the window start time is greater than the time window for the prefix
                if ((current_time - window_start_time_value1) > TIME_WINDOW ||
                    (current_time - window_start_time_value2) > TIME_WINDOW ||
                    (current_time - window_start_time_value3) > TIME_WINDOW) {
                    // reset the CMS and window start time for the prefix
                    cms0.write(hash1_i, 0);
                    cms1.write(hash2_i, 0);
                    cms2.write(hash3_i, 0);
                    window_start_time0.write(hash1_i, current_time);
                    window_start_time1.write(hash2_i, current_time);
                    window_start_time2.write(hash3_i, current_time);

                    window_reset = 1;

                    if (meta.dbg_pkt == 1) {
                        debug_ext_winreset_reg.write(0, 1);
                        debug_ext_stage_reg.write(0, 0xE140); // window reset taken
                    }

                }

                if (meta.dbg_pkt == 1 && window_reset == 0) {
                    debug_ext_winreset_reg.write(0, 0);
                }
                

                bit<32> cms_val1 = 0;
                bit<32> cms_val2 = 0;
                bit<32> cms_val3 = 0;

                if (window_reset == 0) {

                    cms0.read(cms_val1, hash1_i);
                    cms1.read(cms_val2, hash2_i);
                    cms2.read(cms_val3, hash3_i);

                    if (meta.dbg_pkt == 1) {
                        debug_ext_stage_reg.write(0, 0xE150); // cms read
                        debug_ext_cms1_reg.write(0, cms_val1);
                        debug_ext_cms2_reg.write(0, cms_val2);
                        debug_ext_cms3_reg.write(0, cms_val3);
                        //debug_ext_min_reg.write(0, min_val);
                    }


                }

                bit<32> min_val = (cms_val1 < cms_val2) ? (cms_val1 < cms_val3 ? cms_val1 : cms_val3) : (cms_val2 < cms_val3 ? cms_val2 : cms_val3);

                if (meta.dbg_pkt == 1) {
                        debug_ext_min_reg.write(0, min_val);
                    }

                if (min_val > EXTERNAL_FLOODING_THRESHOLD) {

                    if (meta.dbg_pkt == 1) {
                        debug_ext_stage_reg.write(0, 0xE160); // about to drop due to flood
                        debug_drop_reason_reg.write(0, hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000
                                                    ? (bit<32>) 0xEF01  // benign flood drop
                                                    : (bit<32>) 0xEF02  // attack flood drop
                                                );
                        bit<32> ts = 0;
                        if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid()) { ts = hdr.tcp_option_ts.tsVal; }
                        debug_ext_tsval_reg.write(0, ts);
                    }

                    
                    // check if the packet is benign (timestamp tsval == 1000)
                    if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
                        // mark the packet for drop
                        external_benign_flood_drop();
                        // exit the processing pipeline
                        return;
                    } else { // check if the packet is not benign
                        // mark the packet for drop
                        external_flood_drop();
                        // exit the processing pipeline
                        return;
                    }
                }

                // read the last time the flow was seen
                timestamp_t last_time1;
                timestamp_t last_time2;
                timestamp_t last_time3;
                timestamp0.read(last_time1, hash1_i);
                timestamp1.read(last_time2, hash2_i);
                timestamp2.read(last_time3, hash3_i);

                if (meta.dbg_pkt == 1) {
                    debug_ext_stage_reg.write(0, 0xE170); // timestamps read
                    debug_ext_last1_lo_reg.write(0, (bit<32>) last_time1);
                    debug_ext_last2_lo_reg.write(0, (bit<32>) last_time2);
                    debug_ext_last3_lo_reg.write(0, (bit<32>) last_time3);

                    debug_ext_dt1_reg.write(0, (bit<32>) (current_time - last_time1));
                    debug_ext_dt2_reg.write(0, (bit<32>) (current_time - last_time2));
                    debug_ext_dt3_reg.write(0, (bit<32>) (current_time - last_time3));
                    debug_ext_last1_hi_reg.write(0, (bit<32>)(last_time1 >> 32));
                }


                // Check intervals between packets is less than TIME_THRESHOLD
                if ((current_time - last_time1) < TIME_THRESHOLD ||
                    (current_time - last_time2) < TIME_THRESHOLD ||
                    (current_time - last_time3) < TIME_THRESHOLD) {
                    increment_cms(hash1_i, hash2_i, hash3_i);
                }

                // Update the timestamps and counters for the flow
                update_timestamps(hash1_i, hash2_i, hash3_i, current_time);
            }
        }
        */
        

        /*
        //***************************** Internal Flooding Module ******************************
        //*************************************************************************************
        current_time = standard_metadata.ingress_global_timestamp;

        // Check if the packet comes from an external port, i.e., any port <= 5
        if (standard_metadata.ingress_port <= 5 || standard_metadata.ingress_port == 775) {
            // Skip the internal flooding check
            skip_internal_flooding = true;
        }

        // Check if the packet is IPv6
        if (hdr.ipv6.isValid() && !skip_internal_flooding) {
            increment_icmpv6_counter(meta); // Increment ICMPv6 packet counter
            
            // check if packet is NS, NA, RA or RS then increment the respective counter
            if (hdr.icmpv6.type == ICMPV6_NS) {
                bit<32> ns_counter_val;
                ns_counter_reg.read(ns_counter_val, 0);
                ns_counter_val = ns_counter_val + 1;
                ns_counter_reg.write(0, ns_counter_val);
            } else if (hdr.icmpv6.type == ICMPV6_NA) {
                bit<32> na_counter_val;
                na_counter_reg.read(na_counter_val, 0);
                na_counter_val = na_counter_val + 1;
                na_counter_reg.write(0, na_counter_val);
            } else if (hdr.icmpv6.type == ICMPV6_RA) {
                bit<32> ra_counter_val;
                ra_counter_reg.read(ra_counter_val, 0);
                ra_counter_val = ra_counter_val + 1;
                ra_counter_reg.write(0, ra_counter_val);
            } else if (hdr.icmpv6.type == ICMPV6_RS) {
                bit<32> rs_counter_val;
                rs_counter_reg.read(rs_counter_val, 0);
                rs_counter_val = rs_counter_val + 1;
                rs_counter_reg.write(0, rs_counter_val);
            }

            //bit<1> multicast_val1;
            //bit<1> multicast_val2;
            //bit<1> multicast_val3;

            bit<32> multicast_val1;
            bit<32> multicast_val2;
            bit<32> multicast_val3;

            // get the hash inputs for the flow
            get_hash_inputs_flow(hdr.ipv6.srcAddr, hdr.ipv6.dstAddr, hash_input1, hash_input2, hash_input3);

            // compute the hash values
            jenkins_hash(hash_input1, hash1_i);
            jenkins_hash(hash_input2, hash2_i);
            jenkins_hash(hash_input3, hash3_i);

            // check if this is the first time the flow is seen so that we can set the window start time (bloom filter)
            //bit<1> first_window_set_value1;
            //bit<1> first_window_set_value2;
            //bit<1> first_window_set_value3;

            bit<32> first_window_set_value1;
            bit<32> first_window_set_value2;
            bit<32> first_window_set_value3;

            first_window_set.read(first_window_set_value1, hash1_i);
            first_window_set.read(first_window_set_value2, hash2_i);
            first_window_set.read(first_window_set_value3, hash3_i);
            if (first_window_set_value1 == 0 || first_window_set_value2 == 0 || first_window_set_value3 == 0) {
                // set the window start time for the flow
                window_start_time0.write(hash1_i, current_time);
                window_start_time1.write(hash2_i, current_time);
                window_start_time2.write(hash3_i, current_time);
                // set the first_window_set flag for the flow
                first_window_set.write(hash1_i, 1);
                first_window_set.write(hash2_i, 1);
                first_window_set.write(hash3_i, 1);
            }

            // check if dst addr is multicast then set the multicast bit
            bit<8> dst_addr_127_120 = (bit<8>)(hdr.ipv6.dstAddr >> 120);
            if (dst_addr_127_120 == 0xFF) {
                multicast_reg.write(hash1_i, 1);
                multicast_reg.write(hash2_i, 1);
                multicast_reg.write(hash3_i, 1);
            }
            multicast_reg.read(multicast_val1, hash1_i);
            multicast_reg.read(multicast_val2, hash2_i);
            multicast_reg.read(multicast_val3, hash3_i);

            // read the window start time for the flow
            timestamp_t window_start_time_value1;
            timestamp_t window_start_time_value2;
            timestamp_t window_start_time_value3;
            window_start_time0.read(window_start_time_value1, hash1_i);
            window_start_time1.read(window_start_time_value2, hash2_i);
            window_start_time2.read(window_start_time_value3, hash3_i);

            // check if the duration between now and the window start time is greater than the time window for the flow
            if ((current_time - window_start_time_value1) > TIME_WINDOW ||
                (current_time - window_start_time_value2) > TIME_WINDOW ||
                (current_time - window_start_time_value3) > TIME_WINDOW) {
                // reset the CMS and window start time for the flowS
                cms0.write(hash1_i, 0);
                cms1.write(hash2_i, 0);
                cms2.write(hash3_i, 0);
                window_start_time0.write(hash1_i, current_time);
                window_start_time1.write(hash2_i, current_time);
                window_start_time2.write(hash3_i, current_time);
            }

            // read the last time the flow was seen
            timestamp_t last_time1;
            timestamp_t last_time2;
            timestamp_t last_time3;
            timestamp0.read(last_time1, hash1_i);
            timestamp1.read(last_time2, hash2_i);
            timestamp2.read(last_time3, hash3_i);

            bit<32> cms_val1;
            bit<32> cms_val2;
            bit<32> cms_val3;
            cms0.read(cms_val1, hash1_i);
            cms1.read(cms_val2, hash2_i);
            cms2.read(cms_val3, hash3_i);

            bit<32> min_val = (cms_val1 < cms_val2) ? (cms_val1 < cms_val3 ? cms_val1 : cms_val3) : (cms_val2 < cms_val3 ? cms_val2 : cms_val3);
            // check multicast bit
            if (multicast_val1 == 1 && multicast_val2 == 1 && multicast_val3 == 1) {
                // the flow is multicast
                if (min_val > MULTICAST_COUNT_THRESHOLD) {
                    // mark the packet for drop
                    internal_multicast_flood_drop();
                    // exit the processing pipeline
                    return;
                }
            }
            else {
                // the flow is unicast
                if (min_val > UNICAST_COUNT_THRESHOLD) {
                    // check if the packet is NS
                    if (hdr.icmpv6.isValid() && hdr.icmpv6.type == ICMPV6_NS) {
                        // mark the packet for drop
                        internal_ns_unicast_flood_drop();
                    } else if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
                        // mark the packet for drop
                        internal_tcp_benign_unicast_flood_drop();
                    } else {
                        // mark the packet for drop
                        internal_other_unicast_flood_drop();
                    }
                    // exit the processing pipeline
                    return;
                }
            }

            // Check intervals between packets is less than TIME_THRESHOLD
            if ((current_time - last_time1) < TIME_THRESHOLD ||
                (current_time - last_time2) < TIME_THRESHOLD ||
                (current_time - last_time3) < TIME_THRESHOLD) {
                increment_cms(hash1_i, hash2_i, hash3_i);
            }

            // Update the timestamps and counters for the flow
            update_timestamps(hash1_i, hash2_i, hash3_i, current_time);
        }
        
        */
        /******************************** Forward Packet *************************************
        *************************************************************************************/
        if (hdr.ethernet.isValid()) {
            mac_lookup.apply(); // forward to the port if known, else broadcast
        }

        /********************************* End of Processing *********************************
        *************************************************************************************/
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop();
    }

    apply {
        // Prune multicast packet to ingress port to preventing loop
        if (standard_metadata.egress_port == standard_metadata.ingress_port)
            drop();

        // if the egress port is 200, drop the packet
        if (standard_metadata.egress_spec == 200) {
            drop();
        }
        
        // calculate and store the ingress packet processing duration
        
        bit<32> last_duration_index;

        last_duration_index_reg.read(last_duration_index, 0);
        // check if the packet is TCP or ICMPv6 NS (the experiments only have TCP and ICMPv6 NS packets)
        if (hdr.tcp.isValid() || (hdr.icmpv6.isValid() && hdr.icmpv6.type == ICMPV6_NS)) {
            // calculate the duration
            duration_t duration = (duration_t)((timestamp_t)standard_metadata.enq_timestamp - standard_metadata.ingress_global_timestamp);
            // store the duration in the ingress_packet_processing_durations_reg
            ingress_packet_processing_durations_reg.write(last_duration_index, duration);
            // increment the last_duration_index
            last_duration_index = last_duration_index + 1;
            last_duration_index_reg.write(0, last_duration_index);
        }
        
    }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {

    }
}


/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv6);
        
        packet.emit(hdr.tcp);
        packet.emit(hdr.tcp_option_ts);
        packet.emit(hdr.tcp_option_end);
        packet.emit(hdr.tcp_option_padding);

        packet.emit(hdr.icmpv6);
        packet.emit(hdr.icmpv6_ns);
        // Note: this will not cause a problem because TCP and ICMPv6 headers are mutually exclusive in the experiments
        // so the switch will ignore the TCP header if the packet is an ICMPv6 packet and vice versa
        
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;