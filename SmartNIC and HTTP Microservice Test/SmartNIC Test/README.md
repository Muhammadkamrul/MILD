
# How to run the SmartNIC test

# Setup

## Run 'sudo mn -c' in mininet
## Compile the p4 code:
sudo /opt/netronome/p4/bin/nfp4build   -4 defense.p4   -o defense.nffw   -p out_defense   -l carbon   -s AMDA0099-0001:0   -e   --nfp4c_p4_version 16   --nfirc_no_all_header_ops   --no-debug-info

## Start smartNIC RTE with design load (See this link on how to install RTE for Netronome SmartNIC: https://help.netronome.com/support/solutions/36000069814)
sudo env LD_LIBRARY_PATH=/opt/netronome/lib \
/opt/nfp_pif/bin/pif_rte -n 0 -p 20206 -I -z \
  -s /opt/nfp_pif/scripts/pif_ctl_nfd.sh \
  -c /opt/nfp_pif/etc/configs/c-2x25GE-prepend.json \
  -d /home/smartnic/Desktop/IPv6_Security/Rule_based_defense_p416_code/out_defense/pif_design.json \
  -f /home/smartnic/Desktop/IPv6_Security/Rule_based_defense_p416_code/defense.nffw \
  --log_file /tmp/pif_rte_debug.log

## Setup MACs (use script: setup_vfs.sh)
## Do 'Make clean', 'Make run' in mininet
## Load Rules (command: sudo /opt/netronome/p4/bin/rtecli --rte-port 20206 config-reload -c rule_for_smartNIC_split.json)
## set Muliticast (command: sudo /opt/netronome/p4/bin/rtecli -p 20206 -j multicast set -g 1 -p 768,769,770,771,772,773,774,775)
## Run The DAD/NS script (use script: send_ns_register.py)

# Test
## Run "sudo ./collect_mild_poc_dataset.py "
## It will transmit traffic and log them in csv file.


