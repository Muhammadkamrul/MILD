
#How to run the SmartNIC test

#Setup

Run 'sudo mn -c' in mininet
Start smartNIC RTE with design load (See this link on how to install RTE for Netronome SmartNIC)
Setup MACs (use script: setup_vfs.sh)
Do 'Make clean', 'Make run' in mininet
Load Rules (command: sudo /opt/netronome/p4/bin/rtecli --rte-port 20206 config-reload -c rule_for_smartNIC_split.json
) and set Muliticast (command: sudo /opt/netronome/p4/bin/rtecli -p 20206 -j multicast set -g 1 -p 768,769,770,771,772,773,774,775
)
Run The DAD/NS script (use script: send_ns_register.py)


#Test
Run "sudo ./collect_mild_poc_dataset.py "
It will transmit traffic and log them in csv file.


