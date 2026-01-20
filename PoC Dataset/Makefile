#BMV2_SWITCH_EXE = simple_switch_grpc
BMV2_SWITCH_EXE = ./bmv2_grpc_wrapper.sh
TOPO = topo/topo_hw.json
BUILD_DIR = build
PCAP_DIR = pcaps
LOG_DIR = logs

P4C = p4c-bm2-ss
P4C_ARGS += --p4runtime-files $(BUILD_DIR)/$(basename $@).p4.p4info.txt

RUN_SCRIPT = run_hw_exp.py
PYTHON = /home/smartnic/p4dev-python-venv/bin/python3

source = $(wildcard *.p4)
compiled_json := $(source:.p4=.json)

ifndef DEFAULT_PROG
DEFAULT_PROG = $(wildcard *.p4)
endif
DEFAULT_JSON = $(BUILD_DIR)/$(DEFAULT_PROG:.p4=.json)

# Define NO_P4 to start BMv2 without a program
# ifndef NO_P4
# run_args += -j $(DEFAULT_JSON)
# endif

#run_args += -j $(BUILD_DIR)/defense.json
run_args += -j $(BUILD_DIR)/multicast.json

# Set BMV2_SWITCH_EXE to override the BMv2 target
ifdef BMV2_SWITCH_EXE
run_args += -b $(BMV2_SWITCH_EXE)
endif

all: run

run: build
	sudo PATH=$(PATH) PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ${P4GUIDE_SUDO_OPTS} $(PYTHON) $(RUN_SCRIPT) -t $(TOPO) $(run_args)

stop:
	sudo PATH=$(PATH) `which mn` -c

build: dirs $(compiled_json)

%.json: %.p4
	$(P4C) --p4v 16 $(P4C_ARGS) -o $(BUILD_DIR)/$@ $<

dirs:
	mkdir -p $(BUILD_DIR) $(PCAP_DIR) $(LOG_DIR)

clean: stop
	rm -f *.pcap
	rm -rf $(BUILD_DIR) $(PCAP_DIR) $(LOG_DIR)
