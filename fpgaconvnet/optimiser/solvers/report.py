import json
import datetime
import numpy as np

from dataclasses import asdict
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def create_report(self, output_path):
    # create report dictionary
    total_operations = sum([partition.get_total_operations() for partition in self.net.partitions])
    inter_delay = self.get_inter_delay()
    latency = self.net.get_latency(self.platform.board_freq, self.multi_fpga, inter_delay)
    throughput = self.net.get_throughput(self.platform.board_freq, self.multi_fpga, inter_delay)
    report = {}
    report = {
        "name" : self.net.name,
        "date_created" : str(datetime.datetime.now()),
        #"total_iterations" : 0, # TODO
        "platform" : asdict(self.platform),
        "total_operations" : float(total_operations),
        "network" : {
            #"memory_usage" : self.net.get_memory_usage_estimate(),
            "multi_fpga" : self.multi_fpga,
            "performance" : {
                "latency (s)" : latency,
                "throughput (FPS)" : throughput,
                "performance (OP/s)" : total_operations/latency,
                "cycles" : self.net.get_cycle(self.multi_fpga),
                "delays between partitions (s)" : inter_delay
            },
            "num_partitions" : len(self.net.partitions),
            "max_resource_usage" : {
                "LUT" : max([ partition.get_resource_usage()["LUT"] for partition in self.net.partitions ]),
                "FF" : max([ partition.get_resource_usage()["FF"] for partition in self.net.partitions ]),
                "BRAM" : max([ partition.get_resource_usage()["BRAM"] for partition in self.net.partitions ]),
                "DSP" : max([ partition.get_resource_usage()["DSP"] for partition in self.net.partitions ])
            },
            "sum_resource_usage" : {
                "LUT" : int(np.sum([ partition.get_resource_usage()["LUT"] for partition in self.net.partitions ])),
                "FF" : int(np.sum([ partition.get_resource_usage()["FF"] for partition in self.net.partitions ])),
                "BRAM" : int(np.sum([ partition.get_resource_usage()["BRAM"] for partition in self.net.partitions ])),
                "DSP" : int(np.sum([ partition.get_resource_usage()["DSP"] for partition in self.net.partitions ]))
            }
        }
    }

    if self.platform.get_uram() > 0:
        report["network"]["max_resource_usage"]["URAM"] = max([ partition.get_resource_usage()["URAM"] for partition in self.net.partitions ])
        report["network"]["sum_resource_usage"]["URAM"] = int(np.sum([ partition.get_resource_usage()["URAM"] for partition in self.net.partitions ]))
    
    # add information for each partition
    report["partitions"] = {}
    for i in range(len(self.net.partitions)):
        # get some information on the partition
        resource_usage = self.net.partitions[i].get_resource_usage()
        latency = self.net.partitions[i].get_latency(self.platform.board_freq)
        # add partition information
        report["partitions"][i] = {
            "partition_index" : i,
            "batch_size" : self.net.partitions[i].batch_size,
            "num_layers" : len(self.net.partitions[i].graph.nodes()),
            "latency" : latency,
            "cycles" : self.net.partitions[i].get_cycle(),
            "slowdown": self.net.partitions[i].slow_down_factor,
            "weights_reloading_factor" : self.net.partitions[i].wr_factor,
            "weights_reloading_layer" : self.net.partitions[i].wr_layer,
            "resource_usage" : {
                "LUT" : resource_usage["LUT"],
                "FF" : resource_usage["FF"],
                "BRAM" : resource_usage["BRAM"],
                "DSP" : resource_usage["DSP"]
            },
            "bandwidth" : {
                "in (Gbps)" : sum(self.net.partitions[i].get_bandwidth_in(self.platform.board_freq)), 
                "out (Gbps)" : sum(self.net.partitions[i].get_bandwidth_out(self.platform.board_freq)), 
                "weight (Gbps)": sum(self.net.partitions[i].get_bandwidth_weight(self.platform.board_freq))
            }
        }
        if self.platform.get_uram() > 0:
            report["partitions"][i]["resource_usage"]["URAM"] = resource_usage["URAM"]

        # add information for each layer of the partition
        report["partitions"][i]["layers"] = {}
        max_latency = max([self.net.partitions[i].graph.nodes[node]['hw'].latency() for node in self.net.partitions[i].graph.nodes()])
        for node in self.net.partitions[i].graph.nodes():
            hw = self.net.partitions[i].graph.nodes[node]['hw']
            resource_usage = hw.resource()
            report["partitions"][i]["layers"][node] = {
                "type" : str(self.net.partitions[i].graph.nodes[node]['type']),
                "interval" : hw.latency(), #TODO
                "cycle" : hw.latency(),
                "resource_usage" : {
                    "LUT" : resource_usage["LUT"],
                    "FF" : resource_usage["FF"],
                    "BRAM" : resource_usage["BRAM"],
                    "DSP" : resource_usage["DSP"]
                }
            }
            if "URAM" in resource_usage.keys():
                report["partitions"][i]["layers"][node]["resource_usage"]["URAM"] = resource_usage["URAM"]
            if self.net.partitions[i].graph.nodes[node]["type"] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                bits_per_cycle = hw.stream_bw()
                bw_weight = bits_per_cycle * self.platform.board_freq / 1000
                bw_weight = bw_weight * hw.latency() / max_latency
                report["partitions"][i]["layers"][node]["resource_usage"]["BW(weight)"] = bw_weight

    # save as json
    with open(output_path,"w") as f:
        json.dump(report,f,indent=2)
