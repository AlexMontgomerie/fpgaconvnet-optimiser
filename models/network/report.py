import json
import datetime

def create_report(self, output_path):
    # create report dictionary
    report = {
        "name" : self.name,
        "date_created" : str(datetime.datetime.now()),
        "total_iterations" : 0, # TODO
        "platform" : self.platform,
        "network" : {
            "performance" : {
                "latency" : self.get_latency(),
                "throughput" : self.get_throughput()
            },
            "num_partitions" : len(self.partitions),
            "max_resource_usage" : {
                "BRAM" : max([ partition.get_resource_usage()["BRAM"] for partition in self.partitions ]),
                "DSP" : max([ partition.get_resource_usage()["DSP"] for partition in self.partitions ])   
            }
        }
    }
    # add information for each partition
    report["partitions"] = {}
    for i in range(len(self.partitions)):
        # get some information on the partition
        resource_usage = self.partitions[i].get_resource_usage()
        latency = self.partitions[i].get_latency(self.platform["freq"])
        # add partition information
        report["partitions"][i] = {
            "partition_index" : i,
            "num_layers" : len(self.partitions[i].graph.nodes()),
            "latency" : latency,
            "weights_reloading_factor" : self.partitions[i].wr_factor,
            "weights_reloading_layer" : self.partitions[i].wr_layer,
            "resource_usage" : {
                "BRAM" : resource_usage["BRAM"],
                "DSP" : resource_usage["DSP"]
            },
            "bandwidth" : {
                "in" : 0,   # TODO
                "out" : 0   # TODO
            }
        }
        # add information for each layer of the partition
        report["partitions"][i]["layers"] = {}
        for node in self.partitions[i].graph.nodes():
            hw = self.partitions[i].graph.nodes[node]['hw']
            resource_usage = hw.resource()
            report["partitions"][i]["layers"][node] = {
                "type" : str(self.partitions[i].graph.nodes[node]['type']),
                "interval" : hw.get_latency(), #TODO
                "latency" : hw.get_latency(), 
                "resource_usage" : {
                    "BRAM" : resource_usage["BRAM"],
                    "DSP" : resource_usage["DSP"]
                }
            }
    # save as json
    with open(output_path,"w") as f:
        json.dump(report,f,indent=2)


