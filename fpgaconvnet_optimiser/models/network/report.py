import json
import datetime
import numpy as np
import csv
import os
import math

def create_report(self, output_path):
    # create report dictionary
    total_operations = sum([partition.get_total_operations() for partition in self.partitions])
    #total_dsps = np.average([partition.get_resource_usage()["DSP"]) for partition in self.partitions])
    report = {
        "name" : self.name,
        "date_created" : str(datetime.datetime.now()),
        "total_iterations" : 0, # TODO
        "platform" : self.platform,
        "total_operations" : total_operations,
        "network" : {
            "memory_usage" : self.get_memory_usage_estimate(),
            "performance" : {
                "latency" : self.get_latency(),
                "throughput" : self.get_throughput(),
                "performance" : total_operations/self.get_latency()
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
            "batch_size" : self.partitions[i].batch_size,
            "num_layers" : len(self.partitions[i].graph.nodes()),
            "latency" : latency,
            "weights_reloading_factor" : self.partitions[i].wr_factor,
            "weights_reloading_layer" : self.partitions[i].wr_layer,
            "resource_usage" : {
                "BRAM" : resource_usage["BRAM"],
                "DSP" : resource_usage["DSP"]
            },
            "bandwidth" : {
                "in" : self.partitions[i].get_bandwidth_in(self.platform["freq"]),
                "out" : self.partitions[i].get_bandwidth_out(self.platform["freq"])
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


def create_csv_report(self,output_path):
    with open(output_path+"/summary.csv", 'a', newline='') as file:
        fieldnames = ['Cluster Size', 'Throughput(actual)','Throughput(interval)','Latency','SingleSampleLatency','Reconfiguration Time','DSP','BRAM','MaxBandwidthIn','MaxBandwidthOut','MaxBandwidth','MemoryEstimate','MaxInterval','NumPartitions']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if os.path.getsize(output_path+"/summary.csv")==0:
            writer.writeheader()
        writer.writerow({'Cluster Size':len(self.cluster),                  
                         'Throughput(actual)':self.get_throughput(),
                         'Throughput(interval)':self.get_multi_fpga_throughput(),
                         'Latency':self.get_latency(),
                         'SingleSampleLatency':self.get_single_sample_latency(),
                         'Reconfiguration Time':(math.ceil(len(self.partitions)/len(self.cluster))-1)*self.platform["reconf_time"],
                         'DSP':max([ partition.get_resource_usage()["DSP"] for partition in self.partitions ]),
                         'BRAM':max([ partition.get_resource_usage()["BRAM"] for partition in self.partitions ]),
                         'MaxBandwidthIn':max([partition.get_bandwidth_in(self.platform["freq"]) for partition in self.partitions]),
                         'MaxBandwidthOut':max([partition.get_bandwidth_out(self.platform["freq"]) for partition in self.partitions]),
                         'MaxBandwidth':max([partition.get_bandwidth_in(self.platform["freq"])+partition.get_bandwidth_out(self.platform["freq"]) for partition in self.partitions]),
                         'MemoryEstimate':self.get_memory_usage_estimate(),
                         'MaxInterval':self.get_max_interval(),
                         'NumPartitions':len(self.partitions)})

