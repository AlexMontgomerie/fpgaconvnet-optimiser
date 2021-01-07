import matplotlib.pyplot as plt
import numpy as np
import random
import json
import datetime

def get_partition_colours(self):
    random.seed(312312)
    colour = [(0,0,0)]*len(self.partitions)
    for partition_index in range(len(self.partitions)):
        colour[partition_index] = (random.random(),random.random(),random.random())
    return colour

def layer_interval_plot(self):
    layers    = []
    intervals = []
    colours   = []
    partition_colours = self.get_partition_colours()
    # iterate over partitions
    for partition_index in range(len(self.partitions)):
        #colour = random.choice(['b','g','r','c','m','y','k','w'])
        colour = (random.random(),random.random(),random.random())
        # iterate over layers in partition
        for layer in self.partitions[partition_index]['layers']:
            interval = self.partitions[partition_index]['layers'][layer]['hw'].get_latency()
            layers.append(layer)
            intervals.append(interval)
            #colours.append(colour)
            colours.append(partition_colours[partition_index])
    # plot bar graph
    plt.bar(np.arange(len(layers)), intervals, color=colours)
    plt.xticks(np.arange(len(layers)), layers)
    plt.show()

# TODO: preserve colours
def partition_interval_plot(self):
    intervals = []
    partition_colours = self.get_partition_colours()
    # iterate over partitions
    for partition_index in range(len(self.partitions)):
        intervals.append(self.get_latency_partition(partition_index))
    # plot bar graph
    plt.bar(np.arange(len(self.partitions)), intervals, color=partition_colours)
    plt.show()


def partition_resource_plot(self):
    pass
    #resources = []
    #partition_colours = 

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
                "interval" : 0, #TODO
                "latency" : 0, #TODO
                "resource_usage" : {
                    "BRAM" : resource_usage["BRAM"],
                    "DSP" : resource_usage["DSP"]
                }
            }
    # save as json
    print(report)
    with open(output_path,"w") as f:
        json.dump(report,f,indent=2)


def create_markdown_report(self):
    
    partition_info = ""
    for i in range(len(self.partitions)):
        resource_usage = self.partitions[i].get_resource_usage()
        latency = self.partitions[i].get_latency(self.platform["freq"])
        bram = resource_usage['BRAM'] 
        dsp  = resource_usage['DSP'] 
        lut  = resource_usage['LUT'] 
        ff   = resource_usage['FF'] 
        wr_factor = self.partitions[i].wr_factor
        partition_info += f"| {i} | {wr_factor} | {latency} | {bram} | {dsp} | {lut} | {ff} |\n"

    layer_info = ""
    for i in range(len(self.partitions)):
        for node in self.partitions[i].graph.nodes():
            hw = self.partitions[i].graph.nodes[node]['hw']
            rsc = hw.resource()
            bram = rsc['BRAM'] 
            dsp  = rsc['DSP'] 
            lut  = rsc['LUT'] 
            ff   = rsc['FF'] 
            layer_info+=f"| {i} | {node} |  {bram} | {dsp} | {lut} | {ff} |\n"

    report = """
# {name} report

## Structure

## Performance

### Overview

| Latency (s) | {latency} |
| Throughput (fps) | {throughput} |

## Usage

### Overview

### partitions

| partition index | wr factor | latency | BRAM | DSP | LUT | FF |
|:---------------:|:---------:|:-------:|:----:|:---:|:---:|:--:|
{partition_info}

### layers

| partition index | layer | BRAM | DSP | LUT | FF |
|:---------------:|:-----:|:----:|:---:|:---:|:--:|
{layer_info}
""".format(
        name=self.name,
        latency=self.get_latency(),
        throughput=self.get_throughput(),
        partition_info=partition_info,
        layer_info=layer_info
    )
    print(report)
