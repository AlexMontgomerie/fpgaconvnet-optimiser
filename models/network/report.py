import matplotlib.pyplot as plt
import numpy as np
import random

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

def create_markdown_report(self):
    
    partition_info = ""
    for i in range(len(self.partitions)):
        resource_usage = self.get_resource_usage(i)
        latency = self.get_latency_partition(i)
        bram = resource_usage['BRAM'] 
        dsp  = resource_usage['DSP'] 
        lut  = resource_usage['LUT'] 
        ff   = resource_usage['FF'] 
        wr_factor = self.partitions[i]['wr_factor']
        partition_info += f"| {i} | {wr_factor} | {latency} | {bram} | {dsp} | {lut} | {ff} |\n"

    layer_info = ""
    for i in range(len(self.partitions)):
        for node in self.partitions[i]['graph'].nodes():
            hw = self.partitions[i]['graph'].nodes[node]['hw']
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
