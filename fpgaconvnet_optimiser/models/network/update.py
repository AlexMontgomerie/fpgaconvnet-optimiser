import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.transforms.helper import get_factors

def update_partitions(self):

    # remove all auxiliary layers
    for partition_index in range(len(self.partitions)):

        ## remove squeeze layer
        self.partitions[partition_index].remove_squeeze()

    # remove all empty partitions
    for partition_index in range(len(self.partitions)):

        # remove empty partition
        if len(self.partitions[partition_index].graph.nodes) == 0:
            del self.partitions[partition_index]

    # update partitions
    for partition_index in range(len(self.partitions)):

        ## update the partitions
        self.partitions[partition_index].update()

        ## update batch size for partitions
        self.partitions[partition_index].batch_size = self.batch_size

def update_platform(self, platform):
    # platform
    self.platform = {
        'name'          : 'platform',
        'freq'          : self.freq,
        'reconf_time'   : 0.0,
        'wr_time'       : 0.0,
        'ports'         : 4,
        'port_width'    : 64,
        'mem_bandwidth' : 0,
        'mem_capacity'  : 0,
        'constraints'   : {
            'FF'    : 0,
            'LUT'   : 0,
            'DSP'   : 0,
            'BRAM'  : 0,
            'URAM'  : 0
        }
    }

    # update platform information
    #self.platform['name']           = paltform['name']
    self.platform['ports']          = int(platform['ports'])
    self.platform['port_width']     = int(platform['port_width'])
    #self.platform['freq']           = int(platform['freq'])
    self.platform['reconf_time']    = float(platform['reconf_time'])
    self.platform['mem_capacity']   = int(platform['mem_capacity'])
    self.platform['mem_bandwidth']  = float(platform['mem_bandwidth'])

    # update constraints
    self.platform['constraints']['FF']   = platform['FF']
    self.platform['constraints']['DSP']  = platform['DSP']
    self.platform['constraints']['LUT']  = platform['LUT']
    self.platform['constraints']['BRAM'] = platform['BRAM']

    if 'URAM' in platform.keys():
        self.platform['constraints']['URAM'] = platform['URAM']

    if 'rsc_allocation' in platform.keys():
        self.rsc_allocation = platform['rsc_allocation']

def update_coarse_in_out_partition(self):
    if len(self.partitions) > 1:
        # iterate over partitions
        for i in range(1,len(self.partitions)):
            # get input and output port between partitions
            input_node  = graphs.get_input_nodes(self.partitions[i].graph)[0] # TODO: support multi-port
            output_node = graphs.get_output_nodes(self.partitions[i-1].graph)[0] # TODO: support multi-port
            # update input node's coarse in with previous coarse out
            self.partitions[i].graph.nodes[input_node]['hw'].coarse_in = self.partitions[i-1].graph.nodes[output_node]['hw'].streams_out

