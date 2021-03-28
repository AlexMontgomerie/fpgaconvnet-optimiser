import pydot
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

class Partition():

    def __init__(
            self, 
            graph, 
            ports_in=1,
            ports_out=1,
            streams_in=1,
            streams_out=1,
            batch_size=1,
            wr_factor=1,
            data_width=16,
            weight_width=8,
            acc_width=30
        ):

        ## graph for partition
        self.graph = graph
        
        ## ports
        self.ports_in   = ports_in
        self.ports_out  = ports_out

        ## streams in and out
        self.streams_in  = streams_in
        self.streams_out = streams_out

        ## weights reloading
        self.wr_layer   = self.get_wr_layer()
        self.wr_factor  = wr_factor 

        ## featuremap size
        self.size_in    = 0
        self.size_out   = 0
        self.size_wr    = 0

        ## bitwidths (TODO: add as parameters)
        self.port_width     = 64
        self.data_width     = data_width
        self.weight_width   = weight_width
        self.acc_width      = acc_width

        # maximum streams in and out (TODO: turn into function calls)
        self.max_streams_in     = self.ports_in*int(self.port_width/self.data_width)
        self.max_streams_out    = self.ports_out*int(self.port_width/self.data_width)

        # update model coefficients
        self.update_coefficients()

        # node and edge lists
        #self.node_list = list(self.graph.nodes())
        #self.edge_list = list(self.graph.edges())

        # matrices
        #self.connections_matrix = matrix.get_connections_matrix(self.graph)
        #self.workload_matrix    = matrix.get_workload_matrix(self.graph)
       
        # all types of layers
        #self.conv_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)
        #self.pool_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Pooling)


    ## fine transform
    from fpgaconvnet_optimiser.transforms.fine import apply_random_fine_layer 
    from fpgaconvnet_optimiser.transforms.fine import apply_complete_fine 

    ## weights reloading transform
    from fpgaconvnet_optimiser.transforms.weights_reloading import get_wr_layer
    from fpgaconvnet_optimiser.transforms.weights_reloading import get_weights_reloading_factors
    from fpgaconvnet_optimiser.transforms.weights_reloading import apply_random_weights_reloading 
    from fpgaconvnet_optimiser.transforms.weights_reloading import apply_max_weights_reloading 
    from fpgaconvnet_optimiser.transforms.weights_reloading import remove_weights_reloading_transform 
    from fpgaconvnet_optimiser.transforms.weights_reloading import apply_weights_reloading_transform 

    ## coarse transform
    from fpgaconvnet_optimiser.transforms.coarse import apply_random_coarse_layer
    from fpgaconvnet_optimiser.transforms.coarse import fix_coarse

    # auxiliary layer functions
    from fpgaconvnet_optimiser.models.partition.auxiliary import add_squeeze
    from fpgaconvnet_optimiser.models.partition.auxiliary import remove_squeeze

    # metrics
    from fpgaconvnet_optimiser.models.partition.metrics import get_pipeline_depth
    from fpgaconvnet_optimiser.models.partition.metrics import get_interval
    from fpgaconvnet_optimiser.models.partition.metrics import get_latency
    
    # update
    from fpgaconvnet_optimiser.models.partition.update import update_modules
    from fpgaconvnet_optimiser.models.partition.update import update_coefficients

    def get_resource_usage(self):
        # initialise resource usage at 0
        resource_usage = { # TODO: initialise with partition resource usage
            'FF'    : 0,
            'LUT'   : 0,
            'DSP'   : 0,
            'BRAM'  : 0
        }
        # iterate over nodes in partition
        for node in self.graph.nodes():
            # get the resource usage of the node
            resource_usage_node = self.graph.nodes[node]['hw'].resource()
            # update total resource usage for partition
            resource_usage['FF']    += resource_usage_node['FF'] 
            resource_usage['LUT']   += resource_usage_node['LUT'] 
            resource_usage['DSP']   += resource_usage_node['DSP'] 
            resource_usage['BRAM']  += resource_usage_node['BRAM']
        # return resource usage for partition
        return resource_usage

    def visualise(self, partition_index):
        cluster = pydot.Cluster(str(partition_index),label=f"partition: {partition_index}")
        # add clusters
        edge_labels = {}
        for node in self.graph:
            node_cluster, nodes_in, nodes_out = self.graph.nodes[node]['hw'].visualise(node)
            edge_labels[node] = {
                "nodes_in"  : nodes_in,
                "nodes_out" : nodes_out
            }
            cluster.add_subgraph(node_cluster)
        # create edges
        for node in self.graph:
            for edge in graphs.get_next_nodes(self.graph,node):
                for i in range(self.graph.nodes[node]['hw'].coarse_out):
                    cluster.add_edge(pydot.Edge(edge_labels[node]["nodes_out"][i] ,edge_labels[edge]["nodes_in"][i]))
        # return cluster
        return cluster

    def max_compute_node_latency(self):
        max_latency = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] != LAYER_TYPE.Squeeze:
                latency = self.graph.nodes[node]["hw"].get_latency()
                if latency > max_latency:
                    max_latency = latency

        return max_latency


    def is_input_memory_bound(self):
        input_node  = graphs.get_input_nodes(self.graph)[0]
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return self.graph.nodes[input_node]["type"] == LAYER_TYPE.Squeeze and self.graph.nodes[input_node]["hw"].get_latency() > max_compute_latency 
    
    def is_output_memory_bound(self):
        output_node  = graphs.get_output_nodes(self.graph)[0]
        max_compute_latency = self.max_compute_node_latency()

        for node in self.graph.nodes():
            if self.graph.nodes[node]["type"] == LAYER_TYPE.InnerProduct:
                return False

        return self.graph.nodes[output_node]["type"] == LAYER_TYPE.Squeeze and self.graph.nodes[output_node]["hw"].get_latency() > max_compute_latency 

    def reset(self):
        self.remove_squeeze()
        self.remove_weights_reloading_transform()

        for node in self.graph.nodes():
            self.graph.nodes[node]["hw"].update_coarse_in(1)
            self.graph.nodes[node]["hw"].update_coarse_out(1)             
            self.graph.nodes[node]["hw"].update_coarse_group(1)    

            if self.graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
                self.graph.nodes[node]["hw"].fine = 1
