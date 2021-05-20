import pydot
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

class Partition():

    def __init__(
            self,
            graph,
            ctrledges,
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
        ## control flow edges for graph
        self.ctrledges = ctrledges

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
    from fpgaconvnet_optimiser.models.partition.metrics import get_total_operations
    from fpgaconvnet_optimiser.models.partition.metrics import get_bandwidth_in
    from fpgaconvnet_optimiser.models.partition.metrics import get_bandwidth_out
    from fpgaconvnet_optimiser.models.partition.metrics import get_resource_usage

    # update
    from fpgaconvnet_optimiser.models.partition.update import update
    def visualise(self, partition_index):
        cluster = pydot.Cluster(str(partition_index),label=f"partition: {partition_index}")
        # add clusters
        edge_labels = {}
        for node in self.graph:
            #node doesn't provide useful names for pytorch output
            nname = str(node) + "-" + str(self.graph.nodes[node]['type'])[11:]
            node_cluster, nodes_in, nodes_out = self.graph.nodes[node]['hw'].visualise(nname)
            edge_labels[node] = {
                "nodes_in"  : nodes_in,
                "nodes_out" : nodes_out
            }
            cluster.add_subgraph(node_cluster)
        # create edges
        for node in self.graph:
            for edge in graphs.get_next_nodes(self.graph,node):
                #for i in range(self.graph.nodes[node]['hw'].streams_out()):
                #    cluster.add_edge(pydot.Edge(edge_labels[node]["nodes_out"][i] ,edge_labels[edge]["nodes_in"][i]))
                #split layer nodes out duplicated at layer level
                for i in range(self.graph.nodes[node]['hw'].streams_out()):
                    cluster.add_edge(pydot.Edge(edge_labels[node]["nodes_out"][i],
                                    edge_labels[edge]["nodes_in"][i]))
        #control edges
        #print(self.ctrledges)
        #for node in self.graph:
        for ctrl in self.ctrledges:
            for i in range(1,4): #index 1,2,3 of the ctrl edge
                #TODO fix assumption that each in-out pair has only one node
                cluster.add_edge(pydot.Edge(edge_labels[ctrl[0]]["nodes_out"][0],
                                            edge_labels[ctrl[i]]["nodes_in"][0],
                                            color='red'))
                print(edge_labels[ctrl[0]]["nodes_out"])
                print(edge_labels[ctrl[i]]["nodes_in"])
        # return cluster
        return cluster

    def max_compute_node_latency(self):
        # return max([ self.graph.nodes[node]["hw"].get_latency() for node in
        #              self.graph.nodes() ])
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
            self.graph.nodes[node]["hw"].coarse_in = 1
            self.graph.nodes[node]["hw"].coarse_out = 1
            self.graph.nodes[node]["hw"].coarse_group = 1

            if self.graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
                self.graph.nodes[node]["hw"].fine = 1
