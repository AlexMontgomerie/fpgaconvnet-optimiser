class Partition():

    def __init__(
            self, 
            graph, 
            ports_in=1,
            ports_out=1,
            streams_in=1,
            streams_out=1,
            batch_size=1,
            wr_factor=1
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
        self.wr_layer   = ""
        self.wr_factor  = wr_factor 

        ## featuremap size
        self.size_in    = 0
        self.size_out   = 0
        self.size_wr    = 0

        ## bitwidths
        self.data_width     = 16
        self.weight_width   = 8
        self.acc_width      = 30 

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

    # auxiliary layer functions
    from models.partition.auxiliary import add_squeeze
    from models.partition.auxiliary import remove_squeeze

    from models.partition.metrics import get_pipeline_depth
    from models.partition.metrics import get_interval
    from models.partition.metrics import get_latency

    from models.partition.update import update_modules
    from models.partition.update import update_coefficients

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

