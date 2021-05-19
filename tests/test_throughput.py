import math

def singleFPGALatency(partitions,cluster_size,reconfiguration_time,batch_size):
    overall_latency = 0
    for i in range(math.floor(len(partitions)/cluster_size)):
        cluster_latency = 0
        for j in range(cluster_size):
            partition=partitions[i*cluster_size+j]
            partition_latency=partition["latency"]+batch_size*partition["interval"]
            cluster_latency += partition_latency
        overall_latency += cluster_latency + reconfiguration_time
    return overall_latency

def singleFPGAinterval(partitions,cluster_size,reconfiguration_time,batch_size):
    interval = 0
    if cluster_size>=len(partitions):
        for partition in partitions:
            if interval<partition["interval"]:
                interval = partition["interval"]
    else:
        interval = singleFPGALatency(partitions,cluster_size,reconfiguration_time,batch_size)/batch_size
    return interval

def multipleFPGAinterval(partitions,cluster_size,reconfiguration_time,batch_size):
    interval = 0
    if cluster_size>=len(partitions):
        for partition in partitions:
            if interval<partition["interval"]:
                interval = partition["interval"]
    else:
        interval = multipleFPGALatency(partitions,cluster_size,reconfiguration_time,batch_size)/batch_size
    return interval

def multipleFPGALatency(partitions,cluster_size,reconfiguration_time,batch_size):
    overall_latency = 0
    for i in range(math.floor(len(partitions)/cluster_size)):
        cluster_latency=0
        max_interval = 0
        for j in range(cluster_size):
            partition = partitions[i*cluster_size+j]
            if max_interval < partition["interval"]:
                max_interval = partition["interval"]
            partition_latency = partition["latency"]
            cluster_latency += partition_latency
        cluster_latency += max_interval * batch_size
        overall_latency += cluster_latency + reconfiguration_time
    return overall_latency


def main():
    cluster_size = 4
    partitions = [  {"latency":9.865032816,"interval":9.865032816},
                    {"latency":3.288334336,"interval":3.288334336},
                    {"latency":26.306957592,"interval":26.306957592},
                    {"latency":6.578874048,"interval":6.578874048},
                    {"latency":6.57905192,"interval":6.57905192},
                    {"latency": 26.313279744,"interval": 26.313279744},
                    {"latency":0.411071488,"interval":0.411071488},
                    {"latency":3.288384584,"interval":3.288384584},
                    {"latency":9.879209056,"interval":9.879209056},
                    {"latency":6.591428896,"interval":6.591428896},
                    {"latency":6.595118208,"interval":6.595118208},
                    {"latency":4.949088448,"interval":4.949088448},
                    {"latency":6.591111456,"interval":6.591111456},
                    {"latency":6.597018624,"interval":6.597018624},
                    {"latency":1.211106304,"interval":1.211106304},
                    {"latency":0.557050856,"interval":0.557050856},
                    ]
    expected_latency=125.84976837599999
    reconfiguration_time = 0.08255
    batch_size = 256
    singleInterval = singleFPGAinterval(partitions,1,reconfiguration_time,batch_size)
    doubleInterval = multipleFPGAinterval(partitions,2,reconfiguration_time,batch_size)
    tripleInterval = multipleFPGAinterval(partitions,3,reconfiguration_time,batch_size)
    fourInterval = multipleFPGAinterval(partitions,4,reconfiguration_time,batch_size)
    print("Single FPGA latency:     {}".format(singleFPGALatency(partitions,1,reconfiguration_time,batch_size)))
    print("Two    FPGA latency:     {}".format(multipleFPGALatency(partitions,2,reconfiguration_time,batch_size)))
    print("Three  FPGA latency:     {}".format(multipleFPGALatency(partitions,3,reconfiguration_time,batch_size)))
    print("Four   FPGA latency:     {}".format(multipleFPGALatency(partitions,4,reconfiguration_time,batch_size)))
    print("Single FPGA interval:    {}".format(singleInterval))
    print("Two    FPGA interval:    {}".format(doubleInterval))
    print("Three  FPGA interval:    {}".format(tripleInterval))
    print("Four   FPGA interval:    {}".format(fourInterval))
    print("Throughput Normalized:")
    print("Single FPGA throughput:  {}".format(1/(singleInterval)))
    print("Two    FPGA throughput:  {}".format(1/(doubleInterval)))
    print("Three  FPGA throughput:  {}".format(1/(tripleInterval)))
    print("Four   FPGA throughput:  {}".format(1/(fourInterval)))


main()