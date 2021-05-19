platform_path="examples/platforms/zc706.json"
cluster_root="examples/cluster/"
batch_size=256
objective="throughput"

name="lenet"
model_path="examples/models/lenet.onnx"

#name="caffenet"
#model_path="examples/models/caffenet.onnx"
#output_path="outputs/caffenet"

#name="vgg16"
#model_path="data/models/onnx/vgg16bn.onnx"
#output_path="outputs/vgg16_test"

#name="zfnet"
#model_path="data/models/onnx/zfnet512.onnx"
#output_path="outputs/zfnet_test"
output_root="outputs/lenet/test/"

for test in eight nine ten
do
    output_path="$output_root${test}"
    cluster_path="$cluster_root${test}_fpga.json"
    for i in 0 1 2 3 4 
    do
        for j in 1 2
        do 
            run_path="${output_path}/run_$(($j + $i*2))"
            echo "Looping ... number $(($j + $i*2))" 

            mkdir -p $run_path

            python -m fpgaconvnet_optimiser -n $name \
                -m $model_path \
                -p $platform_path \
                -c $cluster_path \
                -o $run_path \
                -b $batch_size \
                --objective $objective \
                --transforms fine weights_reloading coarse partition \
                --optimiser simulated_annealing \
                --optimiser_config_path examples/quick_run.yml &>/dev/null
        done
        wait
    done
done