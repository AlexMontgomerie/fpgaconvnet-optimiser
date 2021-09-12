platform_path="examples/platforms/zc706.json"
batch_size=512
objective="throughput"

name="lenet"
model_path="examples/models/lenet.onnx"
output_path="outputs/lenet"

#name="caffenet"
#model_path="examples/models/caffenet.onnx"
#output_path="outputs/caffenet"

#name="vgg16"
#model_path="data/models/onnx/vgg16bn.onnx"
#output_path="outputs/vgg16_test"

#name="zfnet"
#model_path="data/models/onnx/zfnet512.onnx"
#output_path="outputs/zfnet_test"

# Added branchynet networks for baseline results
name="brn_first_exit"
model_path="examples/models/brn_first-exit_with-bias.onnx"
output_path="outputs/brn_first_exit"

name="brn_second_exit"
model_path="examples/models/brn_second-exit_with-bias.onnx"
output_path="outputs/brn_second_exit"

mkdir -p $output_path

python -m fpgaconvnet_optimiser -n $name \
    -m $model_path \
    -p $platform_path \
    -o $output_path \
    -b $batch_size \
    --objective $objective \
    --transforms fine weights_reloading coarse partition \
    --optimiser simulated_annealing \
    --optimiser_config_path examples/optimiser_example.yml
