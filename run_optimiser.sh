platform_path="data/platforms/zc706.json"
batch_size=512
objective="throughput"

name="lenet"
model_path="data/models/caffe_onnx/lenet.onnx"
output_path="outputs/lenet_test"

name="alexnet"
model_path="data/models/caffe_onnx/alexnet.onnx"
output_path="outputs/alexnet_test"

name="vgg16"
model_path="data/models/onnx/vgg16bn.onnx"
output_path="outputs/vgg16_test"

#name="zfnet"
#model_path="data/models/onnx/zfnet512.onnx"
#output_path="outputs/zfnet_test"


mkdir -p $output_path

python -m scripts.run_optimiser -n $name -m $model_path -p $platform_path -o $output_path -b $batch_size --objective $objective
