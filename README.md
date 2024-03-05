# fpgaConvNet Optimiser

This repo contains code for optimising the mapping a Convolutional Neural Network (CNN) model to an FPGA. Hardware-specific transforms are applied to the model, producing a hardware description that can be used by a hardware backend, such as [fpgaConvNet HLS](https://github.com/AlexMontgomerie/fpgaconvnet-hls). The generated architecture is streaming-based, and optimised for the specific hardware platform.

## Setup

The following programs are required:

- `python=3.10`

To install this package, run from this directory the following:

```
sudo apt-get install protobuf-compiler libprotoc-dev
python -m pip install .
```

## Optimiser Framework

The main tool is the optimisation script which generates an optimised hardware topology for a given model and platform. There are several components needed for this: a model of the hardware, transforms that map the model to the hardware and an optimisation scheme that chooses the best mapping. These will be outlined later.
To use the optimiser, an example of running it using the `run_optimiser.py` script for VGG16 is as follows:

```Shell
python -m fpgaconvnet_optimiser --name vgg16 \
    --model_path examples/models/vgg16.onnx \
    --platform_path examples/platforms/zc706.json \
    --output_path outputs/vgg16 \
    --batch_size 256 \
    --objective throughput \
    --optimiser simulated_annealing \
    --optimiser_config_path examples/optimiser_example.yml
```

This will generate the following files:

- `(output_path)/(name).prototxt`: Hardware topology description for backend
- `(output_path)/report.json`: A report file containing estimations of usage and performance
- `(output_path)/scheduler.csv`: A schedule for running partitions as well as information for memory management
- `(output_path)/topology.png`: Visualisation of the hardware topology

These files in the output directory can be used with [fpgaConvNet HLS](https://github.com/AlexMontgomerie/fpgaconvnet-hls) to generate the actual hardware and run on the board.

### Latency Optimizer Framework
Following the pradigm of the `Optimizer Framework` one can run the `Latency Optimizer` as follows:
```
python -m fpgaconvnet.optimiser.latency --name c3d \
    --model_path examples/models/c3d.onnx \
    --platform_path examples/platforms/zcu104.toml \
    --output_path outputs/c3d \
    --optimiser simulated_annealing \
    --optimiser_config_path examples/optimiser_example.toml
```
The same files are generated as before in this case.


### Running WandB Sweeps
To run a sweep on wandb, you can use the following command:
```
python -m fpgaconvnet.optimiser.sweep_wandb -n unet -m examples/models/unet.onnx -p examples/platforms/u200.toml -o outputs/unet/throughput/u200 -b 1 --objective throughput --optimiser greedy_partition --optimiser_config_path examples/optimisers/single_partition_throughput.toml --enable-wandb --sweep-wandb
```
---

Feel free to post an issue if you have any questions or problems!
