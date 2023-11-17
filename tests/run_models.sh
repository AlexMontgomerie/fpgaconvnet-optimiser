printf "%10s\n" "============= Running Test: resnet_8 ============="
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xtYbYe34mhDTfMNG2c1wax00FU3WhKIS' -O resnet8_fp32.onnx
python -m fpgaconvnet.optimiser --name resnet_8 --model_path resnet8_fp32.onnx \
    --platform_path examples/platforms/zcu104.toml --output_path outputs/resnet_8 --batch_size 1 \
    --objective throughput --optimiser greedy_partition --optimiser_config_path examples/single_partition_throughput.toml 

printf "%10s\n" "============= Running Test: yolov5n_320 ============="
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ZJdHf0DyN8pcG-SMRgh9rTBZa2_1oBXw' -O yolov5n_imgsz320_fp16-fpgaconvnet.onnx
python -m fpgaconvnet.optimiser --name yolov5n_320 --model_path yolov5n_imgsz320_fp16-fpgaconvnet.onnx \
    --platform_path examples/platforms/u250.toml --output_path outputs/yolov5n_320 --batch_size 1 \
    --objective throughput --optimiser greedy_partition --optimiser_config_path examples/single_partition_throughput.toml 