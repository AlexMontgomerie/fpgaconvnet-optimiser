printf "%10s\n" "============= Running Test: resnet_8 ============="
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xtYbYe34mhDTfMNG2c1wax00FU3WhKIS' -O resnet8_fp32.onnx
python -m fpgaconvnet.optimiser --name resnet_8 --model_path resnet8_fp32.onnx \
    --platform_path examples/platforms/zcu104.toml --output_path outputs/resnet_8 --batch_size 1 \
    --objective throughput --optimiser greedy_partition --optimiser_config_path examples/optimisers/single_partition_throughput.toml 
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi


printf "%10s\n" "============= Running Test: lggmri_unet ============="
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1FKXx6MzhQMowMp6lcgvl5My9Rcfhz2S9' -O unet_bsp_bilinear_approx.onnx
python -m fpgaconvnet.optimiser --name lggmri_unet --model_path unet_bsp_bilinear_approx.onnx \
    --platform_path examples/platforms/u200.toml --output_path outputs/lggmri_unet --batch_size 1 \
    --objective throughput --optimiser greedy_partition --optimiser_config_path examples/optimisers/greedy_partition_throughput_unet.toml 
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi


printf "%10s\n" "============= Running Test: yolov5n_320 ============="
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ZJdHf0DyN8pcG-SMRgh9rTBZa2_1oBXw' -O yolov5n_imgsz320_fp16-fpgaconvnet.onnx
python -m fpgaconvnet.optimiser --name yolov5n_320 --model_path yolov5n_imgsz320_fp16-fpgaconvnet.onnx \
    --platform_path examples/platforms/u250.toml --output_path outputs/yolov5n_320 --batch_size 1 \
    --objective throughput --optimiser greedy_partition --optimiser_config_path examples/optimisers/single_partition_throughput.toml 
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi


#printf "%10s\n" "============= Running Test: x3d_s ============="
#wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Dse4g-n3INpfHP4b2CRRy0IbgUOQc5Mx' -O x3d_s.onnx
#python -m fpgaconvnet.optimiser --name x3d_s --model_path x3d_s.onnx \
#    --platform_path examples/platforms/u200.toml --output_path outputs/x3d_s --batch_size 1 \
#    --objective throughput --optimiser greedy_partition --optimiser_config_path examples/optimisers/greedy_partition_throughput_x3d.toml 
#retVal=$?
#if [ $retVal -ne 0 ]; then
#    exit $retVal
#fi

