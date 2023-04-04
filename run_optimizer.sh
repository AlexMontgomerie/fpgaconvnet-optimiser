#!/bin/bash

while getopts ":p:m:n:" opt; do
  case $opt in
    p) platforms="$OPTARG"
    ;;
    m) models="$OPTARG"
    ;;
    n) runs="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

ALL_PLATFORMS=("vus440" "vc709" "zcu102" "zcu104")
ALL_MODELS=("c3d" "r2plus1d_18" "r2plus1d_34" "slowonly" "x3d_m")

DEFAULT_PLATFORMS=("vc709" "zcu102")
DEFAULT_MODELS=("c3d" "r2plus1d_18" "r2plus1d_34" "slowonly" "x3d_m")

PLATFORMS="${platforms:-${DEFAULT_PLATFORMS[@]}}"
MODELS="${models:-${DEFAULT_MODELS[@]}}"

OPTIMIZER="simulated_annealing"
NUM_RUNS="${runs:-5}"

for model_name in ${MODELS[@]}; do
    MODEL_PATH="examples/models/$model_name.onnx"
    for platform_name in ${PLATFORMS[@]}; do
        echo "Running $model_name on $platform_name for $NUM_RUNS times"

        PLATFORM_PATH="examples/platforms/$platform_name.toml"
        OUTPUT_PATH="outputs/$model_name/$platform_name"

        mkdir -p $OUTPUT_PATH

        for ((i=0; i<$NUM_RUNS; i++)); do
            python -m fpgaconvnet.optimiser.latency -n $model_name \
                -m $MODEL_PATH \
                -p $PLATFORM_PATH \
                -o $OUTPUT_PATH \
                --optimiser $OPTIMIZER \
                --optimiser_config_path examples/latency_optimiser_example.toml \
                --enable-wandb
        done
    done
done