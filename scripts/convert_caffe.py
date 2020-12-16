"""
lifted from: https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb
"""

import coremltools
import onnxmltools
import argparse
import os
import sys

sys.path.append(os.environ.get("FPGACONVNET_ROOT"))

import tools.onnx_helper as onnx_helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caffe to ONNX conversion script")
    
    parser.add_argument('-c','--caffe_path',metavar='PATH',required=True,
        help='Path to Caffe Prototxt (.prototxt)')
    parser.add_argument('-m','--model_path',metavar='PATH',required=True,
        help='Path to Caffe Model (.caffemodel)')
    parser.add_argument('-o','--onnx_path',metavar='PATH',required=True,
        help='Path to ONNX Model Output (.onnx)')

    # parse arguments
    args = parser.parse_args()
 
    # Convert Caffe model to CoreML 
    coreml_model = coremltools.converters.caffe.convert((args.model_path, args.caffe_path))

    # Convert the Core ML model into ONNX
    onnx_model = onnxmltools.convert_coreml(coreml_model)

    # add and correct initializers
    onnx_helper.add_input_from_initializer(onnx_model)
    onnx_helper.add_value_info_for_constants(onnx_model)

    # Save as protobuf
    onnxmltools.utils.save_model(onnx_model, args.onnx_path)
