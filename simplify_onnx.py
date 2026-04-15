#!/usr/bin/env python3
"""
ONNX model simplification script.

Uses the onnxsim library to simplify an ONNX model, remove redundant
operations, and optimize the graph structure.

Dependencies:
pip install onnx onnxsim

Usage:
python simplify_onnx.py --input input_model.onnx --output simplified_model.onnx

Arguments:
--input: Input ONNX model path
--output: Output path for the simplified ONNX model
"""
import argparse
import onnx
from onnxsim import simplify


def simplify_onnx_model(input_path, output_path):
    """
    Simplify an ONNX model with onnxsim.
    
    Args:
        input_path: Input ONNX model path.
        output_path: Output path for the simplified ONNX model.
    """
    # Load the ONNX model.
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)
    
    # Simplify the model.
    print("Simplifying model...")
    simplified_model, check = simplify(model)
    
    # Verify that simplification succeeded.
    assert check, "Simplification failed"
    print("Model simplification succeeded")
    
    # Save the simplified model.
    print(f"Saving simplified model to: {output_path}")
    onnx.save(simplified_model, output_path)
    print("Save complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify an ONNX model with onnxsim")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output path for the simplified ONNX model")
    
    args = parser.parse_args()
    simplify_onnx_model(args.input, args.output)
