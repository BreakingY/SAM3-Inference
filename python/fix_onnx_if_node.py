#!/usr/bin/env python3
"""Fix the If node shape mismatch in sam3_dynamic.onnx for TensorRT compatibility.

The If node at /sam3/If has branches with shapes [6,1,200] and [6,1,200,1].
TensorRT requires both branches to have the same shape.
This script adds a Squeeze to the [6,1,200,1] branch to make it [6,1,200].
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
from pathlib import Path

import sys
INPUT  = sys.argv[1] if len(sys.argv) > 1 else "onnx_weights/sam3_dynamic.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "onnx_weights/sam3_fixed.onnx"

print("Loading ONNX model (this may take a while due to external data)...")
model = onnx.load(INPUT, load_external_data=True)
print(f"ir_version: {model.ir_version}, opset: {model.opset_import[0].version}")

# Find the If node and fix each subgraph output
fixed = 0
for node in model.graph.node:
    if node.op_type != "If":
        continue
    print(f"Found If node: {node.name}")

    for attr in node.attribute:
        if attr.g is None:
            continue
        subgraph = attr.g

        for out in subgraph.output:
            shape = out.type.tensor_type.shape
            if shape is None:
                continue
            dims = [d.dim_value if d.dim_value else d.dim_param for d in shape.dim]
            print(f"  branch '{attr.name}' output '{out.name}' shape: {dims}")

            # Fix the [*,*,*,1] branch by inserting a Squeeze on axis -1
            if len(dims) == 4 and (dims[-1] == 1 or dims[-1] == ""):
                squeeze_out_name = out.name + "_squeezed"

                # axes input for Squeeze (opset 13+)
                axes_name = out.name + "_squeeze_axes"
                axes_init = helper.make_tensor(
                    axes_name, TensorProto.INT64, [1], [-1]
                )
                subgraph.initializer.append(axes_init)

                squeeze_node = helper.make_node(
                    "Squeeze",
                    inputs=[out.name, axes_name],
                    outputs=[squeeze_out_name],
                    name=out.name + "_squeeze_fix",
                )
                subgraph.node.append(squeeze_node)

                # Update the subgraph output to point to squeezed tensor
                new_shape = [d for d in dims[:-1]]
                out.name = squeeze_out_name
                del out.type.tensor_type.shape.dim[:]
                for d in new_shape:
                    dim = out.type.tensor_type.shape.dim.add()
                    if isinstance(d, int) and d > 0:
                        dim.dim_value = d
                    else:
                        dim.dim_param = str(d) if d else "?"
                print(f"    → fixed to shape: {new_shape}")
                fixed += 1

print(f"\nFixed {fixed} branch(es).")

print(f"Saving fixed model to {OUTPUT} ...")
onnx.save_model(
    model,
    OUTPUT,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="sam3_fixed_weights.bin",
)
print("Done.")
