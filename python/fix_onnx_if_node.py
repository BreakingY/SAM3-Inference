
#!/usr/bin/env python3

import onnx
from onnx import helper, TensorProto
from pathlib import Path
import sys

INPUT = sys.argv[1] if len(sys.argv) > 1 else "onnx_weights/sam3_dynamic.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "onnx_weights/sam3_fixed.onnx"

input_path = Path(INPUT)
output_path = Path(OUTPUT)

print("Loading:", INPUT)
model = onnx.load(INPUT, load_external_data=True)

opset = next(
    x.version for x in model.opset_import
    if x.domain in ("", "ai.onnx")
)

print("ONNX opset:", opset)


def dim_key(d):
    if d.HasField("dim_value"):
        return ("value", d.dim_value)
    if d.HasField("dim_param"):
        return ("param", d.dim_param)
    return ("unknown", None)


fixed = 0


def process_graph(graph):
    global fixed

    for node in list(graph.node):

        # Recursively process nested subgraphs
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                process_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    process_graph(subgraph)

        if node.op_type != "If":
            continue

        branches = {
            attr.name: attr.g
            for attr in node.attribute
            if attr.type == onnx.AttributeProto.GRAPH
        }

        if "then_branch" not in branches or "else_branch" not in branches:
            continue

        then_graph = branches["then_branch"]
        else_graph = branches["else_branch"]

        if len(then_graph.output) != len(else_graph.output):
            print("Skip:", node.name, "- output count mismatch")
            continue

        for index, (then_out, else_out) in enumerate(
            zip(then_graph.output, else_graph.output)
        ):

            then_shape = then_out.type.tensor_type.shape
            else_shape = else_out.type.tensor_type.shape

            then_dims = list(then_shape.dim)
            else_dims = list(else_shape.dim)

            # Only fix: [D0, D1, ...] vs [D0, D1, ..., 1]
            if len(else_dims) != len(then_dims) + 1:
                continue

            if not else_dims[-1].HasField("dim_value"):
                continue

            if else_dims[-1].dim_value != 1:
                continue

            # Make sure all preceding dimensions match
            if [
                dim_key(d) for d in then_dims
            ] != [
                dim_key(d) for d in else_dims[:-1]
            ]:
                print("Skip:", node.name, "- prefix dimensions mismatch")
                continue

            old_name = else_out.name
            new_name = old_name + "_trt_squeezed"
            axes_name = old_name + "_trt_squeeze_axes"

            # Avoid duplicate names
            existing_names = {
                n for n in
                list(else_graph.initializer)
                for _ in [n.name]
            }
            if axes_name in existing_names:
                raise RuntimeError("Duplicate initializer: " + axes_name)

            if opset >= 13:
                else_graph.initializer.append(
                    helper.make_tensor(
                        axes_name,
                        TensorProto.INT64,
                        [1],
                        [-1]
                    )
                )

                squeeze = helper.make_node(
                    "Squeeze",
                    inputs=[old_name, axes_name],
                    outputs=[new_name],
                    name=node.name + "_squeeze_fix_" + str(index)
                )
            else:
                squeeze = helper.make_node(
                    "Squeeze",
                    inputs=[old_name],
                    outputs=[new_name],
                    name=node.name + "_squeeze_fix_" + str(index),
                    axes=[-1]
                )

            else_graph.node.append(squeeze)

            # Change branch output to the squeezed tensor
            else_out.name = new_name

            # Copy the exact output shape from then_branch
            del else_out.type.tensor_type.shape.dim[:]

            for d in then_dims:
                new_dim = else_out.type.tensor_type.shape.dim.add()
                new_dim.CopyFrom(d)

            print(
                "Fixed:", node.name,
                "|", old_name,
                "|", [dim_key(d) for d in else_dims],
                "->", [dim_key(d) for d in then_dims]
            )

            fixed += 1


process_graph(model.graph)

print("\nTotal fixed outputs:", fixed)

if fixed == 0:
    raise RuntimeError("No matching If branch found. Model was not modified.")

# Save ONNX and external weights together
output_path.parent.mkdir(parents=True, exist_ok=True)
weight_name = output_path.stem + "_weights.bin"

onnx.save_model(
    model,
    str(output_path),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=weight_name,
    size_threshold=0
)

# Check model from file instead of serializing the whole model in memory
print("Checking ONNX model...")
onnx.checker.check_model(str(output_path))

print("ONNX check passed!")
