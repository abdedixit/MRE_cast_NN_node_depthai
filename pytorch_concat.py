#! /usr/bin/env python3
import torch
from torch import nn
import blobconverter

class CatImgs(nn.Module):
    def forward(self, img1, img2):
        return torch.cat((img1, img2), 1)

# Define the expected input shape (dummy input)
height = 400
width = 640
shape = (height, width)
X = torch.ones(shape, dtype=torch.float16)

onnx_file = "concat.onnx"

print(f"Writing to {onnx_file}")
torch.onnx.export(
    CatImgs(),
    (X, X),
    onnx_file,
    opset_version=12,
    do_constant_folding=True,
    input_names = ['img1', 'img2'], # Optional
    output_names = ['output'], # Optional
)

# No need for onnx-simplifier here

# Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_file,
    data_type="FP16",
    shaves=6,
    use_cache=False,
    output_dir=".",
    optimizer_params=[]
)