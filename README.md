<h1 align="center">xonv: Extended convolutional layers</h1>

This repository contains the code for extended convolutional layers.
These layers are akin to the convolutional layers in PyTorch, but with
the key difference that they have spatially varying kernels.

## Installation

For further development and to run the examples, clone the repository
and install the package in editable mode. **Make sure to adapt the
hardcoded CUDA version in `setup.cfg` to the one installed on your system.**

```bash
# Create a new conda environment.
conda create --name xonv python
conda activate xonv

# Clone the repository and install the package in editable mode.
git clone ttps://github.com/alisiahkoohi/xonv
cd xonv/
pip install -e .
```

Run the command below to install the package to be used in your Python environment.

```bash
pip install git+https://github.com/alisiahkoohi/xonv
```


## Usage

The extended convolutional layers can be used as a drop-in replacement
for the PyTorch convolutional layers. The following example demonstrates
how to use the extended convolutional layers:

```python
from xonv.layer import Xonv2D

input_size = (32, 32)  # Height, Width of input
in_channels = 3
out_channels = 16
kernel_size = 3

layer = Xonv2D(in_channels, out_channels, kernel_size, input_size)
input_tensor = torch.randn(1, in_channels, *input_size)
output = layer(input_tensor)
print(output.shape)  # Should be [1, 16, 32, 32]
```

## Examples

To visualize the toeplitz-like matrix associated with the convolutional layer, run the following command:

```bash
python scripts/create_toeplitz_like_matrix.py
```


## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi




