<h1 align="center">xonv</h1>

This repository contains the code for extended convolutional layers.
These layers are akin to the convolutional layers in PyTorch, but with
the key difference that they have spatially varying kernels.

## Installation

Run the commands below to install the required packages. Make sure to adapt the `pytorch-cuda` version to your CUDA version in `environment.yml`. Use `environment-cpu.yml` instead for CPU-only lightweight installations.

```bash
git clone https://github.com/alisiahkoohi/xonv
cd xonv/
conda env create -f environment.yml
conda activate xonv
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate xonv`, the
following times.

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
python create_toeplitz_like_matrix.py
```


## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi




