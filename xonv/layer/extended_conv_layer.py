import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Xonv2D(nn.Module):
    """
    A 2D convolutional layer with spatially varying kernel.

    This layer performs a 2D convolution operation where each spatial location
    in the input has its own unique convolutional kernel. This allows for
    spatially varying filters across the input tensor.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        input_size (Tuple[int, int]): Height and width of the input tensor

    Shape:
        - Input: (batch_size, in_channels, height, width)
        - Output: (batch_size, out_channels, height, width)

    Examples:
        >>> input_size = (32, 32)
        >>> layer = Xonv2D(3, 16, 3, input_size)
        >>> input_tensor = torch.randn(1, 3, 32, 32)
        >>> output = layer(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 16, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_size: Tuple[int, int],
    ):
        super(Xonv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size

        # Create a weight tensor for each spatial location.
        # Shape: (height, width, out_channels, in_channels, kernel_size,
        # kernel_size).
        self.weights = nn.Parameter(
            torch.randn(
                input_size[0],
                input_size[1],
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
            ))

        # Create a bias tensor for each spatial location
        # Shape: (height, width, out_channels)
        self.bias = nn.Parameter(
            torch.randn(
                input_size[0],
                input_size[1],
                out_channels,
            ))

        # Initialize weights using Kaiming uniform initialization.
        torch.nn.init.kaiming_uniform_(self.weights, a=1.)

    def __repr__(self):
        """String representation of the layer.
        """
        return (f'Xonv2D(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'input_size={self.input_size})')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels,
            height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels,
            height, width)
        """
        batch_size, _, height, width = x.shape
        pad = self.kernel_size // 2

        # Pad the input tensor.
        if self.kernel_size % 2 == 1:
            x_padded = F.pad(x, (pad, pad, pad, pad))
        else:
            x_padded = F.pad(x, (pad - 1, pad, pad - 1, pad))

        # Extract patches from the padded input.
        # Shape: (batch_size, height, width, in_channels, kernel_size,
        # kernel_size).
        patches = x_padded.unfold(2, self.kernel_size, 1).unfold(
            3,
            self.kernel_size,
            1,
        )

        # Reshape patches for batch matrix multiplication.
        # Shape: (batch_size, height, width, in_channels * kernel_size *
        # kernel_size).
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, height, width, -1)

        # Reshape weights for batch matrix multiplication.
        # Shape: (height, width, out_channels, in_channels * kernel_size *
        # kernel_size).
        weights = self.weights.view(height, width, self.out_channels, -1)

        # Perform batch matrix multiplication
        # Shape: (batch_size, height, width, out_channels)
        output = torch.matmul(
            patches.unsqueeze(3),
            weights.permute(0, 1, 3, 2),
        )
        output = output.squeeze(3)

        # Add bias
        output += self.bias

        # Reshape to standard output format
        # Shape: (batch_size, out_channels, height, width)
        output = output.permute(0, 3, 1, 2).contiguous()

        return output


# Example usage
if __name__ == "__main__":
    input_size = (32, 32)  # Height, Width of input
    in_channels = 3
    out_channels = 16
    kernel_size = 3

    layer = Xonv2D(in_channels, out_channels, kernel_size, input_size)
    input_tensor = torch.randn(1, in_channels, *input_size)
    output = layer(input_tensor)
    print(output.shape)  # Should be [1, 16, 32, 32]
