from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from xonv.layer import Xonv2D

# Constants
# Input image size.
INPUT_SIZE: Tuple[int, int] = (8, 8)

# Xonv2D layer parameters.
KERNEL_SIZE: int = 3
IN_CHANNELS: int = 1
OUT_CHANNELS: int = 1
STRIDE: int = 1


def create_toeplitz_like_matrix() -> np.ndarray:
    """
    Create a Toeplitz-like matrix representing the behavior of the Xonv2D layer.

    This function initializes an Xonv2D layer, sets its bias to zero, and then
    applies it to a series of input images, each with a single non-zero pixel.
    The results are organized into a matrix where each column represents the
    layer's response to one input image.

    Returns:
        np.ndarray: A 2D numpy array representing the Toeplitz-like matrix.
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Initialize the Xonv2D layer
    xonv = Xonv2D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        INPUT_SIZE,
        stride=STRIDE,
    )
    # Initialize the Conv2D layer
    conv = torch.nn.Conv2d(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        padding=KERNEL_SIZE // 2,
        stride=STRIDE,
    )

    # Set the bias to zero
    with torch.no_grad():
        xonv.bias.zero_()
        conv.bias.zero_()

    # Create the Toeplitz-like matrix
    toeplitz_matrix = {
        'xonv': [],
        'conv': [],
    }

    for i in range(INPUT_SIZE[0]):
        for j in range(INPUT_SIZE[1]):
            # Create an input image with a single 1 at position (i, j)
            input_image = torch.zeros(1, 1, *INPUT_SIZE)
            input_image[0, 0, i, j] = 1

            # Apply the Xonv2D and Conv2D layers.
            with torch.no_grad():
                output_xonv = xonv(input_image)
                output_conv = conv(input_image)

                if i == 0 and j == 0:
                    print("Xonv2D output shape:", output_xonv.shape)
                    print("Conv2D output shape:", output_conv.shape)

            # Flatten the output and add it to the Toeplitz-like matrix.
            toeplitz_matrix['xonv'].append(
                output_xonv.squeeze().numpy().flatten())
            toeplitz_matrix['conv'].append(
                output_conv.squeeze().numpy().flatten())

    # Convert to numpy array
    toeplitz_matrix = {k: np.array(v).T for k, v in toeplitz_matrix.items()}

    return toeplitz_matrix


def plot_toeplitz_matrix(toeplitz_matrix: np.ndarray) -> None:
    """
    Plot the Toeplitz-like matrix as an image.

    This function visualizes the Toeplitz-like matrix, masking zero elements
    and using a diverging colormap to highlight positive and negative values.

    Args:
        toeplitz_matrix (np.ndarray): The Toeplitz-like matrix to be plotted.
    """

    fig, axes = plt.subplots(1, 2, dpi=150, figsize=(12, 6))

    # Plot for the Toeplitz-like matrix for Xonv2D layer
    im1 = axes[1].imshow(toeplitz_matrix['xonv'],
                         cmap="RdGy",
                         aspect='equal',
                         vmin=-0.25,
                         vmax=0.25)
    fig.colorbar(im1, ax=axes[1], pad=0.01, fraction=0.047)
    axes[1].set_xlabel("Input dimension")
    axes[1].set_ylabel("Output dimension")
    axes[1].set_title("Toeplitz-like matrix for Xonv2D layer")

    # Plot for the Toeplitz matrix for Conv2D layer
    im2 = axes[0].imshow(toeplitz_matrix['conv'],
                         cmap="RdGy",
                         aspect='equal',
                         vmin=-0.25,
                         vmax=0.25)
    fig.colorbar(im2, ax=axes[0], pad=0.01, fraction=0.047)
    axes[0].set_xlabel("Input dimension")
    axes[0].set_ylabel("Output dimension")
    axes[0].set_title("Toeplitz matrix for Conv2D layer")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create and plot the Toeplitz-like matrix
    toeplitz_matrix = create_toeplitz_like_matrix()
    plot_toeplitz_matrix(toeplitz_matrix)
