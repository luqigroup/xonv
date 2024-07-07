import torch
import matplotlib.pyplot as plt
from xonv.layer import Xonv2D
import numpy as np
from typing import Tuple

# Constants
# Input image size.
INPUT_SIZE: Tuple[int, int] = (8, 8)

# Xonv2D layer parameters.
KERNEL_SIZE: int = 3
IN_CHANNELS: int = 1
OUT_CHANNELS: int = 1


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
    xonv = Xonv2D(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_SIZE)

    # Set the bias to zero
    with torch.no_grad():
        xonv.bias.zero_()

    # Create the Toeplitz-like matrix
    toeplitz_matrix = []

    for i in range(INPUT_SIZE[0]):
        for j in range(INPUT_SIZE[1]):
            # Create an input image with a single 1 at position (i, j)
            input_image = torch.zeros(1, 1, *INPUT_SIZE)
            input_image[0, 0, i, j] = 1

            # Apply the Xonv2D layer
            with torch.no_grad():
                output = xonv(input_image)

            # Flatten the output and add it to the Toeplitz-like matrix
            toeplitz_matrix.append(output.squeeze().numpy().flatten())

    # Convert to numpy array
    toeplitz_matrix = np.array(toeplitz_matrix)

    return toeplitz_matrix


def plot_toeplitz_matrix(toeplitz_matrix: np.ndarray) -> None:
    """
    Plot the Toeplitz-like matrix as an image.

    This function visualizes the Toeplitz-like matrix, masking zero elements
    and using a diverging colormap to highlight positive and negative values.

    Args:
        toeplitz_matrix (np.ndarray): The Toeplitz-like matrix to be plotted.
    """

    plt.figure(dpi=200)
    plt.imshow(toeplitz_matrix,
               cmap="RdGy",
               aspect='equal',
               vmin=-2.0,
               vmax=2.0)
    plt.colorbar(pad=0.01)
    plt.title("Toeplitz-like matrix for Xonv2D layer")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create and plot the Toeplitz-like matrix
    toeplitz_matrix = create_toeplitz_like_matrix()
    plot_toeplitz_matrix(toeplitz_matrix)
