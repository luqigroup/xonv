from typing import Dict

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import torch

from xonv.layer import Xonv1D

sns.set_style("whitegrid")
font = {"family": "serif", "style": "normal", "size": 14}
matplotlib.rc("font", **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

# Constants
# Input signal length
INPUT_SIZE: int = 32

# Xonv1D layer parameters
KERNEL_SIZE: int = 5
IN_CHANNELS: int = 1
OUT_CHANNELS: int = 1
STRIDE: int = 1


def create_toeplitz_like_matrix() -> Dict[str, np.ndarray]:
    """
    Create a Toeplitz-like matrix representing the behavior of the Xonv1D layer.

    This function initializes an Xonv1D layer, sets its bias to zero, and then
    applies it to a series of input signals, each with a single non-zero element.
    The results are organized into a matrix where each column represents the
    layer's response to one input signal.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing Toeplitz-like matrices for both
        Xonv1D and Conv1D.
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Initialize the Xonv1D layer
    xonv = Xonv1D(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        INPUT_SIZE,
        stride=STRIDE,
    )
    # Initialize the Conv1D layer
    conv = torch.nn.Conv1d(
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
        "xonv": [],
        "conv": [],
    }

    for i in range(INPUT_SIZE):
        # Create an input signal with a single 1 at position i
        input_signal = torch.zeros(1, 1, INPUT_SIZE)
        input_signal[0, 0, i] = 1

        # Apply the Xonv1D and Conv1D layers
        with torch.no_grad():
            output_xonv = xonv(input_signal)
            output_conv = conv(input_signal)

            if i == 0:
                print("Xonv1D output shape:", output_xonv.shape)
                print("Conv1D output shape:", output_conv.shape)

        # Flatten the output and add it to the Toeplitz-like matrix
        toeplitz_matrix["xonv"].append(output_xonv.squeeze().numpy().flatten())
        toeplitz_matrix["conv"].append(output_conv.squeeze().numpy().flatten())

    # Convert to numpy array - transpose to get proper Toeplitz structure
    toeplitz_matrix = {k: np.array(v).T for k, v in toeplitz_matrix.items()}

    return toeplitz_matrix


def plot_toeplitz_matrix(toeplitz_matrix: Dict[str, np.ndarray]) -> None:
    """
    Plot the Toeplitz-like matrices as images.

    This function visualizes the Toeplitz-like matrices, using a diverging
    colormap to highlight positive and negative values.

    Args:
        toeplitz_matrix (Dict[str, np.ndarray]): The Toeplitz-like matrices to be plotted.
    """
    # Calculate the maximum absolute value for consistent color scaling
    max_val = max(
        np.abs(toeplitz_matrix["xonv"]).max(),
        np.abs(toeplitz_matrix["conv"]).max(),
    )
    vmin, vmax = -max_val, max_val

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    # Plot for the Toeplitz-like matrix for Xonv1D layer
    im1 = axes.imshow(
        toeplitz_matrix["xonv"],
        cmap="RdGy",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes.grid(False)
    fig.colorbar(im1, ax=axes, pad=0.01, fraction=0.047)
    axes.set_xlabel("Input position")
    axes.set_ylabel("Output position")
    axes.set_title("Toeplitz-like matrix for Xonv1D")
    plt.savefig("xonv1d.png", format="png", bbox_inches="tight", dpi=400)
    plt.close(fig)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    # Plot for the Toeplitz matrix for Conv1D layer
    im2 = axes.imshow(
        toeplitz_matrix["conv"],
        cmap="RdGy",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes.grid(False)
    fig.colorbar(im2, ax=axes, pad=0.01, fraction=0.047)
    axes.set_xlabel("Input position")
    axes.set_ylabel("Output position")
    axes.set_title("Toeplitz matrix for Conv1D layer")
    plt.savefig("conv1d.png", format="png", bbox_inches="tight", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    # Create and plot the Toeplitz-like matrix
    toeplitz_matrix = create_toeplitz_like_matrix()
    plot_toeplitz_matrix(toeplitz_matrix)

# Fatima comments to test