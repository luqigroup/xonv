"""Convolutional 1D and 2D Regression Models using PyTorch's Conv2d and Xonv2D layers."""

import torch

from xonv.layer import Xonv1D, Xonv2D


class Conv1dRegressionModel(torch.nn.Module):
    """
    Convolutional 1D Regression Model using PyTorch's Conv1d layers.

    Args:
        num_channels (int): Number of input and output channels for each layer.
        kernel_size (int): Size of the convolving kernel.
        num_layers (int, optional): Number of convolutional layers. Default is 5.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        num_layers: int = 5,
    ) -> None:
        super(Conv1dRegressionModel, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                torch.nn.Conv1d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                )
            )
            layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)


class Xonv1dRegressionModel(torch.nn.Module):
    """
    Custom Convolutional 1D Regression Model using Xonv1D layers.

    Args:
        num_channels (int): Number of input and output channels for each layer.
        kernel_size (int): Size of the convolving kernel.
        input_size (int): Length of the input signal.
        num_layers (int, optional): Number of convolutional layers. Default is 5.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        input_size: int,
        num_layers: int = 5,
    ) -> None:
        super(Xonv1dRegressionModel, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                Xonv1D(
                    num_channels,
                    num_channels,
                    kernel_size,
                    input_size,
                )
            )
            layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)


class Conv2dRegressionModel(torch.nn.Module):
    """
    Convolutional 2D Regression Model using PyTorch's Conv2d layers.

    Args:
        num_channels (int): Number of input and output channels for each layer.
        kernel_size (int): Size of the convolving kernel.
        num_layers (int, optional): Number of convolutional layers. Default is 5.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        num_layers: int = 5,
    ) -> None:
        super(Conv2dRegressionModel, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                torch.nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                )
            )
            layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)


class Xonv2dRegressionModel(torch.nn.Module):
    """
    Custom Convolutional 2D Regression Model using Xonv2D layers.

    Args:
        num_channels (int): Number of input and output channels for each layer.
        kernel_size (int): Size of the convolving kernel.
        input_size (tuple): Size of the input tensor (height, width).
        num_layers (int, optional): Number of convolutional layers. Default is 5.
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        input_size: tuple[int, int],
        num_layers: int = 5,
    ) -> None:
        super(Xonv2dRegressionModel, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                Xonv2D(
                    num_channels,
                    num_channels,
                    kernel_size,
                    input_size,
                )
            )
            layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)


if __name__ == "__main__":
    k = 3  # Kernel size
    nc = 4  # Number of channels
    num_layers = 5  # Number of layers
    nx = ny = 16  # Input tensor dimensions

    # Instantiate and print the Conv2dRegressionModel
    conv2d_model = Conv2dRegressionModel(nc, k, num_layers)
    print(conv2d_model)

    # Instantiate and print the Xonv2dRegressionModel
    xonv2d_model = Xonv2dRegressionModel(nc, k, (nx, ny), num_layers)
    print(xonv2d_model)

    # Random input tensor.
    x = torch.randn(10, nc, nx, ny)

    try:
        # Forward pass through the Conv2d model.
        y_conv2d = conv2d_model(x)
    except (RuntimeError, ValueError) as e:
        print(f"Error during forward pass of Conv2d model: {e}")

    try:
        # Forward pass through the Xonv2d model.
        y_xonv2d = xonv2d_model(x)
    except (RuntimeError, ValueError) as e:
        print(f"Error during forward pass of Xonv2d model: {e}")

    k = 5  # Kernel size
    nc = 4  # Number of channels
    num_layers = 5  # Number of layers
    nx = 128  # Input signal length

    # Instantiate and print the Conv1dRegressionModel
    conv1d_model = Conv1dRegressionModel(nc, k, num_layers)
    print(conv1d_model)

    # Instantiate and print the Xonv1dRegressionModel
    xonv1d_model = Xonv1dRegressionModel(nc, k, nx, num_layers)
    print(xonv1d_model)

    # Random input tensor.
    x = torch.randn(10, nc, nx)

    try:
        # Forward pass through the Conv1d model.
        y_conv1d = conv1d_model(x)
        print(f"Conv1d output shape: {y_conv1d.shape}")
    except (RuntimeError, ValueError) as e:
        print(f"Error during forward pass of Conv1d model: {e}")

    try:
        # Forward pass through the Xonv1d model.
        y_xonv1d = xonv1d_model(x)
        print(f"Xonv1d output shape: {y_xonv1d.shape}")
    except (RuntimeError, ValueError) as e:
        print(f"Error during forward pass of Xonv1d model: {e}")
