# pylint: disable=E1102
"""
This script compares convergence of extended and conventional
1D convolutional layers in a regression context.

It implements a class RegressionConvergence1D that computes and visualizes
convergence for both conventional convolutional (Conv1d) and extended
convolutional (Xonv1d) models in a regression setting.
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from tqdm import tqdm

from xonv.loss_landscape import (
    filter_normalization,
    update_parameters_dict,
)

from xonv.model import Conv1dRegressionModel, Xonv1dRegressionModel
from xonv.utils import (
    checkpointsdir,
    make_experiment_name,
    plotsdir,
    process_sequence_arguments,
    query_arguments,
    upload_to_dropbox,
)

# Set up Seaborn and Matplotlib configurations
sns.set_style("whitegrid")
font = {"family": "serif", "style": "normal", "size": 8}
matplotlib.rc("font", **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

CONFIG_FILE: str = "regression_convergence_1d_comparison.json"


class RegressionConvergence1D:
    """
    Compares convergence of Conv1d and Xonv1d in a regression context.

    This class implements methods to compute and compare convergence for
    conventional 1D convolutional layers and extended 1D convolutional (Xonv) layers.

    Attributes:
        device (torch.device): The device (cpu/cuda) used for computation.
        true_conv_weights (Dict[str, Tensor]): The true weights of the
            convolutional model.
        x (Tensor): The input tensor for regression.
        y (Tensor): The target output tensor for regression.
        norm_rand_dirs (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Two random
            directions in the parameter space.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the RegressionConvergence1D object.

        Args:
            args (argparse.Namespace): The command line arguments.
        """
        # Set the device for computation
        self.device: torch.device = torch.device(
            f"cuda:{args.gpu_id}"
            if torch.cuda.is_available() and args.gpu_id > -1
            else "cpu"
        )

        # Initialize the conventional convolutional model
        conv_model: Conv1dRegressionModel = Conv1dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.num_layers,
        ).to(self.device)

        # Store the true weights of the convolutional model
        self.true_conv_weights: Dict[str, Tensor] = dict(
            conv_model.named_parameters()
        )

        # Generate input and target tensors
        self.x: Tensor = 1e-2 * torch.randn(
            [
                args.batchsize,
                args.num_channels,
                *args.input_size,
            ]
        ).to(self.device)
        self.y: Tensor = conv_model(self.x)

        # Generate normalized random directions
        self.norm_rand_dirs: Tuple[Dict[str, Tensor], Dict[str, Tensor]] = (
            filter_normalization(self.true_conv_weights),
            filter_normalization(self.true_conv_weights),
        )

    def set_conv_model_weights(
        self, weights: Dict[str, Tensor], conv_model: Conv1dRegressionModel
    ) -> None:
        """
        Set the weights of the convolutional model.

        Args:
            weights (Dict[str, Tensor]): The weights to set.
            conv_model (Conv1dRegressionModel): The model to update.
        """
        for name, param in conv_model.named_parameters():
            param.data = weights[name].data

    def train_conv(
        self, args: argparse.Namespace
    ) -> Dict[Tuple[float, float], List[float]]:
        """
        Train the conventional convolutional model and compute the objective log.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Dict[Tuple[float, float], List[float]]: The computed objective log.
        """
        obj_log: Dict[Tuple[float, float], List[float]] = {
            (args.vis_range[0], args.vis_range[0]): [],
            (args.vis_range[0], args.vis_range[1]): [],
            (args.vis_range[1], args.vis_range[0]): [],
            (args.vis_range[1], args.vis_range[1]): [],
        }

        for alpha, beta in obj_log.keys():
            # Update parameters based on random directions
            updated_params_dict: Dict[str, Tensor] = update_parameters_dict(
                self.true_conv_weights,
                self.norm_rand_dirs,
                alpha,
                beta,
            )

            # Initialize the convolutional model
            conv_model: Conv1dRegressionModel = Conv1dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.num_layers,
            ).to(self.device)

            # Set the weights of the model
            self.set_conv_model_weights(updated_params_dict, conv_model)

            # Initialize the optimizer
            conv_optimizer: torch.optim.Adam = torch.optim.Adam(
                conv_model.parameters(),
                lr=args.lr,
            )

            # Training loop
            for _ in tqdm(
                range(args.max_itrs),
                unit="epoch",
                colour="#B5F2A9",
                dynamic_ncols=True,
            ):
                # Forward pass
                y_hat: Tensor = conv_model(self.x)
                loss: Tensor = 0.5 * torch.norm(self.y - y_hat) ** 2

                # Backward pass
                conv_model.zero_grad()
                grads: List[Tensor] = torch.autograd.grad(
                    loss,
                    conv_model.parameters(),
                )
                for param, grad in zip(conv_model.parameters(), grads):
                    param.grad = grad

                # Update parameters
                conv_optimizer.step()
                conv_model.zero_grad()

                # Log the loss
                obj_log[(alpha, beta)].append(loss.detach().item())

        return obj_log

    def train_xonv(
        self, args: argparse.Namespace
    ) -> Dict[Tuple[float, float], List[float]]:
        """
        Train the extended convolutional (Xonv) model and compute the objective log.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Dict[Tuple[float, float], List[float]]: The computed objective log.
        """
        obj_log: Dict[Tuple[float, float], List[float]] = {
            (args.vis_range[0], args.vis_range[0]): [],
            (args.vis_range[0], args.vis_range[1]): [],
            (args.vis_range[1], args.vis_range[0]): [],
            (args.vis_range[1], args.vis_range[1]): [],
        }

        for alpha, beta in obj_log.keys():
            # Update parameters based on random directions
            updated_params_dict: Dict[str, Tensor] = update_parameters_dict(
                self.true_conv_weights,
                self.norm_rand_dirs,
                alpha,
                beta,
            )

            # Initialize the convolutional model
            conv_model: Conv1dRegressionModel = Conv1dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.num_layers,
            ).to(self.device)

            # Set the weights of the convolutional model
            self.set_conv_model_weights(updated_params_dict, conv_model)

            # Initialize the extended convolutional model
            xonv_model: Xonv1dRegressionModel = Xonv1dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.input_size[0],
                args.num_layers,
            ).to(self.device)

            # Initialize optimizers
            conv_optimizer: torch.optim.Adam = torch.optim.Adam(
                conv_model.parameters(),
                lr=args.lr,
            )
            xonv_optimizer: torch.optim.Adam = torch.optim.Adam(
                xonv_model.parameters(),
                lr=args.lr,
            )

            # Training loop
            for _ in tqdm(
                range(args.max_itrs),
                unit="epoch",
                colour="#B5F2A9",
                dynamic_ncols=True,
            ):
                # Inner loop for Xonv model
                for _ in range(args.inner_max_itrs):
                    # Forward pass
                    y_hat: Tensor = xonv_model(self.x)
                    mse_loss: Tensor = 0.5 * torch.norm(self.y - y_hat) ** 2

                    # Compute penalty
                    penalty: Tensor = sum(
                        args.gamma * torch.norm(xparam - param) ** 2
                        for xparam, param in zip(
                            xonv_model.parameters(),
                            conv_model.parameters(),
                        )
                    )

                    # Total loss
                    loss: Tensor = mse_loss + penalty

                    # Backward pass
                    xgrads: List[Tensor] = torch.autograd.grad(
                        loss,
                        xonv_model.parameters(),
                    )
                    for xparam, xgrad in zip(xonv_model.parameters(), xgrads):
                        xparam.grad = xgrad

                    # Update Xonv parameters
                    xonv_optimizer.step()
                    xonv_model.zero_grad()

                # Compute penalty for conv. model
                penalty: Tensor = sum(
                    args.gamma * torch.norm(xparam - param) ** 2
                    for xparam, param in zip(
                        xonv_model.parameters(),
                        conv_model.parameters(),
                    )
                )

                # Backward pass for conv. model
                grads: List[Tensor] = torch.autograd.grad(
                    penalty,
                    conv_model.parameters(),
                )
                for param, grad in zip(conv_model.parameters(), grads):
                    param.grad = grad

                # Update conv. parameters
                conv_optimizer.step()
                conv_model.zero_grad()

                # Log the loss
                obj_log[(alpha, beta)].append(loss.detach().item())

        return obj_log

    def load_checkpoint(
        self,
        args: argparse.Namespace,
        filepath: str,
    ) -> Tuple[
        Dict[Tuple[float, float], List[float]],
        Dict[Tuple[float, float], List[float]],
    ]:
        """
        Load model checkpoint.

        Args:
            args (argparse.Namespace): The command line arguments.
            filepath (str): Path to the checkpoint file.

        Returns:
            Tuple[Dict[Tuple[float, float], List[float]], Dict[Tuple[float, float], List[float]]]:
                Loaded conv_obj_log and xonv_obj_log.

        Raises:
            ValueError: If checkpoint does not exist.
        """
        if os.path.isfile(filepath):
            # Load checkpoint based on device
            if self.device == torch.device(type="cpu"):
                checkpoint: Dict[str, Any] = torch.load(
                    filepath,
                    map_location="cpu",
                )
            else:
                checkpoint: Dict[str, Any] = torch.load(filepath, weights_only=False)

            # Extract data from checkpoint
            self.x = checkpoint["x"]
            self.y = checkpoint["y"]
            self.true_conv_weights = checkpoint["true_conv_weights"]
            conv_obj_log: Dict[Tuple[float, float], List[float]] = checkpoint[
                "conv_obj_log"
            ]
            xonv_obj_log: Dict[Tuple[float, float], List[float]] = checkpoint[
                "xonv_obj_log"
            ]

        else:
            raise ValueError("Checkpoint does not exist.")

        return conv_obj_log, xonv_obj_log


if __name__ == "__main__":
    # Parse arguments
    args: argparse.Namespace = query_arguments(CONFIG_FILE)[0]
    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args)

    # Set checkpoint filepath
    checkpoint_filepath: str = os.path.join(
        checkpointsdir(args.experiment),
        "loss_objs_1d.pth",
    )

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize RegressionConvergence1D object
    reg_convergence: RegressionConvergence1D = RegressionConvergence1D(args)

    if args.phase == "compute":
        if not os.path.exists(checkpoint_filepath):
            # Compute convergence for conv. and Xonv models
            conv_obj_log: Dict[Tuple[float, float], List[float]] = (
                reg_convergence.train_conv(args)
            )
            xonv_obj_log: Dict[Tuple[float, float], List[float]] = (
                reg_convergence.train_xonv(args)
            )

            # Save checkpoint
            torch.save(
                {
                    "true_conv_weights": reg_convergence.true_conv_weights,
                    "x": reg_convergence.x,
                    "y": reg_convergence.y,
                    "conv_obj_log": conv_obj_log,
                    "xonv_obj_log": xonv_obj_log,
                    "args": args,
                },
                checkpoint_filepath,
            )

    # Load checkpoint
    conv_obj_log, xonv_obj_log = reg_convergence.load_checkpoint(
        args,
        checkpoint_filepath,
    )

    # Define colors for conv. and Xonv
    conv_color = "blue"
    xonv_color = "red"

    # Define different line styles to differentiate (alpha, beta) pairs
    style_idx = 0  # Initialize index for line styles

    # Create a single plot
    fig, ax = plt.subplots(figsize=(6, 1.7))

    # Normalize and plot results
    for alpha, beta in conv_obj_log.keys():
        # Normalize logs
        conv_obj_log[(alpha, beta)] = [
            obj / conv_obj_log[(alpha, beta)][0]
            for obj in conv_obj_log[(alpha, beta)]
        ]
        xonv_obj_log[(alpha, beta)] = [
            obj / xonv_obj_log[(alpha, beta)][0]
            for obj in xonv_obj_log[(alpha, beta)]
        ]

        # Plot conv. logs using the same color but different line styles
        ax.plot(
            np.arange(args.max_itrs),
            conv_obj_log[(alpha, beta)],
            color=conv_color,
            alpha=0.5,
            linewidth=0.9,
        )

        # Annotate conv. curve with (alpha, beta)
        ax.text(
            args.max_itrs - 15,  # Position near the end of the plot
            conv_obj_log[(alpha, beta)][-1] - 0.04,  # Y-value at the end
            f"({alpha}, {beta})",
            color=conv_color,
            fontsize=6,
            verticalalignment="center",
        )

        # Plot Xonv logs using the same color but different line styles
        ax.plot(
            np.arange(args.max_itrs),
            xonv_obj_log[(alpha, beta)],
            color=xonv_color,
            alpha=0.5,
            linewidth=0.9,
        )

        # Increment line style index
        style_idx += 1

    # Format the plot
    ax.ticklabel_format(axis="y", style="sci", useMathText=True)
    ax.set_title(
        "Extended 1D conv. converges in fewer iterations than regular 1D conv.",
        fontsize=8,
    )
    ax.set_ylabel("Normalized loss value")
    ax.set_xlabel("Iterations")
    plt.xlim([-1, args.max_itrs + 10])

    # Simplified legend with only two entries
    ax.legend(
        ["Regular 1D conv.", "Extended 1D conv."],
        loc="upper right",
        ncol=2,
        fontsize=7,
    )
    ax.grid(True)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Save the plot
    plt.savefig(
        os.path.join(
            plotsdir(args.experiment),
            "combined_training_loss_conv1d_annotations.png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )

    plt.close(fig)

    if args.upload_results:
        upload_to_dropbox(args)
