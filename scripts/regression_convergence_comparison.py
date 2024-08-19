# pylint: disable=E1102
"""
This script compares convergence of extended and conventional
convolutional layers in a regression context.

It implements a class RegressionConvergence that computes and visualizes
convergence for both conventional convolutional (Conv2d) and extended
convolutional (Xonv2d) models in a regression setting.
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
from xonv.model import Conv2dRegressionModel, Xonv2dRegressionModel
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
font = {"family": "serif", "style": "normal", "size": 10}
matplotlib.rc("font", **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

CONFIG_FILE: str = "regression_convergence_comparison.json"


class RegressionConvergence:
    """
    Compares convergence of Conv2d and Xonv2d in a regression context.

    This class implements methods to compute and compare convergence for
    conventional convolutional layers and extended convolutional (Xonv) layers.

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
        Initialize the RegressionConvergence object.

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
        conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.num_layers,
        ).to(self.device)

        # Store the true weights of the convolutional model
        self.true_conv_weights: Dict[str, Tensor] = dict(conv_model.named_parameters())

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
        self, weights: Dict[str, Tensor], conv_model: Conv2dRegressionModel
    ) -> None:
        """
        Set the weights of the convolutional model.

        Args:
            weights (Dict[str, Tensor]): The weights to set.
            conv_model (Conv2dRegressionModel): The model to update.
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
            conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
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
            conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.num_layers,
            ).to(self.device)

            # Set the weights of the convolutional model
            self.set_conv_model_weights(updated_params_dict, conv_model)

            # Initialize the extended convolutional model
            xonv_model: Xonv2dRegressionModel = Xonv2dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.input_size,
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

                # Compute penalty for Conv model
                penalty: Tensor = sum(
                    args.gamma * torch.norm(xparam - param) ** 2
                    for xparam, param in zip(
                        xonv_model.parameters(),
                        conv_model.parameters(),
                    )
                )

                # Backward pass for Conv model
                grads: List[Tensor] = torch.autograd.grad(
                    penalty,
                    conv_model.parameters(),
                )
                for param, grad in zip(conv_model.parameters(), grads):
                    param.grad = grad

                # Update Conv parameters
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
        Dict[Tuple[float, float], List[float]], Dict[Tuple[float, float], List[float]]
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
                checkpoint: Dict[str, Any] = torch.load(filepath)

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
        "loss_objs.pth",
    )

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize RegressionConvergence object
    reg_convergence: RegressionConvergence = RegressionConvergence(args)

    if args.phase == "compute":
        if not os.path.exists(checkpoint_filepath):
            # Compute convergence for Conv and Xonv models
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

    # Normalize and plot results
    for alpha, beta in conv_obj_log.keys():
        # Normalize logs
        conv_obj_log[(alpha, beta)] = [
            obj / conv_obj_log[(alpha, beta)][0] for obj in conv_obj_log[(alpha, beta)]
        ]
        xonv_obj_log[(alpha, beta)] = [
            obj / xonv_obj_log[(alpha, beta)][0] for obj in xonv_obj_log[(alpha, beta)]
        ]

        # Create plot
        fig = plt.figure("training logs", figsize=(7, 4))

        plt.plot(
            np.arange(args.max_itrs),
            conv_obj_log[(alpha, beta)],
            label="training loss: Conv2d",
            color="orange",
            alpha=1.0,
        )
        plt.plot(
            np.arange(args.max_itrs),
            xonv_obj_log[(alpha, beta)],
            label="training loss: Xonv2d",
            color="k",
            alpha=0.8,
        )

        plt.ticklabel_format(axis="y", style="sci", useMathText=True)

        plt.title("Training loss over training")
        plt.ylabel("Normalized loss value")
        plt.xlabel("Epochs")
        plt.legend()

        # Save plot
        plt.savefig(
            os.path.join(
                plotsdir(args.experiment),
                f"log_alpha-{alpha}_beta-{beta}.png",
            ),
            format="png",
            bbox_inches="tight",
            dpi=400,
            pad_inches=0.02,
        )

        plt.close(fig)

    if args.upload_results:
        upload_to_dropbox(args)
