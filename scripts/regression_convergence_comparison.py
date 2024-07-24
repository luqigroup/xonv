# pylint: disable=E1102
"""
This script compares loss landscapes of extended and conventional
convolutional layers in a regression context.

It implements a class RegressionLossLandscape that computes and visualizes
loss landscapes for both conventional convolutional (Conv2d) and extended
convolutional (Xonv2d) models in a regression setting.
"""

from typing import Dict, Tuple, List, Optional
import os
import argparse
from tqdm import tqdm
import torch
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from xonv.model import Conv2dRegressionModel, Xonv2dRegressionModel
from xonv.loss_landscape import (
    filter_normalization,
    update_parameters_dict,
    plot_loss_landscape,
)
from xonv.utils import (
    query_arguments,
    make_experiment_name,
    process_sequence_arguments,
    checkpointsdir,
    upload_to_dropbox,
    plotsdir,
)

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

CONFIG_FILE: str = 'regression_convergence_comparison.json'


class RegressionConvergence:
    """
    Compares convergence of Conv2d and Xonv2d in a regression context.

    This class implements methods to compute and compare loss landscapes for
    conventional convolutional layers and extended convolutional (Xonv) layers.

    Attributes:
        device (torch.device): The device (cpu/cuda) used for computation.

        conv_model (Conv2dRegressionModel): The conventional convolutional
            regression model.
        true_conv_weights (Dict[str, Tensor]): The true weights of the
            convolutional model.
        x (Tensor): The input tensor for regression.
        y (Tensor): The target output tensor for regression.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the RegressionLossLandscape object.

        Args:
            args (argparse.Namespace): The command line arguments.
        """
        # Set the device based on CUDA availability and input arguments
        self.device: torch.device = torch.device(
            f'cuda:{args.gpu_id}'
            if torch.cuda.is_available() and args.gpu_id > -1 else 'cpu')

        # Initialize the conventional convolutional model
        conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.num_layers,
        ).to(self.device)

        # Store the true weights of the convolutional model
        self.true_conv_weights: Dict[str, Tensor] = dict(
            conv_model.named_parameters())

        # Generate input data and compute target output
        self.x: Tensor = 1e-2 * torch.randn([
            args.batchsize,
            args.num_channels,
            *args.input_size,
        ]).to(self.device)
        self.y: Tensor = conv_model(self.x)

        # Generate two random directions in the parameter space
        self.norm_rand_dirs: Tuple[Dict[str, Tensor], Dict[str, Tensor]] = (
            filter_normalization(self.true_conv_weights),
            filter_normalization(self.true_conv_weights),
        )

    def set_conv_model_weights(self, weights: Dict[str, Tensor],
                               conv_model) -> None:
        """
        Set the weights of the convolutional model.

        Args:
            weights (Dict[str, Tensor]): The weights to set.
        """
        for name, param in conv_model.named_parameters():
            param.data = weights[name].data

    def train_conv(self, args: argparse.Namespace) -> Tensor:
        """
        Compute the loss landscape for the conventional convolutional model.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Tensor: The computed loss landscape.
        """

        obj_log = {
            (args.vis_range[0], args.vis_range[0]): [],
            (args.vis_range[0], args.vis_range[1]): [],
            (args.vis_range[1], args.vis_range[0]): [],
            (args.vis_range[1], args.vis_range[1]): [],
        }

        for alpha, beta in obj_log.keys():
            updated_params_dict: Dict[str, Tensor] = update_parameters_dict(
                self.true_conv_weights,
                self.norm_rand_dirs,
                alpha,
                beta,
            )

            # Initialize the conventional convolutional model
            conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.num_layers,
            ).to(self.device)

            self.set_conv_model_weights(updated_params_dict, conv_model)

            # Set up the optimizer for the Xonv model
            conv_optimizer: torch.optim.Adam = torch.optim.Adam(
                conv_model.parameters(),
                lr=args.lr,
            )

            for _ in tqdm(
                    range(args.max_itrs),
                    unit='epoch',
                    colour='#B5F2A9',
                    dynamic_ncols=True,
            ):
                # Compute the model output using the given weights
                y_hat: Tensor = conv_model(self.x)

                # Compute and return the mean squared error loss
                loss: Tensor = 0.5 * torch.norm(self.y - y_hat)**2

                # Compute gradients and update parameters.
                conv_model.zero_grad()

                # Compute gradients and update parameters
                grads: List[Tensor] = torch.autograd.grad(
                    loss,
                    conv_model.parameters(),
                )
                for param, grad in zip(conv_model.parameters(), grads):
                    param.grad = grad

                conv_optimizer.step()
                conv_model.zero_grad()

                obj_log[(alpha, beta)].append(loss.detach().item())

        return obj_log

    def train_xonv(self, args: argparse.Namespace) -> Tensor:
        """
        Compute the loss landscape for the conventional convolutional model.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Tensor: The computed loss landscape.
        """

        obj_log = {
            (args.vis_range[0], args.vis_range[0]): [],
            (args.vis_range[0], args.vis_range[1]): [],
            (args.vis_range[1], args.vis_range[0]): [],
            (args.vis_range[1], args.vis_range[1]): [],
        }

        for alpha, beta in obj_log.keys():
            updated_params_dict: Dict[str, Tensor] = update_parameters_dict(
                self.true_conv_weights,
                self.norm_rand_dirs,
                alpha,
                beta,
            )

            # Initialize the conventional convolutional model
            conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.num_layers,
            ).to(self.device)

            self.set_conv_model_weights(updated_params_dict, conv_model)

            # Initialize the Xonv model
            xonv_model: Xonv2dRegressionModel = Xonv2dRegressionModel(
                args.num_channels,
                args.kernel_size,
                args.input_size,
                args.num_layers,
            ).to(self.device)

            # Set up the optimizer for the Xonv model
            conv_optimizer: torch.optim.Adam = torch.optim.Adam(
                conv_model.parameters(),
                lr=args.lr,
            )
            # Set up the optimizer for the Xonv model
            xonv_optimizer: torch.optim.Adam = torch.optim.Adam(
                xonv_model.parameters(),
                lr=args.lr,
            )

            for _ in tqdm(
                    range(args.max_itrs),
                    unit='epoch',
                    colour='#B5F2A9',
                    dynamic_ncols=True,
            ):

                for _ in range(args.inner_max_itrs):
                    # Compute the model output using the given weights
                    y_hat: Tensor = xonv_model(self.x)

                    # Compute and return the mean squared error loss
                    loss: Tensor = 0.5 * torch.norm(self.y - y_hat)**2

                    # Compute the penalty term (L2 distance between Xonv and Conv
                    # weights)
                    penalty: Tensor = sum(args.gamma *
                                          torch.norm(xparam - param)**2
                                          for xparam, param in zip(
                                              xonv_model.parameters(),
                                              conv_model.parameters(),
                                          ))

                    # Compute the total loss (MSE + penalty)
                    loss: Tensor = 0.5 * torch.norm(
                        self.y - xonv_model(self.x))**2 + penalty

                    # Compute gradients and update parameters
                    xgrads: List[Tensor] = torch.autograd.grad(
                        loss,
                        xonv_model.parameters(),
                    )
                    for xparam, xgrad in zip(xonv_model.parameters(), xgrads):
                        xparam.grad = xgrad

                    xonv_optimizer.step()
                    xonv_model.zero_grad()

                # Compute the penalty term (L2 distance between Xonv and Conv
                # weights)
                penalty: Tensor = sum(args.gamma *
                                      torch.norm(xparam - param)**2
                                      for xparam, param in zip(
                                          xonv_model.parameters(),
                                          conv_model.parameters(),
                                      ))

                # Compute gradients and update parameters
                grads: List[Tensor] = torch.autograd.grad(
                    penalty,
                    conv_model.parameters(),
                )
                for param, grad in zip(conv_model.parameters(), grads):
                    param.grad = grad

                conv_optimizer.step()
                conv_model.zero_grad()

                obj_log[(alpha, beta)].append(loss.detach().item())

        return obj_log

    def load_checkpoint(
        self,
        args: argparse.Namespace,
        filepath: str,
    ) -> Tuple[Tensor, Tensor]:
        """
        Load model checkpoint.

        Args:
            args (argparse.Namespace): The command line arguments.
            filepath (str): Path to the checkpoint file.

        Returns:
            Tuple[Tensor, Tensor]: Loaded conv_loss_landscape and
                xonv_loss_landscape.

        Raises:
            ValueError: If checkpoint does not exist.
        """
        if os.path.isfile(filepath):
            if self.device == torch.device(type='cpu'):
                checkpoint: Dict[str, Any] = torch.load(
                    filepath,
                    map_location='cpu',
                )
            else:
                checkpoint: Dict[str, Any] = torch.load(filepath)

            self.x = checkpoint['x']
            self.y = checkpoint['y']
            self.true_conv_weights = checkpoint['true_conv_weights']
            conv_obj_log: Tensor = checkpoint['conv_obj_log']
            xonv_obj_log: Tensor = checkpoint['xonv_obj_log']

        else:
            raise ValueError('Checkpoint does not exist.')

        return conv_obj_log, xonv_obj_log


if __name__ == '__main__':
    # Read input arguments from a JSON file and process them
    args: argparse.Namespace = query_arguments(CONFIG_FILE)[0]
    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args)

    checkpoint_filepath: str = os.path.join(
        checkpointsdir(args.experiment),
        'loss_objs.pth',
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize the RegressionLossLandscape object
    reg_convergence: RegressionConvergence = RegressionConvergence(args)

    if args.phase == 'compute':
        if not os.path.exists(checkpoint_filepath):
            # Compute the loss landscapes for both models
            conv_obj_log: Tensor = reg_convergence.train_conv(args)
            xonv_obj_log: Tensor = reg_convergence.train_xonv(args)

            # Save the results
            torch.save(
                {
                    'true_conv_weights': reg_convergence.true_conv_weights,
                    'x': reg_convergence.x,
                    'y': reg_convergence.y,
                    'conv_obj_log': conv_obj_log,
                    'xonv_obj_log': xonv_obj_log,
                    'args': args,
                },
                checkpoint_filepath,
            )

    # Load the computed loss landscapes
    conv_obj_log, xonv_obj_log = reg_convergence.load_checkpoint(
        args,
        checkpoint_filepath,
    )

    for alpha, beta in conv_obj_log.keys():

        conv_obj_log[(alpha, beta)] = [
            obj / conv_obj_log[(alpha, beta)][0]
            for obj in conv_obj_log[(alpha, beta)]
        ]
        xonv_obj_log[(alpha, beta)] = [
            obj / xonv_obj_log[(alpha, beta)][0]
            for obj in xonv_obj_log[(alpha, beta)]
        ]

        # Create a new figure for plotting
        fig = plt.figure("training logs", figsize=(7, 4))

        plt.plot(
            np.arange(args.max_itrs),
            conv_obj_log[(alpha, beta)],
            label='training loss: Conv2d',
            color="orange",
            alpha=1.0,
        )
        plt.plot(
            np.arange(args.max_itrs),
            xonv_obj_log[(alpha, beta)],
            label='training loss: Xonv2d',
            color="k",
            alpha=0.8,
        )

        # Format y-axis labels using scientific notation
        plt.ticklabel_format(axis="y", style="sci", useMathText=True)

        # Set plot title and labels
        plt.title("Training loss over training")
        plt.ylabel("Normalized loss value")
        plt.xlabel("Epochs")
        plt.legend()

        # Save the plot
        plt.savefig(
            os.path.join(
                plotsdir(args.experiment),
                "log_alpha-" + str(alpha) + "_beta-" + str(beta) + ".png",
            ),
            format="png",
            bbox_inches="tight",
            dpi=400,
            pad_inches=.02,
        )

        # Close the figure to release memory
        plt.close(fig)

    # Upload results to Dropbox if specified
    if args.upload_results:
        upload_to_dropbox(args)
