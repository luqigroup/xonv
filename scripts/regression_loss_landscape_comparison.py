# pylint: disable=E1102
"""This script compares loss landscapes of extended and conventional
convolutional layers to in a regression context."""

from typing import Dict, Tuple, List
import os
import argparse
from tqdm import tqdm
import torch
from torch import Tensor

from xonv.model import Conv2dRegressionModel, Xonv2dRegressionModel
from xonv.loss_landscape import filter_normalization, update_parameters_dict
from xonv.utils import (
    query_arguments,
    make_experiment_name,
    process_sequence_arguments,
    checkpointsdir,
    upload_to_dropbox,
)

CONFIG_FILE = 'regression_loss_landscape_comparison.json'


class RegressionLossLandscape:
    """Comparing loss landscapes of Conv2d and Xonv2d in a regression context.

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
        self.conv_model: Conv2dRegressionModel = Conv2dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.num_layers,
        ).to(self.device)

        # Store the true weights of the convolutional model
        self.true_conv_weights: Dict[str, Tensor] = dict(
            self.conv_model.named_parameters())

        # Generate input data and compute target output
        self.x = 1e-2 * torch.randn([
            args.batchsize,
            args.num_channels,
            *args.input_size,
        ]).to(self.device)
        self.y = self.conv_model(self.x)

    def conv_regression_loss(self, conv_weights: Dict[str, Tensor]) -> Tensor:
        """
        Compute the regression loss for the conventional convolutional model.

        Args:
            conv_weights (Dict[str, Tensor]): The weights of the convolutional model.

        Returns:
            Tensor: The computed loss.
        """
        # Compute the model output using the given weights
        y_hat = torch.func.functional_call(
            self.conv_model,
            conv_weights,
            self.x,
        )

        # Compute and return the mean squared error loss
        loss = 0.5 * torch.norm(self.y - y_hat)**2
        return loss.detach()

    def xonv_regression_loss(self, args: argparse.Namespace,
                             conv_weights: Dict[str, Tensor]) -> Tensor:
        """
        Compute the regression loss for the Xonv model.

        Args:
            args (argparse.Namespace): The command line arguments.
            conv_weights (Dict[str, Tensor]): The weights of the convolutional
                model.

        Returns:
            Tensor: The computed loss.
        """
        # Initialize the Xonv model
        xonv_model: Xonv2dRegressionModel = Xonv2dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.input_size,
            args.num_layers,
        ).to(self.device)

        # Set up the optimizer for the Xonv model
        xonv_optimizer: torch.optim.Adam = torch.optim.Adam(
            xonv_model.parameters(),
            lr=args.lr,
        )

        # Training loop for the Xonv model
        for _ in range(args.max_itrs):
            xonv_optimizer.zero_grad()

            # Compute the penalty term
            penalty = sum(args.gamma * torch.norm(xparam - param)**2
                          for xparam, param in zip(xonv_model.parameters(),
                                                   conv_weights.values()))

            # Compute the total loss (MSE + penalty)
            loss = 0.5 * torch.norm(self.y - xonv_model(self.x))**2 + penalty

            # Compute gradients and update parameters
            xgrads: List[Tensor] = torch.autograd.grad(
                loss,
                xonv_model.parameters(),
            )
            for xparam, xgrad in zip(xonv_model.parameters(), xgrads):
                xparam.grad = xgrad

            xonv_optimizer.step()

        return loss.detach()

    def compute_conv_loss_landscape(self, args: argparse.Namespace) -> Tensor:
        """
        Compute the loss landscape for the conventional convolutional model.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Tensor: The computed loss landscape.
        """
        # Generate two random directions in the parameter space
        norm_rand_dirs: Tuple[Dict[str, Tensor], Dict[str, Tensor]] = (
            filter_normalization(self.true_conv_weights),
            filter_normalization(self.true_conv_weights),
        )

        # Initialize the loss landscape tensor
        loss_landscape = torch.zeros(
            [args.vis_res, args.vis_res],
            device=self.device,
        )

        # Create mesh grid for the loss landscape
        param_grid_alpha = torch.linspace(
            *args.vis_range,
            args.vis_res,
        )
        param_grid_beta = torch.linspace(
            *args.vis_range,
            args.vis_res,
        )

        # Compute the loss landscape
        for i, alpha in enumerate(tqdm(param_grid_alpha)):
            for j, beta in enumerate(param_grid_beta):
                updated_params_dict: Dict[str,
                                          Tensor] = update_parameters_dict(
                                              self.true_conv_weights,
                                              norm_rand_dirs,
                                              alpha,
                                              beta,
                                          )
                loss_landscape[i, j] = self.conv_regression_loss(
                    updated_params_dict)

        return loss_landscape

    def compute_xonv_loss_landscape(self, args: argparse.Namespace) -> Tensor:
        """
        Compute the loss landscape for the Xonv model.

        Args:
            args (argparse.Namespace): The command line arguments.

        Returns:
            Tensor: The computed loss landscape.
        """
        # Generate two random directions in the parameter space
        norm_rand_dirs: Tuple[Dict[str, Tensor], Dict[str, Tensor]] = (
            filter_normalization(self.true_conv_weights),
            filter_normalization(self.true_conv_weights),
        )

        # Initialize the loss landscape tensor
        loss_landscape = torch.zeros(
            [args.vis_res, args.vis_res],
            device=self.device,
        )

        # Create mesh grid for the loss landscape
        param_grid_alpha = torch.linspace(
            *args.vis_range,
            args.vis_res,
        )
        param_grid_beta = torch.linspace(
            *args.vis_range,
            args.vis_res,
        )

        # Compute the loss landscape
        for i, alpha in enumerate(tqdm(param_grid_alpha)):
            for j, beta in enumerate(param_grid_beta):
                updated_params_dict: Dict[str,
                                          Tensor] = update_parameters_dict(
                                              self.true_conv_weights,
                                              norm_rand_dirs,
                                              alpha,
                                              beta,
                                          )
                loss_landscape[i, j] = self.xonv_regression_loss(
                    args,
                    updated_params_dict,
                )

        return loss_landscape


if __name__ == '__main__':
    # Read input arguments from a JSON file and process them
    args: argparse.Namespace = query_arguments(CONFIG_FILE)[0]
    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize the RegressionLossLandscape object
    reg_loss_landscape: RegressionLossLandscape = RegressionLossLandscape(args)

    # Compute the loss landscapes for both models
    conv_loss_landscape = reg_loss_landscape.compute_conv_loss_landscape(args)
    xonv_loss_landscape = reg_loss_landscape.compute_xonv_loss_landscape(args)

    # Save the results
    torch.save(
        {
            'true_conv_weights': reg_loss_landscape.true_conv_weights,
            'x': reg_loss_landscape.x,
            'y': reg_loss_landscape.y,
            'conv_loss_landscape': conv_loss_landscape,
            'xonv_loss_landscape': xonv_loss_landscape,
            'args': args,
        },
        os.path.join(
            checkpointsdir(args.experiment),
            'loss_landscapes.pth',
        ),
    )

    # Upload results to Dropbox if specified
    if args.upload_results:
        upload_to_dropbox(args)
