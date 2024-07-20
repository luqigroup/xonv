# pylint: disable=E1102
# pylint: disable=invalid-name
"""A script that uses extended convolutional layers for regression.
"""
from typing import Dict
import os

import argparse
from tqdm import tqdm
import torch

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
    """An example for performing regression with the Xonv2D layer.

    Attributes:
        device (torch.device): The device (cpu/cuda) used for computation.
        data_dist: The distribution of the data.
        latent_dist: The latent distribution (standard normal).
        train_obj: Placeholder for training objective values.
        val_obj: Placeholder for validation objective values.
    """

    def __init__(self, args: argparse.ArgumentParser) -> None:
        """
        Initialize the GaussianExample object.

        Args:
            args (argparse.ArgumentParser): The command line arguments.
        """
        # Setting default device (cpu/cuda) depending on CUDA availability and
        # input arguments.
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device('cuda:' + str(args.gpu_id))
        else:
            self.device = torch.device('cpu')

        self.conv_model = Conv2dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.num_layers,
        ).to(self.device)

        self.true_conv_weights = dict(self.conv_model.named_parameters())

        self.x = 1e-2 * torch.randn([
            args.batchsize,
            args.num_channels,
            *args.input_size,
        ]).to(self.device)
        self.y = self.conv_model(self.x)

    def conv_regression_loss(
        self,
        conv_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        y_hat = torch.func.functional_call(
            self.conv_model,
            conv_weights,
            self.x,
        )

        loss = 5e-1 * torch.norm(self.y - y_hat)**2

        return loss.detach()

    def xonv_regression_loss(
        self,
        args,
        conv_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        xonv_model = Xonv2dRegressionModel(
            args.num_channels,
            args.kernel_size,
            args.input_size,
            args.num_layers,
        ).to(self.device)

        xonv_optimizer = torch.optim.Adam(
            xonv_model.parameters(),
            lr=args.lr,
        )

        for _ in range(args.max_itrs):
            xonv_optimizer.zero_grad()

            penalty = 0.0
            for xparam, param in zip(
                    xonv_model.parameters(),
                    conv_weights.values(),
            ):
                penalty += args.gamma * torch.norm(xparam - param)**2
            loss = 5e-1 * torch.norm(self.y - xonv_model(self.x))**2 + penalty

            xgrads = torch.autograd.grad(
                loss,
                xonv_model.parameters(),
            )
            for xparam, xgrad in zip(xonv_model.parameters(), xgrads):
                xparam.grad = xgrad

            xonv_optimizer.step()

        return loss.detach()

    def compute_conv_loss_landscape(self, args) -> torch.Tensor:

        # Get two random directions in the parameter space.
        norm_rand_dirs = (
            filter_normalization(self.true_conv_weights),
            filter_normalization(self.true_conv_weights),
        )

        # Initialize the loss landscape tensor.
        loss_landscape = torch.zeros(
            [args.vis_res, args.vis_res],
            device=self.device,
        )

        # Create mesh grid of loss landscape.
        param_grid_alpha = torch.linspace(*args.vis_range, args.vis_res)
        param_grid_beta = torch.linspace(*args.vis_range, args.vis_res)

        for i, alpha in enumerate(tqdm(param_grid_alpha)):
            for j, beta in enumerate(param_grid_beta):

                updated_params_dict = update_parameters_dict(
                    self.true_conv_weights,
                    norm_rand_dirs,
                    alpha,
                    beta,
                )

                loss_landscape[i, j] = self.conv_regression_loss(
                    updated_params_dict)

        return loss_landscape

    def compute_xonv_loss_landscape(self, args) -> torch.Tensor:

        # Get two random directions in the parameter space.
        norm_rand_dirs = (
            filter_normalization(self.true_conv_weights),
            filter_normalization(self.true_conv_weights),
        )

        # Initialize the loss landscape tensor.
        loss_landscape = torch.zeros(
            [args.vis_res, args.vis_res],
            device=self.device,
        )

        # Create mesh grid of loss landscape.
        param_grid_alpha = torch.linspace(*args.vis_range, args.vis_res)
        param_grid_beta = torch.linspace(*args.vis_range, args.vis_res)

        for i, alpha in enumerate(tqdm(param_grid_alpha)):
            for j, beta in enumerate(param_grid_beta):

                updated_params_dict = update_parameters_dict(
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


if '__main__' == __name__:

    # Read input arguments from a json file and make an experiment name.
    args = query_arguments(CONFIG_FILE)[0]
    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args)

    # Random seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize the RegressionLossLandscape object.
    reg_loss_landscape = RegressionLossLandscape(args)

    # Compute the loss landscape.
    conv_loss_landscape = reg_loss_landscape.compute_conv_loss_landscape(args)
    xonv_loss_landscape = reg_loss_landscape.compute_xonv_loss_landscape(args)

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

    # Upload results to Dropbox.
    if args.upload_results:
        upload_to_dropbox(args)
