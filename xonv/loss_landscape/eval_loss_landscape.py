from typing import Tuple, Callable
from tqdm import tqdm
import torch

from .loss_landscape import filter_normalization, update_parameters_dict


@torch.no_grad()
def evaluate_loss_landscape(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    num_samples: int,
    alpha_range: Tuple[float, float],
    beta_range: Tuple[float, float],
) -> torch.Tensor:

    parameters_dict = dict(model.named_parameters())

    # Get two random directions in the parameter space.
    norm_rand_dirs = (
        filter_normalization(parameters_dict),
        filter_normalization(parameters_dict),
    )

    # Initialize the loss landscape tensor.
    loss_landscape = torch.zeros(
        [num_samples, num_samples],
        device=data.device,
    )

    # Create mesh grid of loss landscape.
    param_grid_alpha = torch.linspace(*alpha_range, num_samples)
    param_grid_beta = torch.linspace(*beta_range, num_samples)

    for i, alpha in enumerate(tqdm(param_grid_alpha)):
        for j, beta in enumerate(param_grid_beta):

            updated_params_dict = update_parameters_dict(
                parameters_dict,
                norm_rand_dirs,
                alpha,
                beta,
            )

            pred = torch.func.functional_call(
                model,
                updated_params_dict,
                data,
            )

            loss_landscape[i, j] = loss_fn(data, pred)

    return loss_landscape
