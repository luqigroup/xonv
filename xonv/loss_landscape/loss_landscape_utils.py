from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def filter_normalization(
        parameters_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    normalized_random_direction = {
        key: torch.randn_like(value)
        for key, value in parameters_dict.items()
    }

    for (param_name, param_data), rand_direction in zip(
            parameters_dict.items(),
            normalized_random_direction.values(),
    ):

        # Compute norms across all dimensions >= 1
        tmp = rand_direction.clone()
        tmp = tmp.view(tmp.size(0), -1) if len(tmp.size()) > 1 else tmp

        rand_direction_norms = tmp.norm(
            dim=-1,
            keepdim=False if len(tmp.size()) > 1 else True,
        )
        rand_direction_norms = rand_direction_norms.view(
            rand_direction_norms.shape[0],
            *[1 for _ in rand_direction.shape[1:]],
        )

        param_data = param_data.view(param_data.size(0), -1) if len(
            param_data.size()) > 1 else param_data
        param_data_norms = param_data.norm(
            dim=-1,
            keepdim=False if len(tmp.size()) > 1 else True,
        )
        param_data_norms = param_data_norms.view(
            param_data_norms.shape[0],
            *[1 for _ in rand_direction.shape[1:]],
        )

        # Normalize the random direction tensor
        normalized_random_direction[
            param_name] = rand_direction * param_data_norms / (
                rand_direction_norms + 1e-8)

    return normalized_random_direction


@torch.no_grad()
def update_parameters_dict(
    parameters_dict: Dict[str, torch.Tensor],
    norm_rand_dirs: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    alpha: float,
    beta: float,
) -> Dict[str, torch.Tensor]:

    updated_params_dict = {
        key: value.detach().clone()
        for key, value in parameters_dict.items()
    }

    for param_name in updated_params_dict.keys():
        rand_offset = (alpha * norm_rand_dirs[0][param_name].clone() +
                       beta * norm_rand_dirs[1][param_name].clone())
        updated_params_dict[param_name] += rand_offset

    return updated_params_dict


@torch.no_grad()
def get_test_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> float:
    """
    Compute the average test loss for a model over a given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader providing (input, target) batches.

    Returns:
        float: The average test loss across the dataset.
    """
    model.eval()

    total_loss: float = 0.0
    total_samples: int = 0
    device: torch.device = next(model.parameters()).device

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        # Forward pass
        y_hat: Tensor = model(xb)
        loss: Tensor = 0.5 * torch.norm(yb - y_hat) ** 2

        # Accumulate weighted loss
        batch_size: int = xb.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss: float = total_loss / total_samples

    model.train()
    return avg_loss
