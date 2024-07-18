from typing import Dict, Tuple

import torch


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
        rand_direction_norms = rand_direction.norm(
            dim=list(range(1, rand_direction.ndim)),
            keepdim=True,
        )
        param_data_norms = param_data.norm(
            dim=list(range(1, param_data.ndim)),
            keepdim=True,
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
