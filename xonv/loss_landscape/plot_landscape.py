import argparse
import os

import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from xonv.utils import plotsdir

sns.set_style("whitegrid")
font = {"family": "serif", "style": "normal", "size": 10}
matplotlib.rc("font", **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


def plot_loss_landscape(
    args: argparse.Namespace,
    loss_landscape: torch.Tensor,
    fig_name_extension: str = "",
):
    argmin_index = torch.argmin(loss_landscape)
    row, col = divmod(argmin_index.item(), loss_landscape.size(1))
    param_grid = torch.linspace(
        *args.vis_range,
        args.vis_res,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(
        np.linspace(
            args.vis_range[0], args.vis_range[1], loss_landscape.shape[1]
        ),
        np.linspace(
            args.vis_range[0], args.vis_range[1], loss_landscape.shape[0]
        ),
        loss_landscape.numpy()[::-1, :],
        levels=50,
        extent=[*args.vis_range, *args.vis_range],
        colors="black",
        alpha=0.5,
        linewidths=0.7,
    )
    image = ax.imshow(
        loss_landscape,
        resample=True,
        interpolation="nearest",
        filterrad=1,
        aspect=1,
        cmap=cc.cm["linear_protanopic_deuteranopic_kbw_5_98_c40"],
        extent=[*args.vis_range, *args.vis_range],
        norm=matplotlib.colors.LogNorm(),
    )
    # plt.scatter(
    #     param_grid[col],
    #     param_grid[row],
    #     c="red",
    #     marker="*",
    #     s=7.5,
    #     label="Minimizer",
    # )
    plt.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.0473, pad=0.01, format=sfmt)
    # Optionally, change the font size of the colorbar label
    # Set the font size for the colorbar labels
    cbar.ax.tick_params(labelsize=10)
    # plt.legend(loc="upper right", fontsize=12)
    ax.set_title("Loss landscape: " + fig_name_extension, fontsize=15)
    # ax.set_xlabel(r"$\alpha$")
    # ax.set_ylabel(r"$\beta$")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.savefig(
        os.path.join(
            plotsdir(args.experiment),
            "loss_landscape_contourf_" + fig_name_extension + ".png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.02,
    )
    plt.close(fig)
