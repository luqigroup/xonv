from typing import Tuple
import os

import torch
import numpy as np
import argparse
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc

from xonv.utils import plotsdir

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
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

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(
        loss_landscape,
        resample=True,
        interpolation="nearest",
        filterrad=1,
        aspect=1,
        cmap=cc.cm["linear_protanopic_deuteranopic_kbw_5_98_c40"],
        extent=[*args.vis_range, *args.vis_range],
        norm=matplotlib.colors.LogNorm(),
    )
    plt.colorbar(fraction=0.047, pad=0.01, format=sfmt)
    plt.scatter(
        param_grid[col],
        param_grid[row],
        c="red",
        marker="*",
        s=7.5,
        label="Minimizer",
    )
    plt.title("Loss landscape: " + fig_name_extension)
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.savefig(
        os.path.join(
            plotsdir(args.experiment),
            'loss_landscape_' + fig_name_extension + '.png',
        ),
        format="png",
        bbox_inches="tight",
        dpi=600,
        pad_inches=.02,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    contour = ax.contour(
        np.linspace(args.vis_range[0], args.vis_range[1],
                    loss_landscape.shape[1]),
        np.linspace(args.vis_range[0], args.vis_range[1],
                    loss_landscape.shape[0]),
        loss_landscape.numpy()[::-1, :],
        levels=50,
        extent=[*args.vis_range, *args.vis_range],
        cmap=cc.cm["linear_protanopic_deuteranopic_kbw_5_98_c40"],
    )
    plt.scatter(
        param_grid[col],
        param_grid[row],
        c="red",
        marker="*",
        s=7.5,
        label="Minimizer",
    )
    plt.grid(False)
    fig.colorbar(contour, ax=ax, fraction=0.05, pad=0.01, format=sfmt)
    plt.legend(loc="upper right")
    ax.set_title("Loss landscape: " + fig_name_extension)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    plt.savefig(
        os.path.join(
            plotsdir(args.experiment),
            'loss_landscape_contourf_' + fig_name_extension + '.png',
        ),
        format="png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=.02,
    )
    plt.close(fig)
