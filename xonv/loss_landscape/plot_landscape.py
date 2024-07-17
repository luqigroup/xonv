from typing import Tuple
import os

import torch
import numpy as np
import argparse
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from xonv.utils import plotsdir

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


def plot_landscape(
    args: argparse.Namespace,
    loss_landscape: torch.Tensor,
    alpha_range: Tuple[float, float],
    beta_range: Tuple[float, float],
):

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(
        loss_landscape,
        resample=True,
        interpolation="nearest",
        filterrad=1,
        aspect=1,
        cmap="gist_earth",
        extent=[*alpha_range, *beta_range],
        norm=matplotlib.colors.LogNorm(),
    )
    plt.title("Loss landscape")
    plt.colorbar(fraction=0.047, pad=0.01, format=sfmt)
    plt.grid(False)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.savefig(
        os.path.join(plotsdir(args.experiment), "loss_landscape.png"),
        format="png",
        bbox_inches="tight",
        dpi=600,
        pad_inches=.02,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    contour = ax.contour(
        np.linspace(alpha_range[0], alpha_range[1], loss_landscape.shape[1]),
        np.linspace(beta_range[0], beta_range[1], loss_landscape.shape[0]),
        loss_landscape,
        levels=50,  # Adjust number of contour levels as needed
        cmap="gist_earth",
    )
    plt.grid(False)
    fig.colorbar(contour, ax=ax, fraction=0.05, pad=0.01, format=sfmt)
    ax.set_title("Loss landscape")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    plt.savefig(
        os.path.join(
            plotsdir(args.experiment),
            "loss_landscape_contourf.png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=600,
        pad_inches=.02,
    )
    plt.close(fig)
