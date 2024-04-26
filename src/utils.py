import pickle
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


# saving and loading made-easy
def save(pickle_file, array):
    """
    Pickle array (in general any formattable object)
    args::
        pickle_file: str, path to save the file
        array: object to be pickled
    ret::
        None
    """
    with open(pickle_file, "wb") as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(pickle_file):
    """
    Loading pickled array
    args::
        pickle_file: str, path to load the file
    ret::
        b: object loaded from pickle file
    """
    with open(pickle_file, "rb") as handle:
        b = pickle.load(handle)
    return b


def add_cbar(fig, ax, **kwargs):
    """Add a colorbar to an existing figure/axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to get the colorbar from
    ax : matplotlib.axes.Axes
        axes to add the colorbar to

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        tuple of figure and axes with the colorbar added
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical", **kwargs)
    return fig, ax


def annotate_heatmap(
    fig: plt.Figure, ax: plt.Axes, matrix: np.ndarray, adapt_color: float = 0
) -> tuple[plt.Figure, plt.Axes]:
    """Annotate heatmap with values from a matrix.

    Parameters
    ----------
    fig : plt.Figure
        figure to annotate
    ax : plt.Axes
        axis to annotate
    matrix : np.ndarray
        data matrix
    adapt_color : float, optional
        threshold value to alternate from dark (above) to light (below) text, by
        default 0

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        figure and axes with the heatmap annotated
    """

    for row_i, row in enumerate(matrix):
        for col_j, val in enumerate(row):

            color = "w" if val < adapt_color else "k"
            if val > 0:
                ax.text(
                    row_i, col_j, f"{val:1.2f}", ha="center", va="center", color=color
                )
    return fig, ax
