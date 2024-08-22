from typing import Callable, Union
from scipy import linalg
from tqdm.notebook import tqdm
import torch
import numpy as np
import warnings


def combine_paths_matrices_torch(
    matrices: torch.tensor, alpha: Union[float, torch.tensor] = 0
) -> torch.tensor:
    # NOTE: this function is not optimizable for alpha
    # however this would be a starting point for that
    # so we include it in the solver loop for now
    """Create a design matrix for the path model by combining the design matrices of
    each path lengths.

    Parameters
    ----------
    matrices : np.ndarray
        individual design matrices for each path length
    alpha : Union[float, np.ndarray], optional
        hyperparameter to include the influence of sub-optimal paths (could be one
        single value or a value for length greater than the shortest path), by default 0

    Returns
    -------
    np.ndarray
        design matrix of the path model.

    Raises
    ------
    ValueError
        the `alpha` parameter should be a scalar or have the same length as the number
        of matrices.
    """

    design = torch.zeros_like(matrices[0])
    alpha_id_vector = torch.zeros(design.shape[-1], dtype=int)
    alpha_norm = torch.zeros_like(alpha_id_vector)

    # Compatiblity for the type of alpha
    if isinstance(alpha, (float, int)):
        alpha = torch.tensor([alpha] * len(matrices))
    if isinstance(alpha, (list, tuple)):
        alpha = torch.tensor(alpha)

    if len(alpha) != len(matrices):
        raise ValueError(
            "The alpha parameter must be a scalar or have the same length as the number"
            f" of matrices ({len(alpha)} alphas for {len(matrices)} matrices)."
        )

    for m in matrices:
        # Find rows that have already been filled
        has_shortest_paths = torch.any(design, axis=1)

        # Update the alpha vector for paths that have already been filled
        alpha_vector = has_shortest_paths * alpha[alpha_id_vector] + ~has_shortest_paths
        alpha_id_vector += has_shortest_paths * torch.any(m, axis=1)

        # Update the design matrix
        design += torch.diag(alpha_vector).type(torch.float64) @ m

    # Normalize the design matrix by 1 plus the sum of existing alpha weights
    alpha_norm = torch.tensor([1 + alpha[:i].sum() for i in alpha_id_vector])
    design = torch.diag(1 / alpha_norm).type(torch.float64) @ design
    return design


def forward(a_design: torch.tensor, effective_delay: torch.tensor) -> torch.tensor:
    """
    Computes the estimated delay based on the design matrix and the effective delay.

    Parameters
    ----------
    a_design : torch.tensor
        The design matrix.
    effective_delay : torch.tensor
        The effective delay.

    Returns
    -------
    torch.tensor
        The estimated delay.
    """
    estimated_delay = a_design @ effective_delay
    return estimated_delay


def gradient_descent_solver(
    x: torch.tensor,
    y_ground: torch.tensor,
    a_design: torch.tensor,
    delta: float = 0,
    early_stop: float = 1e-5,
    step_size: float = 1e-3,
    l2_penalty: float = 0.1,
    n_iter: int = 1000,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Performs gradient descent optimization to minimize the mean squared error (MSE)
    between the predicted output `y_pred` and the ground truth `y_ground`.

    Parameters:
        x (torch.tensor): The input tensor to optimize.
        y_ground (torch.tensor): The ground truth output tensor.
        a_design (torch.tensor): The design matrix.
        delta (float, optional): Value assigned for the synaptic delay parameter.
        early_stop (float, optional): The early stopping threshold. Defaults to 1e-5.
        step_size (float, optional): The step size for gradient descent. Defaults to 1e-3.
        n_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print progress during optimization. Defaults to False.

    Returns:
        tuple[np.ndarray, float]: The optimized input tensor `x_opt` and the final MSE loss.
    """

    def mse(y_est, y_ground):
        return torch.linalg.norm(y_est - y_ground)

    loss_logs = [-1]
    for i in tqdm(range(n_iter)):
        y_pred = forward(a_design, x + delta * (x > 0))

        data_fit = mse(y_pred, y_ground)
        pseudo_fit = torch.linalg.norm(x, ord=2)
        positivity = torch.abs(torch.sum(x * (x < 0).type(torch.float)))

        loss = data_fit + pseudo_fit * l2_penalty + positivity
        loss.backward()

        x.data = x.data - step_size * x.grad.data
        x.grad.data.zero_()

        if verbose:
            if (i % (n_iter // 10)) == 0:
                print(f"###### ITER {i} #######")
                print(
                    f"""datafit loss: {data_fit.item()}
L2 norm: {pseudo_fit.item()}
positivity loss: {positivity.item()}"""
                )
                print()
        loss_logs.append(loss.item())

        # NOTE: arbitrary value
        if torch.diff(torch.tensor(loss_logs[-5:])).abs().mean() < early_stop:
            print(f"Stopped at iteration #{i}")
            break

    x_opt = x.detach().numpy()
    return x_opt, data_fit.item()


def naive_gradient_descent(
    x: torch.tensor,
    y_ground: torch.tensor,
    alpha: torch.tensor,
    multi_design: torch.tensor,
    early_stop: float = 1e-5,
    step_size: float = 1e-3,
    n_iter: int = 1000,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Performs naive gradient descent optimization on a given input tensor `x` to
    minimize the mean squared error (MSE) between the predicted output `y_pred` and the
    ground truth `y_ground`.

    Parameters:
        x (torch.tensor): The input tensor to be optimized.
        y_ground (torch.tensor): The ground truth output tensor.
        alpha (torch.tensor): The alpha parameter used to combine the path matrices.
        multi_design (torch.tensor): The multi-design matrix.
        early_stop (float, optional): The early stopping criterion based on the change
        in the last 5 loss values. Defaults to 1e-5.
        step_size (float, optional): The step size for the gradient descent update.
        Defaults to 1e-3.
        n_iter (int, optional): The maximum number of iterations for the gradient
        descent. Defaults to 1000.
        verbose (bool, optional): Whether to print the loss at every 10% of the
        iterations. Defaults to False.

    Returns:
        tuple[np.ndarray, float]: The optimized input tensor `x_opt` and the final loss
        value.
    """

    warnings.warn(
        "Not using the newest path designing function", DeprecationWarning, stacklevel=2
    )

    def mse(y_est, y_ground):
        return torch.linalg.norm(y_est - y_ground)

    loss_logs = [-1]
    for i in range(n_iter):
        a_design = combine_paths_matrices_torch(multi_design, alpha=alpha)
        y_pred = forward(a_design, x)

        data_fit = mse(y_pred, y_ground)
        pseudo_fit = torch.linalg.norm(x, ord=2)
        positivity = torch.sum(x * (x < 0).type(torch.float))

        loss = data_fit + pseudo_fit + positivity
        loss.backward()

        x.data = x.data - step_size * x.grad.data
        x.grad.data.zero_()

        if verbose:
            if (i % (n_iter // 10)) == 0:
                print(f"###### ITER {i} #######")
                print(
                    f"""datafit loss: {data_fit.item()}
L2 norm: {pseudo_fit.item()}
positivity loss: {positivity.item()}"""
                )
                print()
        loss_logs.append(loss.item())

        # NOTE: arbitrary value
        if torch.diff(torch.tensor(loss_logs[-5:])).abs().mean() < early_stop:
            print(f"Stopped at iteration #{i}")
            break

    x_opt = x.detach().numpy()
    return x_opt, data_fit.item()


def pseudo_inverse(y: np.ndarray, a_design: np.ndarray, rcond: float = 1e-15):
    """
    Computes the pseudo-inverse of the input matrix `a_design` and applies it to the
    input vector `y` to obtain the optimal solution `x_opt`.

    Parameters:
        y (numpy.ndarray): The input vector.
        a_design (numpy.ndarray): The design matrix.
        rcond (float, optional): The relative condition number threshold. Defaults to
        1e-15.

    Returns:
        numpy.ndarray: The optimal solution `x_opt`.
    """

    Ainv = np.linalg.pinv(a_design, rcond=rcond)
    x_opt = Ainv @ y
    return x_opt
