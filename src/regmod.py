import numpy as np
from scipy import linalg
import networkx as nx


def build_design_binary(adjacency: np.ndarray):
    """Create a design matrix from the structural connectivity matrix. Each row
    represents a connection (e.g. delay) between two nodes and each column is an edge
    of the structural connectome. There is a 1 a structural connectome edge is going
    connected to the "sending" node.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the structural connectivity.

    Returns
    -------
    np.ndarray
        binary design matrix of the degree model.
    """
    n = adjacency.shape[0]

    blocks = [np.vstack([np.delete(adjacency[i], i)] * (n - 1)) for i in range(n)]

    a_binary = linalg.block_diag(*blocks)
    return a_binary


def build_design_degree_fast(adjacency: np.ndarray, target_deg: bool = False):
    """Create a design matrix similar to the binary model but with weights. Weights are
    dependent on the degree of the "sending" node, the "receiving" node and nodes
    connected to the "sending" node (see `target_deg` parameter for details).

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the structural connectivity.
    target_deg : bool, optional
        condition to use the degree of the "receiving" node instead of the degrees of
        nodes connected to the "sending" node as the numerator of the weight (see
        formula below), by default False.

        True: w_ij = deg(i) / (deg(i) + deg(j))
        False: w_ij = deg(j) / (deg(i) + deg(j))

    Returns
    -------
    np.ndarray
        weighted design matrix of the degree model.
    """
    n = adjacency.shape[0]

    a_binary = build_design_binary(adjacency)

    degrees = adjacency.sum(axis=1)

    nominator = np.ones((len(degrees), 1)) * degrees

    if target_deg:
        nominator = nominator.T

    denum = degrees + degrees.reshape((-1, 1))
    weights = np.divide(nominator, denum)

    blocks = [np.delete(np.delete(weights, i, axis=0), i, axis=1) for i in range(n)]
    a_weights = linalg.block_diag(*blocks)
    return a_binary * a_weights


def build_design_paths(adjacency: np.ndarray, max_path_length: int = 2):
    """Create a design matrix for the path model. It has a value of k if the edge on
    column j is in a path of length k between the two nodes connected by the edge on
    row i. The maximum path length is defined by the `max_path_length` parameter.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix of the structural connectivity.
    max_path_length : int, optional
        maximum path length allowed in the search, by default 2

    Returns
    -------
    np.ndarray
        binary design matrix of the path model.
    """

    s_graph = nx.Graph(adjacency)
    n = len(adjacency)
    n_edges = n * (n - 1)

    edges_id = [(i, j) for i in range(n) for j in range(n) if i != j]

    edge_to_edge_id_dict = {ed: i for i, ed in enumerate(edges_id)}

    edges_id = np.array(edges_id)

    a_design1 = np.diag(adjacency[*edges_id.T]).astype(int)

    for path_length in np.arange(2, max_path_length + 1):
        a_design = np.zeros((n_edges, n_edges))
        for design_i, (node_in, node_out) in enumerate(edges_id):
            all_paths = nx.all_simple_edge_paths(
                s_graph, node_in, node_out, cutoff=path_length
            )
            for path in all_paths:
                if len(path) == path_length:
                    ids = [edge_to_edge_id_dict[coords] for coords in path]
                    a_design[design_i, ids] = path_length
        a_design1[a_design1 == 0] = a_design[a_design1 == 0]

    return a_design1
