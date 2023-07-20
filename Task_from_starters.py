import numpy as np
# Numexpr is used for parallel operations
import numexpr as ne
from numba import njit


#---------------------------TASK 1--------------------------


def corr(first, second):
    """
    Calculates Pearson correlation between two vectors.

    Parameters
    ----------

    first: np.array 1-d
        first array
    second: np.array 1-d
        second array

    Returns
    -------
    int
        Pearson correlation between first and second array
    """

    return ((first*second).mean() - first.mean() * second.mean()) / (first.std()*second.std())


@njit
def corr_numba(first, second):
    """
    Numba analog for 'corr'
    """
    
    return ((first*second).mean() - first.mean() * second.mean()) / (first.std()*second.std())



def first_task_baseline(A, B):
    """
    Calculates mean correlation between A[i,j,:] trace and all traces from B array

    Parameters
    ----------
    A: np.ndarray
        Array of shape (X, Y, Z), where Z is the length of trace 
    B: np.ndarray
        Array of shape (N, Z), where Z is the length of trace, and N is number of traces

    Returns:
        array of shape (X, Y) with mean correlations
    """

    X,Y,Z = A.shape
    N, Z = B.shape
    corr_matrix = np.empty((X, Y))

    for ij in np.ndindex(A.shape[:2]):
        A_trace = A[ij]
        sum_ = 0
        for B_trace in B:
            sum_ += corr(A_trace, B_trace)
        corr_matrix[ij] = sum_/B.shape[0]
    return corr_matrix


@njit
def first_task_baseline_numba(A, B):
    """
    Numba analog for 'first_task_baseline'.
    """

    X,Y,Z = A.shape
    N, Z = B.shape
    corr_matrix = np.empty((X, Y))

    for ij in np.ndindex(A.shape[:2]):
        A_trace = A[ij]
        sum_ = 0
        for B_trace in B:
            sum_ += corr_numba(A_trace, B_trace)
        corr_matrix[ij] = sum_/B.shape[0]
    return corr_matrix


def first_task(A, B):
    """
    Calculates mean correlation between A[i,j,:] trace and all traces from B array

    Parameters
    ----------
    A: np.ndarray
        Array of shape (X, Y, Z), where Z is the length of trace 
    B: np.ndarray
        Array of shape (N, Z), where Z is the length of trace, and N is number of traces

    Returns:
        array of shape (X, Y) with mean correlations
    """

    X, Y, Z = A.shape
    N, Z = B.shape

    A_tmp = A.reshape(-1, Z)

    mean_A = A_tmp.mean(axis=1, keepdims=True)
    mean_B = B.mean(axis=1, keepdims=True)

    A_center = ne.evaluate('A_tmp - mean_A')
    B_center = ne.evaluate('B - mean_B')

    A_center_squared = ne.evaluate('A_center**2')
    B_center_squared = ne.evaluate('B_center**2')

    A_var = np.sum(A_center_squared, axis=1, keepdims=True)
    B_var = np.sum(B_center_squared, axis=1, keepdims=True)

    dot_var = A_var @ B_var.T
    std_dot = ne.evaluate('sqrt(dot_var)')
    mat_dot = A_center @ B_center.T

    result = ne.evaluate('mat_dot/std_dot').sum(axis=1)
    final_result = ne.evaluate('result/N').reshape((X, Y))

    return final_result


#---------------------------TASK 2--------------------------


def _create_blocks(A, K):
    """
    Creates K x K x Z blocks from original array, and reshapes it into K^2 x Z blocks

    Returns:
        array of shape: (n_blocks, K^2, Z)
    """

    itemsize = A.itemsize
    X, Y, Z = A.shape

    A = np.pad(A, pad_width=((K//2, K//2), (K//2, K//2), (0,0)), mode='edge')

    BATCH_SIZE = (K, K, Z)

    blocks = np.lib.stride_tricks.as_strided(
        A,
        shape = (X, Y, K, K, Z),
        strides = (
            itemsize*Z*(Y + (K - 1)),
            itemsize*Z,
            itemsize*Z*(Y + (K - 1)),
            itemsize*Z,
            itemsize
        )
    )
    blocks = blocks.reshape(X*Y, K, K, Z)
    # flattening kxkxz traces into k^2xz so that central trace will be in center of resulting array
    blocks = blocks.reshape(X*Y, K*K, Z)
    return blocks


def second_task_baseline(A, K): # works really slow
    """
    Calculates mean correlation between A[i,j,:] trace and traces that surround if within KxK window

    Returns:
        array of shape (X,Y), where each value is a mean correlation between all traces in KxK window
    """

    result = np.empty((A.shape[0], A.shape[1]))
    A = np.pad(A, pad_width=((K//2, K//2), (K//2, K//2), (0,0)), mode='edge')

    for i in range(K//2, A.shape[0] - K//2):
        for j in range(K//2, A.shape[1] - K//2):
            # take all traces in KxK window except for the central trace
            neighbour_traces = [A[i + k_i, j + k_j] for k_j in range((-K)//2 + 1, K//2 + 1) for k_i in range((-K)//2 + 1, K//2 + 1) if (k_i != 0) or (k_j != 0)]

            correlations = []
            for trace in neighbour_traces:
                correlations.append(corr(trace, A[i,j]))
            result[i-(K//2),j-(K//2)] = np.sum(correlations) / (K**2 - 1)

    return result


def second_task_v1(A, K):
    """
    Calculates mean correlation between A[i,j,:] trace and traces that surround if within KxK window

    Returns:
        array of shape (X,Y), where each value is a mean correlation between all traces in KxK window
    """
    X, Y, Z = A.shape

    blocks = _create_blocks(A, K)
    central_tr = blocks[:, blocks.shape[1]//2, :].reshape(X*Y, 1, Z)

    mul_blocks_central = ne.evaluate('blocks * central_tr').mean(axis=2, keepdims=True)

    mean_central_tr = central_tr.mean(axis=2, keepdims=True)
    mean_blocks = blocks.mean(axis=2, keepdims=True)

    numenator = ne.evaluate('mul_blocks_central - mean_central_tr * mean_blocks').squeeze()

    center_tr_squared = ne.evaluate('(central_tr - mean_central_tr)**2')
    blocks_squared = ne.evaluate('(blocks - mean_blocks)**2')
    sum_center_tr = center_tr_squared.sum(axis=2)
    sum_blocks = blocks_squared.sum(axis=2)

    std_central_tr = ne.evaluate('sqrt(sum_center_tr/Z)')
    std_blocks = ne.evaluate('sqrt(sum_blocks/Z)')

    denominator = ne.evaluate('std_central_tr * std_blocks')

    division = ne.evaluate('numenator / denominator')
    sum_division = division.sum(axis=1)
    # subtracting 1 because we included central trace in calculations
    result = ne.evaluate('(sum_division - 1) / (K**2 - 1)')

    return result.reshape((X, Y))


def second_task_v2(A, K):
    """
    Calculates mean correlation between A[i,j,:] trace and traces that surround if within KxK window

    An optimization of second_task_v1. Simply reduced number of computations.

    Returns:
        array of shape (X,Y), where each value is a mean correlation between all traces in KxK window
    """

    X, Y, Z = A.shape

    blocks = _create_blocks(A, K)
    central_tr = blocks[:, blocks.shape[1]//2, :].reshape(X*Y, 1, Z)

    sum_of_prod = ne.evaluate('blocks * central_tr').sum(axis=2)
    z_block_central = ne.evaluate('sum_of_prod * Z')

    sum_block = blocks.sum(axis=2)
    sum_central = central_tr.sum(axis=2)

    sum_block_sq = ne.evaluate('blocks ** 2').sum(axis=2)
    sum_central_sq = ne.evaluate('central_tr ** 2').sum(axis=2)

    std_block = ne.evaluate('Z * sum_block_sq - sum_block**2')
    std_central = ne.evaluate('Z * sum_central_sq - sum_central**2')

    numenator = ne.evaluate('z_block_central - sum_central * sum_block')
    denominator = ne.evaluate('sqrt(std_block * std_central)')

    division = ne.evaluate('numenator / denominator')
    sum_division = division.sum(axis=1)
    # subtracting 1 because we included central trace in calculations
    result = ne.evaluate('(sum_division - 1) / (K**2 - 1)')

    return result.reshape((X, Y))


def second_task_via_distance(A, K):
    """
    Calculates mean correlation between A[i,j,:] trace and traces that surround if within KxK window

    Follows the formula:

        corr(x,y) = 1 - d(x, y)/(2n),

    where x,y normalized n-dimensional arrays.

    Returns:
        array of shape (X,Y), where each value is a mean correlation between all traces in KxK window
    """

    X, Y, Z = A.shape

    blocks = _create_blocks(A, K)
    central_tr = blocks[:, blocks.shape[1]//2, :].reshape(X*Y, 1, Z)

    mean_blocks = blocks.mean(axis=2, keepdims=True)
    mean_central_tr = central_tr.mean(axis=2, keepdims=True)
    std_blocks = blocks.std(axis=2, keepdims=True)
    std_central_tr = central_tr.std(axis=2, keepdims=True)

    stand_blocks = (blocks - mean_blocks) / std_blocks
    stand_central_tr = (central_tr - mean_central_tr) / std_central_tr

    squared_dist = ne.evaluate('(stand_blocks - stand_central_tr)**2')
    sum_sq = squared_dist.sum(axis=2)
    correlations = ne.evaluate('1 - sum_sq / (2 * Z)')

    result = (correlations.sum(axis=1) - 1) / (K**2 - 1)
    return result.reshape((X, Y))


def second_task_numpy(A, K):
    """
    Calculates mean correlation between A[i,j,:] trace and traces that surround if within KxK window

    Uses only numpy, no numexpr of numba.

    Returns:
        array of shape (X,Y), where each value is a mean correlation between all traces in KxK window
    """

    X, Y, Z = A.shape

    blocks = _create_blocks(A, K)
    central_tr = blocks[:, blocks.shape[1]//2, :].reshape(X*Y, 1, Z)

    result = ((blocks * central_tr).mean(axis=2) - central_tr.mean(axis=2) * blocks.mean(axis=2)) / (central_tr.std(axis=2) * blocks.std(axis=2))

    result = (result.sum(axis=1) - 1) / (K**2 - 1)
    return result.reshape((X, Y))