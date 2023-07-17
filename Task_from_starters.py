import numpy as np
# Numexpr is used for parallel operations
import numexpr as ne


def first_task(A, B):
    """
    Calculates mean correlation between A[i,j,:] trace and all traces from B array

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


def _create_blocks(A, K):
    """
    Creates K x K x Z blocks from original array, and reshapes it into K^2 x Z blocks

    Returns:
        array of shape: (n_blocks, K^2, Z)
    """

    itemsize = A.itemsize
    X, Y, Z = A.shape

    BATCH_SIZE = (K, K, Z)
    x_blocks = A.shape[0] - (K - 1)
    y_blocks = A.shape[1] - (K - 1)
    n_blocks = x_blocks * y_blocks

    blocks = np.lib.stride_tricks.as_strided(
        A,
        shape = (n_blocks, K, K, Z),
        strides = (
            itemsize*Z,
            itemsize*Z*Y,
            itemsize*Z,
            itemsize
        )
    )
    # flattening kxkxz traces into k^2xz so that central trace will be in center of resulting array
    blocks = blocks.reshape(n_blocks, K*K, Z)
    return blocks, n_blocks, x_blocks, y_blocks


def second_task(A, K):
    """
    Calculates mean correlation between A[i,j,:] trace and traces that surround if within KxK window

    Returns:
        array of shape (x_blocks, y_blocks), where x_blocks and y_blocks are number of windows that can fit
        in each of the axes of original array.
    """
    X, Y, Z = A.shape

    blocks, n_blocks, x_blocks, y_blocks = _create_blocks(A, K)
    central_tr = blocks[:, blocks.shape[1]//2, :].reshape(n_blocks, 1, Z) 

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

    return result.reshape((x_blocks, y_blocks))