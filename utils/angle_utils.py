from typing import Optional, Tuple

import numpy as np

# Angle normalization function


def angle_normalize(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def rotate_pos_nk2(pos_nk2: np.ndarray, theta_n11: np.ndarray) -> np.ndarray:
    """ Utility function to rotate positions in pos_nk2
    by angles indicated in theta_n11. Assumes the rotation
    does not vary over time (hence theta_n11 not theta_nk1)."""
    # Broadcast theta_n11 to size nk1. broadcast_to does not track gradients so addition is used
    # here instead
    # theta_nk1 = theta_n11 + 0. * pos_nk2[:, :, 0:1]
    t_0 = theta_n11[0][0]  # since theta_n11 is nx1x1 matrix
    # (element wise vector multiplication)
    theta_nk1 = np.ones_like(pos_nk2) * t_0
    # theta_nk1 = theta_n11 + 0. * pos_nk2[:, :, 0:1]
    if type(theta_nk1) is np.ndarray:
        n, k, _ = [x for x in theta_nk1.shape]
    else:
        n, k, _ = [x.value for x in theta_nk1.shape]
    rot_matrix_nk22 = padded_rotation_matrix(theta_n11, shape=(n, k, 2))
    pos_rot_nk2 = np.matmul(rot_matrix_nk22, pos_nk2[:, :, :, None])[:, :, :, 0]
    return pos_rot_nk2


def padded_rotation_matrix(
    theta_n11: np.ndarray,
    shape: Tuple[float, float, float],
    lower_identity: Optional[bool] = False,
) -> np.ndarray:
    """ Returns a rotation matrix of shape (n, k, d, d)
    where the first (n, k, 2, 2) elements correspond to
    a 2d rotation matrix for each element in the batch
    (broadcast across time). The rest of the elements are
    zero. If lower_identity is true the lower right
    e x e matrix is set to the be the identity (here e=d-2)."""
    n, k, d = shape
    e = d - 2
    assert d >= 2
    dtype = theta_n11.dtype
    t_0 = theta_n11[0][0]
    # theta_nk11 = np.broadcast_to(theta_n11[:, None], (n, k, 1, 1))
    theta_nk11 = np.ones(shape=(n, k, 1, 1), dtype=dtype) * t_0

    first_row_nkd1 = np.concatenate(
        [
            np.cos(theta_nk11),
            np.sin(theta_nk11),
            np.zeros((n, k, d - 2, 1), dtype=dtype),
        ],
        axis=2,
    )
    second_row_nkd1 = np.concatenate(
        [
            -np.sin(theta_nk11),
            np.cos(theta_nk11),
            np.zeros((n, k, d - 2, 1), dtype=dtype),
        ],
        axis=2,
    )

    # If lower_identity is true, make the lower right
    # e x e matrix the identity matrix
    if lower_identity:
        # identity_block_nkee = np.eye(e, dtype=dtype).reshape(n, k)
        identity_block_nkee = []
        identity_block_nkee.append([])
        for _ in range(k):
            identity_block_nkee[0].append([[1]])
        identity_block_nkee = np.array(identity_block_nkee, dtype=dtype)
        remaining_rows_nkde = np.concatenate(
            [np.zeros((n, k, 2, e), dtype=dtype), identity_block_nkee], axis=2
        )
    else:
        remaining_rows_nkde = np.zeros((n, k, d, d - 2), dtype=dtype)
    rot_matrix_nkdd = np.concatenate(
        [first_row_nkd1, second_row_nkd1, remaining_rows_nkde], axis=3
    )
    return rot_matrix_nkdd
