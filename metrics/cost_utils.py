import numpy as np


def vel2sig(vel: float) -> float:
    """
    Computes variance of asymmetric gaussian cost in the direction of motion
    based on Rachel Kirby

    Input
    ------
    vel in m/s

    Output
    -------
    max(2v, 1/2)

    """
    return np.maximum(2 * vel, 0.5)
