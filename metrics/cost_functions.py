from typing import Optional
from trajectory.trajectory import SystemConfig, Trajectory
import numpy as np
from metrics.cost_utils import *


def asym_gauss_from_vel(
    x: float,
    y: float,
    velx: float,
    vely: float,
    xc: Optional[float] = 0,
    yc: Optional[float] = 0,
) -> np.ndarray:
    """
    computation of the value of an arbitrarily rotated (by theta)
    centered at (xc, yc)
    Asymmetric Gaussian at some point (x, y)
    Obviously, the velocities are for the peds
    around whom the gaussian is centered
    Variances are:
    sig_theta: in direction of motion
    sig_r: opp direction of motion (rear)
    sig_s: variance to the sides

    can calculate sig_theta = max(2*velocity, 0.5) [Rachel Kirby thesis 2005?]
    """
    speed = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(vely, velx)
    sig_theta = vel2sig(speed)
    return asym_gauss(x, y, theta, sig_theta, xc=xc, yc=yc)


def asym_gauss(
    x: float,
    y: float,
    theta: Optional[float] = 0,
    sig_theta: Optional[float] = 2,
    xc: Optional[float] = 0,
    yc: Optional[float] = 0,
) -> np.ndarray:
    """
    computation of the value of an arbitrarily rotated (by theta)
    centered at (xc, yc)
    Asymmetric Gaussian at some point (x, y)
    Variances are:
    sig_theta: in direction of motion
    sig_r: opp direction of motion (rear)
    sig_s: variance to the sides

    can calculate sig_theta = max(2*velocity, 0.5) [Rachel Kirby thesis 2005?]
    """
    alpha = np.arctan2(y - yc, x - xc) - theta + np.pi / 2
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    # print(alpha[np.where(alpha>np.pi)])
    # sigma = np.zeros_like(x)
    # sigma = sig_r if alpha <= 0 else sig_h

    sig_s = sig_theta / 4
    sig_r = sig_theta / 3
    sigma = np.where(alpha <= 0, sig_r, sig_theta)

    a = ((np.cos(theta) / sigma) ** 2 + (np.sin(theta) / sig_s) ** 2) / 2
    b = np.sin(2 * theta) * (1 / (sigma ** 2) - 1 / (sig_s ** 2)) / 4
    c = ((np.sin(theta) / sigma) ** 2 + (np.cos(theta) / sig_s) ** 2) / 2

    # gaussian
    agxy = np.exp(
        -(a * (x - xc) ** 2 + 2 * b * (x - xc) * (y - yc) + c * (y - yc) ** 2)
    )

    return agxy


def path_length(trajectory: Trajectory) -> float:
    if trajectory.shape[-1] == 3:
        trajectory = trajectory[:, :-1]
    distance_sq = np.sum(np.power(np.diff(trajectory, axis=0), 2), axis=1)
    distance = np.sum(np.sqrt(distance_sq))
    return distance


def path_length_ratio(
    trajectory: Trajectory, goal_config: Optional[SystemConfig] = None
) -> float:
    """
    Returns displacement/distance -- displacement may be zero
    (Distance is an approximation based on the time resolution of stored trajectory)
    Higher should be better but also depends on your exact scenario
    """
    # TODO: make this run in batch mode

    if trajectory.shape[-1] == 3:
        trajectory = trajectory[:, :-1]

    # for incomplete trajectories you want to pass in the aspirational goal
    if goal_config is None:
        goal_config = trajectory[-1, :]

    start_config = trajectory[0, :]
    epsilon = 0.00001  # for numerical stability
    distance = path_length(trajectory) + epsilon
    displacement = np.linalg.norm(goal_config - start_config)
    return distance / displacement


def path_irregularity(
    trajectory: Trajectory, goal_config: Optional[SystemConfig] = None
) -> float:
    """
    defined as the amount of unnecessary turning per unit path length performed by a robot,
    where unnecessary turning corresponds to the
    total amount of robot rotation minus the minimum amount of rotation
    which would be needed to reach the same targets
    with the most direct path. Path irregularity is measured in rad/m
    :return:
    """
    if goal_config is None:
        goal_config = trajectory[-1, :]
    assert trajectory.shape[-1] == 3 and goal_config.shape[-1] == 3

    # To compute the per step angle away from straight line to goal
    # compute the ray to goal from each traj step
    traj_xy = trajectory[:, :-1]
    point_to_goal_traj = np.squeeze(goal_config)[:-1] - traj_xy
    # cos inv of dot product of vectors
    cos_theta = np.sum(point_to_goal_traj * traj_xy, axis=1) / (
        np.linalg.norm(point_to_goal_traj, axis=1) * np.linalg.norm(traj_xy, axis=1)
        + (1 / 1e10)
    )
    theta_to_goal_traj = np.arccos(cos_theta)
    path_irr = np.sum(np.abs(theta_to_goal_traj)) / len(theta_to_goal_traj)

    return path_irr


# def time_to_collision(sim_state):
#     """
#
#     :param sim_state:
#     :return: least time to collision of ego agent to any agent
#     """
#     ttc=0
#     return ttc
