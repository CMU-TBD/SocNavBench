import numpy as np
from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from trajectory.trajectory import Trajectory

from objectives.objective_function import Objective


class ObstacleAvoidance(Objective):
    """
    Define the obstacle avoidance objective. Penalty is incurred for distances closer to the
    obstacle than obstacle_margin1. Cost is normalized by a normalization factor ensuring
    the cost is 1 at obstacle_margin0.
    """

    def __init__(self, params: DotMap, obstacle_map: SBPDMap):
        assert params.obstacle_margin0 <= params.obstacle_margin1
        self.factor: float = params.obstacle_margin1 - params.obstacle_margin0
        self.p: DotMap = params
        self.obstacle_map: SBPDMap = obstacle_map
        self.tag: str = "obstacle_avoidance"

    def evaluate_objective(self, trajectory: Trajectory) -> np.ndarray:
        dist_to_obstacles_nk: np.ndarray = self.obstacle_map.dist_to_nearest_obs(
            trajectory.position_nk2()
        )
        infringement_nk: np.ndarray = np.maximum(
            self.p.obstacle_margin1 - dist_to_obstacles_nk, 0
        )
        return self.p.obstacle_cost * np.power(
            infringement_nk / self.factor, self.p.power
        )
