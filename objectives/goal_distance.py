import numpy as np
from dotmap import DotMap
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap

from objectives.objective_function import Objective


class GoalDistance(Objective):
    """
    Define the goal reaching objective.
    """

    tag: str = "goal_distance"

    def __init__(self, params: DotMap, fmm_map: FmmMap):
        self.p: DotMap = params
        self.fmm_map: FmmMap = fmm_map
        self.tag: str = "distance_to_goal"
        self.cost_at_margin: float = self.p.goal_cost * np.power(
            self.p.goal_margin, self.p.power
        )

    def compute_dist_to_goal_nk(self, trajectory: Trajectory) -> np.ndarray:
        return self.fmm_map.fmm_distance_map.compute_voxel_function(
            trajectory.position_nk2()
        )

    def evaluate_objective(self, trajectory: Trajectory) -> np.ndarray:
        dist_to_goal_nk = self.compute_dist_to_goal_nk(trajectory)
        return (
            self.p.goal_cost * np.power(dist_to_goal_nk, self.p.power)
            - self.cost_at_margin
        )
