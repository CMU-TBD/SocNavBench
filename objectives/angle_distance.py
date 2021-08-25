import numpy as np
from dotmap import DotMap
from trajectory.trajectory import Trajectory
from utils.angle_utils import angle_normalize
from utils.fmm_map import FmmMap

from objectives.objective_function import Objective


class AngleDistance(Objective):
    """
    Compute the angular distance to the optimal path.
    """

    def __init__(self, params: DotMap, fmm_map: FmmMap):
        self.p: DotMap = params
        self.fmm_map: FmmMap = fmm_map
        self.tag: str = "angular_distance_to_optimal_direction"

    def evaluate_objective(self, trajectory: Trajectory) -> np.ndarray:
        optimal_angular_orientation_nk = self.fmm_map.fmm_angle_map.compute_voxel_function(
            trajectory.position_nk2()
        )
        angular_dist_to_optimal_path_nk = angle_normalize(
            trajectory.heading_nk1()[:, :, 0] - optimal_angular_orientation_nk
        )
        return self.p.angle_cost * np.power(
            np.abs(angular_dist_to_optimal_path_nk), self.p.power
        )
