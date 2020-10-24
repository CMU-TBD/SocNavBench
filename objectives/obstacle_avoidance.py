import numpy as np

from objectives.objective_function import Objective


class ObstacleAvoidance(Objective):
    """
    Define the obstacle avoidance objective. Penalty is incurred for distances closer to the
    obstacle than obstacle_margin1. Cost is normalized by a normalization factor ensuring
    the cost is 1 at obstacle_margin0.
    """

    def __init__(self, params, obstacle_map):
        assert(params.obstacle_margin0 <= params.obstacle_margin1)
        self.factor = params.obstacle_margin1 - params.obstacle_margin0
        self.p = params
        self.obstacle_map = obstacle_map
        self.tag = 'obstacle_avoidance'

    def evaluate_objective(self, trajectory):
        dist_to_obstacles_nk = self.obstacle_map.dist_to_nearest_obs(
            trajectory.position_nk2())
        infringement_nk = np.maximum(
            self.p.obstacle_margin1 - dist_to_obstacles_nk, 0)
        return self.p.obstacle_cost * np.power(infringement_nk / self.factor, self.p.power)
