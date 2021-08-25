from typing import List, Optional

import numpy as np
from dotmap import DotMap
from trajectory.trajectory import Trajectory


class SimState:
    # gross "forward declaration" workaround for circular deps
    pass


class Objective(object):
    def evaluate_objective(self, trajectory: Trajectory) -> np.ndarray:
        raise NotImplementedError


class ObjectiveFunction(object):
    """
    Define an objective function.
    """

    def __init__(self, params: DotMap):
        self.params: DotMap = params
        self.objectives: List[Objective] = []

    def add_objective(self, objective: Objective) -> None:
        """
        Add an objective to the objective function.

        """
        self.objectives.append(objective)

    def evaluate_function_by_objective(
        self, trajectory, sim_state_hist: Optional[List[SimState]] = None
    ) -> List[List[str and np.ndarray]]:
        """
        Evaluate each objective corresponding to a system trajectory or sim_state
        sim_states are only relevant for personal_space cost functions
        """

        objective_values_by_tag: List[List[str and np.ndarray]] = []
        # import here to avoid circular dep error
        from objectives.personal_space_cost import PersonalSpaceCost

        for objective in self.objectives:
            if isinstance(objective, PersonalSpaceCost):
                obj_value = objective.evaluate_objective(trajectory, sim_state_hist)
            else:
                obj_value = objective.evaluate_objective(trajectory)
            objective_values_by_tag += [[objective.tag, obj_value]]
        return objective_values_by_tag

    def evaluate_function(
        self, trajectory: Trajectory, sim_state_hist: Optional[List[SimState]] = None
    ) -> float:
        """
        Evaluate the entire objective function corresponding to a system trajectory or traj+sim_state.
        sim_states are only relevant for personal_space cost functions
        """
        objective_values_by_tag = self.evaluate_function_by_objective(
            trajectory, sim_state_hist
        )
        objective_function_values = 0.0
        for tag, objective_values in objective_values_by_tag:
            objective_function_values += self._reduce_objective_values(
                trajectory, objective_values
            )
        return objective_function_values

    def _reduce_objective_values(
        self, trajectory: Trajectory, objective_values: np.ndarray
    ) -> np.ndarray:
        """Reduce objective_values according to
        self.params.obj_type."""
        if self.params.obj_type == "mean":
            res = np.mean(objective_values, axis=1)
        elif self.params.obj_type == "valid_mean":
            valid_mask_nk = trajectory.valid_mask_nk
            obj_sum = np.sum(objective_values * valid_mask_nk, axis=1)
            res = obj_sum / trajectory.valid_horizons_n1[:, 0]
        else:
            assert False
        return res
