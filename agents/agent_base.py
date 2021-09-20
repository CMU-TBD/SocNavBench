import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotmap import DotMap
from objectives.goal_distance import GoalDistance
from objectives.objective_function import ObjectiveFunction
from obstacles.sbpd_map import SBPDMap
from planners.planner import Planner
from systems.dynamics import Dynamics
from trajectory.trajectory import SystemConfig, Trajectory
from utils.angle_utils import angle_normalize
from utils.fmm_map import FmmMap
from utils.utils import euclidean_dist2, generate_name


class AgentBase(object):
    color_indx: int = 0
    possible_colors: List[str] = ["b", "g", "r", "c", "m", "y"]

    def __init__(
        self, start: SystemConfig, goal: SystemConfig, name: Optional[str] = None
    ):
        self.name: str = name if name else generate_name(20)
        self.start_config: SystemConfig = start
        self.goal_config: SystemConfig = goal
        # upon initialization, the current config of the agent is start
        self.current_config: SystemConfig = copy.deepcopy(start)
        # path planning and acting fields
        self.end_acting: bool = False
        # default termination cause is timeout
        self.termination_cause: str = "Timeout"
        # name of the agent that the agent collided with (if applicable)
        self.latest_collider: str = ""
        # cooldown (in simulator updates) before colliding with another agent
        self.collision_cooldown: int = 0
        # when the last collision occured
        self.last_collision_t: float = 1e8  # basically infinity but printable
        # Whether to continue the episode even if the robot collides with a pedestrian
        self.keep_episode_running: bool = True
        # cosmetic items (for drawing the trajectories)
        self.trajectory_color: str = AgentBase.init_colors()
        # initialize some trajectory to be updated via actions
        self.trajectory: Trajectory = None
        # default planner fields
        self.init_planner_fields()
        # no params yet
        self.params: DotMap = None

    def init_planner_fields(self):
        self.obj_fn: ObjectiveFunction = None
        self.obstacle_map: SBPDMap = None
        self.fmm_map: FmmMap = None
        self.system_dynamics: Dynamics = None
        self.planner: Planner = None
        self.planner_data: Dict[str, Any] = None
        self.vehicle_data: Dict = None

    # Getters for the Agent class
    def get_name(self) -> str:
        return self.name

    def get_config(self, config: SystemConfig, deepcpy: bool) -> SystemConfig:
        if deepcpy:
            return SystemConfig.copy(config)
        return config

    def get_start_config(self, deepcpy: Optional[bool] = False) -> SystemConfig:
        return self.get_config(self.start_config, deepcpy)

    def set_start_config(self, start: SystemConfig) -> None:
        self.start_config = start

    def get_goal_config(self, deepcpy: Optional[bool] = False) -> SystemConfig:
        return self.get_config(self.goal_config, deepcpy)

    def set_goal_config(self, goal: SystemConfig) -> None:
        self.goal_config = goal

    def get_current_config(self, deepcpy: Optional[bool] = False) -> SystemConfig:
        return self.get_config(self.current_config, deepcpy)

    def set_current_config(self, current: SystemConfig) -> None:
        self.current_config = current

    def get_trajectory(self, deepcpy: Optional[bool] = False) -> Trajectory:
        if deepcpy:
            return Trajectory.copy(self.trajectory, check_dimens=False)
        return self.trajectory

    def get_end_acting(self) -> bool:
        return self.end_acting

    def get_collided(self) -> bool:
        return "Collision" in self.termination_cause

    def get_completed(self) -> bool:
        return self.get_end_acting() and self.termination_cause == "Success"

    def get_collision_cooldown(self) -> int:
        return self.collision_cooldown

    def get_radius(self) -> float:
        return self.params.radius

    def get_color(self) -> str:
        return self.trajectory_color

    @staticmethod
    def generate() -> None:
        # used to spawn all the agents into the simulator, defined per class
        pass

    @staticmethod
    def init_colors() -> str:
        # cosmetic items (for drawing the trajectories)
        if AgentBase.color_indx < len(AgentBase.possible_colors) - 1:
            AgentBase.color_indx = AgentBase.color_indx + 1
        else:
            AgentBase.color_indx = 0
        return AgentBase.possible_colors[AgentBase.color_indx]

    def sense(self) -> None:
        raise NotImplementedError

    def plan(self) -> None:
        raise NotImplementedError

    def act(self) -> None:
        raise NotImplementedError

    """AGENT UTILS"""

    def _collision_in_group(self, own_pos: np.ndarray, group: List) -> bool:
        for a in group:
            othr_pos = a.get_current_config().position_and_heading_nk3(squeeze=True)
            is_same_agent: bool = a.get_name() is self.get_name()
            if (
                not is_same_agent
                and a.get_collision_cooldown() == 0
                and euclidean_dist2(own_pos, othr_pos)
                < self.get_radius() + a.get_radius()
            ):
                # instantly collide (with agent) and stop updating
                self.termination_cause = "Pedestrian Collision"
                # name of the latest agent that the agent collided with (applicable)
                self.latest_collider = a.get_name()
                if not self.keep_episode_running:
                    self.end_acting = True
                    self.collision_point_k = self.trajectory.k  # this instant
                return True
        # reached here means no collisions have occured, therefore there is no latest_collider
        self.latest_collider = ""
        self.termination_cause = (
            "Timeout"  # no more collisions with these group members
        )
        return False

    def check_collisions(
        self,
        world_state,
        include_agents: Optional[bool] = True,
        include_robots: Optional[bool] = True,
    ) -> bool:
        # TODO: world_state is a SimState, but including it yields a circular dependency error
        if self.collision_cooldown > 0:
            # no double collisions
            return False
        if world_state is not None:
            own_pos = self.get_current_config().position_and_heading_nk3(squeeze=True)
            collided_with_robot = include_robots and self._collision_in_group(
                own_pos, world_state.get_robots().values()
            )
            collided_with_pedestrian = include_agents and self._collision_in_group(
                own_pos, world_state.get_pedestrians().values()
            )
            if collided_with_robot or collided_with_pedestrian:
                self.last_collision_t = world_state.get_sim_t()
                return True
        return False

    def enforce_termination_conditions(self) -> bool:
        assert self.obj_fn is not None  # to calculate objective values
        assert self.obstacle_map is not None  # to check obstacle collisions
        p = self.params
        time_idxs: List[np.ndarray] = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(
                self._compute_time_idx_for_termination_condition(condition)
            )
        try:
            idx = np.argmin(time_idxs)
        except ValueError:
            idx = np.argmin([time_idx for time_idx in time_idxs])

        try:
            termination_time = time_idxs[idx]
        except ValueError:
            termination_time = time_idxs[idx]

        if termination_time != np.inf:
            end_episode = True
            for i, condition in enumerate(p.episode_termination_reasons):
                if time_idxs[i] != np.inf:
                    self.termination_cause = condition
                    self.collision_point_k = termination_time
            self.trajectory.clip_along_time_axis(termination_time)
            if self.planner is not None and self.planner_data is not None:
                (
                    self.planner_data,
                    planner_data_last_step,
                    last_step_data_valid,
                ) = self.planner.mask_and_concat_data_along_batch_dim(
                    self.planner_data, k=termination_time
                )
                # If all of the data was masked then
                # the episode simulated is not valid
                valid_episode = True
                if self.planner_data["system_config"] is None:
                    valid_episode = False
                episode_data = {
                    "vehicle_trajectory": self.trajectory,
                    "vehicle_data": self.planner_data,
                    "vehicle_data_last_step": planner_data_last_step,
                    "last_step_data_valid": last_step_data_valid,
                    "episode_type": idx,
                    "valid_episode": valid_episode,
                }
            else:
                episode_data = {}
        else:
            end_episode = False
            episode_data = {}
        self.episode_data = episode_data
        return end_episode

    def _compute_time_idx_for_termination_condition(self, condition: str) -> np.ndarray:
        """
        For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        infinity if a condition is not met.
        """
        # TODO: clean this up
        if condition == "Timeout":
            time_idx = self._compute_time_idx_for_timeout()
        elif condition == "Obstacle Collision":
            time_idx = self._compute_time_idx_for_collision()
        elif condition == "Pedestrian Collision":
            # TODO: implement here
            time_idx = np.inf  # implemented elsewhere in Agent
        elif condition == "Success":
            time_idx = self._compute_time_idx_for_success()
        else:
            raise NotImplementedError

        return time_idx

    def _compute_time_idx_for_timeout(self) -> np.ndarray:
        """
        If vehicle_trajectory has exceeded episode_horizon,
        return episode_horizon, else return infinity.
        """
        if self.planner is not None:
            if self.trajectory.k >= self.params.episode_horizon:
                time_idx = np.array(self.params.episode_horizon)
            else:
                time_idx = np.array(np.inf)
        else:
            time_idx = np.array(np.inf)
        return time_idx

    def _compute_time_idx_for_collision(
        self, use_current_config: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute and return the earliest time index of collision in vehicle
        trajectory. If there is no collision return infinity.
        """
        if use_current_config is None:
            pos_1k2 = self.trajectory.position_nk2()
        else:
            pos_1k2 = self.get_current_config().position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = np.where(np.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[1]
        if np.size(collision_idxs) != 0:
            time_idx = collision_idxs[0]
            self.collision_point_k = self.trajectory.k
        else:
            time_idx = np.array(np.inf)
        return time_idx

    def _dist_to_goal(self) -> np.ndarray:
        """Calculate the FMM distance between
        each state in trajectory and the goal."""
        for objective in self.obj_fn.objectives:
            if isinstance(objective, GoalDistance):
                dist_to_goal_nk = objective.compute_dist_to_goal_nk(self.trajectory)
        return dist_to_goal_nk

    def _compute_time_idx_for_success(self) -> np.ndarray:
        """
        Compute and return the earliest time index of success (reaching the goal region)
        in vehicle trajectory. If there is no collision return infinity.
        """
        dist_to_goal_1k = self._dist_to_goal()
        successes = np.where(np.less(dist_to_goal_1k, self.params.goal_margin))
        success_idxs = successes[1]
        if np.size(success_idxs) != 0:
            time_idx = success_idxs[0]
        else:
            time_idx = np.array(np.inf)
        return time_idx

    @staticmethod
    def apply_control_open_loop(
        self,
        start_config: SystemConfig,
        control_nk2: np.ndarray,
        T: int,
        sim_mode: Optional[str] = "ideal",
    ) -> Tuple[Trajectory, np.ndarray]:
        """
        Apply control commands in control_nk2 in an open loop
        fashion to the system starting from start_config.
        """
        x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
        applied_actions: List[np.ndarray] = []
        states: List[float] = [x0_n1d * 1.0]
        x_next_n1d = x0_n1d * 1.0
        for t in range(T):
            u_n1f = control_nk2[:, t : t + 1]
            x_next_n1d = self.system_dynamics.simulate(x_next_n1d, u_n1f, mode=sim_mode)

            # Append the applied action to the action list
            if sim_mode == "ideal":
                applied_actions.append(u_n1f)
            elif sim_mode == "realistic":
                # TODO: This line is intended for a real hardware setup.
                # If running this code on a real robot the user will need to
                # implement hardware.state_dx such that it reflects the current
                # sensor reading of the robot's applied actions
                applied_actions.append(
                    np.array(self.system_dynamics.hardware.state_dx * 1.0)[None, None]
                )
            else:
                assert False

            states.append(x_next_n1d)

        commanded_actions_nkf = np.concatenate([control_nk2[:, :T], u_n1f], axis=1)
        u_nkf = np.concatenate(applied_actions, axis=1)
        x_nkd = np.concatenate(states, axis=1)
        trajectory = self.system_dynamics.assemble_trajectory(
            x_nkd, u_nkf, pad_mode="repeat"
        )
        return trajectory, commanded_actions_nkf

    @staticmethod
    def apply_control_closed_loop(
        self,
        start_config: SystemConfig,
        trajectory_ref: Trajectory,
        k_array_nTf1: np.ndarray,
        K_array_nTfd: np.ndarray,
        T: int,
        sim_mode: Optional[str] = "ideal",
    ) -> Tuple[Trajectory, np.ndarray]:
        """
        Apply LQR feedback control to the system to track trajectory_ref
        Here k_array_nTf1 and K_array_nTfd are tensors of dimension
        (n, self.T-1, f, 1) and (n, self.T-1, f, d) respectively.
        """
        with np.name_scope("apply_control"):
            x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
            assert len(x0_n1d.shape) == 3  # [n,1,x_dim]
            angle_dims = self.system_dynamics._angle_dims
            commanded_actions_nkf = []
            applied_actions = []
            states = [x0_n1d * 1.0]
            x_ref_nkd, u_ref_nkf = self.system_dynamics.parse_trajectory(trajectory_ref)
            x_next_n1d = x0_n1d * 1.0
            angle_norm = angle_normalize  # shorter function name
            for t in range(T):
                x_ref_n1d, u_ref_n1f = x_ref_nkd[:, t : t + 1], u_ref_nkf[:, t : t + 1]
                error_t_n1d = x_next_n1d - x_ref_n1d

                norm_angle = angle_norm(error_t_n1d[:, :, angle_dims : angle_dims + 1])

                # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT
                # is not working to place non-gpu ops (i.e. mod) on the cpu
                # turning tensors into numpy arrays is a hack around this.
                error_t_n1d = np.concatenate(
                    [
                        error_t_n1d[:, :, :angle_dims],
                        norm_angle,
                        error_t_n1d[:, :, angle_dims + 1 :],
                    ],
                    axis=2,
                )
                fdback_nf1 = np.matmul(
                    K_array_nTfd[:, t], np.transpose(error_t_n1d, perm=[0, 2, 1])
                )
                u_n1f = u_ref_n1f + np.transpose(
                    k_array_nTf1[:, t] + fdback_nf1, perm=[0, 2, 1]
                )

                x_next_n1d = self.system_dynamics.simulate(
                    x_next_n1d, u_n1f, mode=sim_mode
                )

                commanded_actions_nkf.append(u_n1f)
                # Append the applied action to the action list
                if sim_mode == "ideal":
                    applied_actions.append(u_n1f)
                elif sim_mode == "realistic":
                    # TODO: This line is intended for a real hardware setup.
                    # If running this code on a real robot the user will need to
                    # implement hardware.state_dx such that it reflects the current
                    # sensor reading of the robot's applied actions
                    applied_actions.append(
                        np.array(self.system_dynamics.hardware.state_dx * 1.0)[
                            None, None
                        ]
                    )
                else:
                    assert False

                states.append(x_next_n1d)

            commanded_actions_nkf.append(u_n1f)
            commanded_actions_nkf = np.concatenate(commanded_actions_nkf, axis=1)
            u_nkf = np.concatenate(applied_actions, axis=1)
            x_nkd = np.concatenate(states, axis=1)
            trajectory = self.system_dynamics.assemble_trajectory(
                x_nkd, u_nkf, pad_mode="repeat"
            )
            return trajectory, commanded_actions_nkf
