import numpy as np
import copy

from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.personal_space_cost import PersonalSpaceCost

from trajectory.trajectory import SystemConfig, Trajectory
from utils.fmm_map import FmmMap
from utils.utils import *
from agents.agent_base import AgentBase
from params.central_params import create_agent_params


class Agent(AgentBase):
    sim_t: float = None   # global simulator time that all agents know
    sim_dt: float = None  # simulator (world) refresh rate

    def __init__(self, start, goal, name):
        # Dynamics and movement attributes
        self.fmm_map = None
        self.path_step = 0
        # NOTE: JSON serialization is done within sim_state.py
        self.velocities = {}
        self.accelerations = {}
        # every *planning* agent gets their own copy of 'sim state' history
        self.world_state = None
        super().__init__(start, goal, name)

    def simulation_init(self, sim_map, with_planner: bool = True,
                        with_system_dynamics: bool = True,
                        with_objectives: bool = True,
                        keep_episode_running: bool = False):
        """ Initializes important fields for the CentralSimulator"""
        if(not hasattr(self, "params")):
            self.params = create_agent_params(with_planner=with_planner)
        self.obstacle_map = sim_map
        if(with_objectives):
            # Initialize Fast-Marching-Method map for agent's pathfinding
            self.obj_fn = Agent._init_obj_fn(self)
            Agent._init_fmm_map(self)
        if(with_planner):
            # Initialize planner and vehicle data
            self.planned_next_config = copy.deepcopy(self.current_config)
            self.planner = Agent._init_planner(self)
            self.vehicle_data = self.planner.empty_data_dict()
        if(with_system_dynamics):
            # Initialize system dynamics and planner fields
            self.system_dynamics = Agent._init_system_dynamics(self)
        # the point in the trajectory where the agent collided
        self.collision_point_k = np.inf
        # whether or not to end the episode upon robot collision or continue
        self.keep_episode_running = keep_episode_running
        # default trajectory
        self.trajectory = Trajectory(dt=self.params.dt, n=1, k=0)

    # Setters for the Agent class

    def update_world(self, state):
        self.world_state = state

    @staticmethod
    def set_sim_dt(sim_dt):
        # all the agents know the same simulator refresh rate
        Agent.sim_dt = sim_dt

    @staticmethod
    def set_sim_t(t):
        # all agents know the same world time
        Agent.sim_t = t

    @staticmethod
    def restart_coloring():
        AgentBase.color_indx = 0

    def update(self, sim_state=None):
        """ Run the agent.plan() and agent.act() functions to generate a path and follow it """
        self.sense(sim_state)
        self.plan()
        self.act()

    def sense(self, sim_state, dt: int = 0.05):
        self.update_world(sim_state)

    def plan(self):
        """
        Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and generates relevant planner data
        NOTE: this planner only considers obstacles in the environment, not
        collisions with other agents/personal space
        """
        assert(hasattr(self, 'planner'))
        assert(self.sim_dt is not None)
        # Generate the next trajectory segment, update next config, update actions/data
        if self.end_acting:
            return

        self.planner_data = self.planner.optimize(self.planned_next_config,
                                                  self.goal_config)
        traj_segment = \
            Trajectory.new_traj_clip_along_time_axis(self.planner_data['trajectory'],
                                                     self.params.control_horizon,
                                                     repeat_second_to_last_speed=True)
        self.planned_next_config = \
            SystemConfig.init_config_from_trajectory_time_index(
                traj_segment, t=-1)

        tr_acc = self.params.planner_params.track_accel
        self.trajectory.append_along_time_axis(traj_segment,
                                               track_trajectory_acceleration=tr_acc)
        self.enforce_termination_conditions()

    def act(self):
        """ A utility method to initialize a config object
        from a particular timestep of a given trajectory object"""
        if self.end_acting:
            # exit if there is no more acting to do
            return
        assert(self.sim_dt is not None)
        step = int(np.floor(self.sim_dt / self.params.dt))

        # first check for collisions with any other gen_agents
        self.check_collisions(self.world_state)

        # then update the current config incrementally (can teleport to end if t=-1)
        new_config = SystemConfig.init_config_from_trajectory_time_index(self.trajectory,
                                                                         t=self.path_step)
        self.set_current_config(new_config)

        # updating "next step" for agent path after traversing it
        self.path_step += step

        # considers a full on collision once the agent has passed its "collision point"
        if self.path_step >= self.trajectory.k or self.path_step >= self.collision_point_k:
            self.end_acting = True

        if self.end_acting:
            if self.params.verbose:
                print("terminated act for agent", self.get_name())
            # save memory by deleting control pipeline (very memory intensive)
            del self.planner

        # reset the collision cooldown (to "uncollided") if it has reached 0
        if(self.collision_cooldown > 0):
            self.collision_cooldown -= 1

    """BEGIN HELPER FUNCTIONS"""

    def process_planner_data(self):
        """
        Process the planners current plan. This could mean applying
        open loop control or LQR feedback control on a system.
        """
        # The 'plan' is open loop control
        if 'trajectory' not in self.planner_data.keys():
            trajectory, vel_cmds = \
                Agent.apply_control_open_loop(self, self.current_config,
                                              self.planner_data['optimal_control_nk2'],
                                              T=self.params.control_horizon - 1,
                                              sim_mode=self.system_dynamics.simulation_params.simulation_mode)
        # The 'plan' is LQR feedback control
        else:
            # If we are using ideal system dynamics the planned trajectory
            # is already dynamically feasible. Clip it to the control horizon
            if self.system_dynamics.simulation_params.simulation_mode == 'ideal':
                trajectory = \
                    Trajectory.new_traj_clip_along_time_axis(self.planner_data['trajectory'],
                                                             self.params.control_horizon,
                                                             repeat_second_to_last_speed=True)
            elif self.system_dynamics.simulation_params.simulation_mode == 'realistic':
                trajectory, vel_cmds = \
                    Agent.apply_control_closed_loop(self.current_config,
                                                    self.planner_data['spline_trajectory'],
                                                    self.planner_data['k_nkf1'],
                                                    self.planner_data['K_nkfd'],
                                                    T=self.params.control_horizon - 1,
                                                    sim_mode='realistic')
            else:
                assert False
        # NOTE: can also obtain velocity commands (linear & angular) with the following:
        # commanded_actions_nkf = self.system_dynamics.parse_trajectory(trajectory)
        self.planner.clip_data_along_time_axis(self.planner_data,
                                               self.params.control_horizon)
        return trajectory, self.planner_data

    """BEGIN STATIC HELPER FUNCTIONS"""

    @staticmethod
    def _init_obj_fn(self, params=None):
        """
        Initialize the objective function given sim params
        """
        if params is None:
            params = self.params
        obstacle_map = self.obstacle_map
        obj_fn = ObjectiveFunction(params.objective_fn_params)
        if not params.avoid_obstacle_objective.empty():
            obj_fn.add_objective(
                ObstacleAvoidance(params=params.avoid_obstacle_objective,
                                  obstacle_map=obstacle_map))
        if not params.goal_distance_objective.empty():
            obj_fn.add_objective(
                GoalDistance(params=params.goal_distance_objective,
                             fmm_map=obstacle_map.fmm_map))
        if not params.goal_angle_objective.empty():
            obj_fn.add_objective(
                AngleDistance(params=params.goal_angle_objective,
                              fmm_map=obstacle_map.fmm_map))
        return obj_fn

    @staticmethod
    def _update_obj_fn(self):
        """ 
        Update the objective function to use a new obstacle_map and fmm map
        """
        for objective in self.obj_fn.objectives:
            if isinstance(objective, ObstacleAvoidance):
                objective.obstacle_map = self.obstacle_map
            elif isinstance(objective, GoalDistance):
                objective.fmm_map = self.fmm_map
            elif isinstance(objective, AngleDistance):
                objective.fmm_map = self.fmm_map
            elif isinstance(objective, PersonalSpaceCost):
                pass
            else:
                assert False

    @staticmethod
    def _init_psc_objective(params):
        return PersonalSpaceCost(params=params.personal_space_objective)

    @staticmethod
    def _init_planner(self, params=None):
        if(params is None):
            params = self.params
        return params.planner_params.planner(obj_fn=self.obj_fn,
                                             params=params.planner_params)

    @staticmethod
    def _init_fmm_map(self, goal_pos_n2=None, params=None):
        if(params is None):
            params = self.params
        obstacle_map = self.obstacle_map
        obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]
        self.fmm_map = \
            FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                         map_size_2=np.array(
                                                             self.obstacle_map.get_map_size_2()),
                                                         dx=self.obstacle_map.get_dx(),
                                                         map_origin_2=self.obstacle_map.get_map_origin_2(),
                                                         mask_grid_mn=obstacle_occupancy_grid)
        Agent._update_fmm_map(self)

    @staticmethod
    def _init_system_dynamics(self, params=None):
        """
        If there is a control pipeline (i.e. model based method)
        return its system_dynamics. Else create a new system_dynamics
        instance.
        """
        if params is None:
            params = self.params
        try:
            planner = self.planner
            return planner.control_pipeline.system_dynamics
        except AttributeError:
            p = params.system_dynamics_params
            return p.system(dt=p.dt, params=p)

    @staticmethod
    def _update_fmm_map(self, params=None):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
        assert(hasattr(self, 'fmm_map'))
        if self.fmm_map is not None:
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = Agent._init_fmm_map(self, params=params)
        Agent._update_obj_fn(self)

    # wrapper functions for the helper base class
    @staticmethod
    def apply_control_open_loop(self, start_config, control_nk2,
                                T, sim_mode='ideal'):
        return super().apply_control_open_loop(self, start_config,
                                               control_nk2, T, sim_mode)

    @staticmethod
    def apply_control_closed_loop(self, start_config, trajectory_ref,
                                  k_array_nTf1, K_array_nTfd, T,
                                  sim_mode='ideal'):
        return super().apply_control_closed_loop(self, start_config, trajectory_ref,
                                                 k_array_nTf1, K_array_nTfd, T,
                                                 sim_mode='ideal')
