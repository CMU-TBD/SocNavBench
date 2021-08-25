from typing import List, Optional

import numpy as np
from agents.agent import Agent
from dotmap import DotMap
from objectives.objective_function import ObjectiveFunction
from objectives.personal_space_cost import PersonalSpaceCost
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_agent_params
from socnav.socnav_renderer import SocNavRenderer
from trajectory.trajectory import SystemConfig, Trajectory
from utils.utils import euclidean_dist2

from joystick_py.joystick_base import JoystickBase


class JoystickWithPlanner(JoystickBase):
    def __init__(self):
        # planner variables
        # the list of commands sent to the robot to execute
        self.commands: List[str] = []
        self.simulator_joystick_update_ratio: int = 1
        # our 'positions' are modeled as (x, y, theta)
        self.robot_current: np.ndarray = None  # current position of the robot
        self.robot_v: float = 0  # not tracked in the base simulator
        self.robot_w: float = 0  # not tracked in the base simulator
        super().__init__("SamplingPlanner")  # parent class needs to know the algorithm

    def init_obstacle_map(self, renderer: Optional[SocNavRenderer] = 0) -> SBPDMap:
        """ Initializes the sbpd map."""
        p: DotMap = self.agent_params.obstacle_map_params
        env = self.current_ep.get_environment()
        return p.obstacle_map(
            p,
            renderer,
            res=float(env["map_scale"]) * 100.0,
            map_trav=np.array(env["map_traversible"]),
        )

    def init_control_pipeline(self) -> None:
        # NOTE: this is like an init() run *after* obtaining episode metadata
        # robot start and goal to satisfy the old Agent.planner
        self.start_config: SystemConfig = SystemConfig.from_pos3(self.get_robot_start())
        self.goal_config: SystemConfig = SystemConfig.from_pos3(self.get_robot_goal())
        # rest of the 'Agent' params used for the joystick planner
        self.agent_params: DotMap = create_agent_params(
            with_planner=True, with_obstacle_map=True
        )
        # update generic 'Agent params' with joystick-specific params
        self.agent_params.episode_horizon_s = self.joystick_params.episode_horizon_s
        self.agent_params.control_horizon_s = self.joystick_params.control_horizon_s
        # init obstacle map
        self.obstacle_map: SBPDMap = self.init_obstacle_map()
        self.obj_fn: ObjectiveFunction = Agent._init_obj_fn(
            self, params=self.agent_params
        )
        psc_obj = PersonalSpaceCost(params=self.agent_params.personal_space_objective)
        self.obj_fn.add_objective(psc_obj)

        # Initialize Fast-Marching-Method map for agent's pathfinding
        Agent._init_fmm_map(self, params=self.agent_params)

        # Initialize system dynamics and planner fields
        self.planner = Agent._init_planner(self, params=self.agent_params)
        self.vehicle_data = self.planner.empty_data_dict()
        self.system_dynamics = Agent._init_system_dynamics(
            self, params=self.agent_params
        )
        # init robot current config from the starting position
        self.robot_current = self.current_ep.get_robot_start().copy()
        # init a list of commands that will be sent to the robot
        self.commands = None

    def joystick_sense(self):
        # ping's the robot to request a sim state
        self.send_to_robot("sense")

        # store previous pos3 of the robot (x, y, theta)
        robot_prev = self.robot_current.copy()  # copy since its just a list
        # listen to the robot's reply
        self.joystick_on = self.listen_once()

        # NOTE: at this point, self.sim_state_now is updated with the
        # most up-to-date simulation information

        # Update robot current position
        robot = list(self.sim_state_now.get_robots().values())[0]
        self.robot_current = robot.get_current_config().position_and_heading_nk3(
            squeeze=True
        )

        # Updating robot speeds (linear and angular) based off simulator data
        self.robot_v = euclidean_dist2(self.robot_current, robot_prev) / self.sim_dt
        self.robot_w = (self.robot_current[2] - robot_prev[2]) / self.sim_dt

    def joystick_plan(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data
        - Access to sim_states from the self.current_world
        """
        robot_config = SystemConfig.from_pos3(
            pos3=self.robot_current,
            dt=self.agent_params.dt,
            v=self.robot_v,
            w=self.robot_w,
        )
        self.planner_data = self.planner.optimize(
            robot_config, self.goal_config, sim_state_hist=self.sim_states
        )

        # TODO: make sure the planning control horizon is greater than the
        # simulator_joystick_update_ratio else it will not plan far enough

        # LQR feedback control loop
        t_seg = Trajectory.new_traj_clip_along_time_axis(
            self.planner_data["trajectory"],
            self.agent_params.control_horizon,
            repeat_second_to_last_speed=True,
        )
        # From the new planned subtrajectory, parse it for the requisite v & w commands
        _, cmd_actions_nkf = self.system_dynamics.parse_trajectory(t_seg)
        self.commands = cmd_actions_nkf[0]

    def joystick_act(self):
        if self.joystick_on:
            # sends velocity commands within the robot's system dynamics
            assert self.joystick_params.use_system_dynamics
            # runs through the entire planned horizon just with a cmds_step
            num_cmds_per_step = self.simulator_joystick_update_ratio
            # get velocity bounds from the system dynamics params
            self.v_bounds = self.system_dynamics_params.v_bounds
            self.w_bounds = self.system_dynamics_params.w_bounds
            for _ in range(int(np.floor(len(self.commands) / num_cmds_per_step))):
                # initialize the command containers
                velocity_cmds = []
                # only going to send the first simulator_joystick_update_ratio commands
                clipped_cmds = self.commands[:num_cmds_per_step]
                for v_cmd, w_cmd in clipped_cmds:
                    velocity_cmds.append((float(v_cmd), float(w_cmd)))
                self.send_cmds(velocity_cmds, send_vel_cmds=True)
                # remove the sent commands
                self.commands = self.commands[num_cmds_per_step:]
                # break if the robot finished
                if not self.joystick_on:
                    break

    def update_loop(self):
        super().pre_update()  # pre-update initialization
        self.simulator_joystick_update_ratio = int(
            np.floor(self.sim_dt / self.agent_params.dt)
        )
        while self.joystick_on:
            # gather information about the world state based off the simulator
            self.joystick_sense()
            # create a plan for the next steps of the trajectory
            self.joystick_plan()
            # send a command to the robot
            self.joystick_act()
        # complete this episode, move on to the next if need be
        self.finish_episode()


class JoystickWithPlannerPosns(JoystickWithPlanner):
    def __init__(self):
        super().__init__()
        # sends positional commands with no notion of system dynamics
        assert not self.joystick_params.use_system_dynamics

    def from_conf(self, configs, idx):
        x = float(configs._position_nk2[0][idx][0])
        y = float(configs._position_nk2[0][idx][1])
        th = float(configs._heading_nk1[0][idx][0])
        v = float(configs._speed_nk1[0][idx][0])
        return (x, y, th, v)

    def joystick_plan(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data
        - Access to sim_states from the self.current_world
        """
        # get information about robot by its "current position" which was updated in sense()
        [x, y, th] = self.robot_current
        v = self.robot_v
        # can also try:
        #     # assumes the robot has executed all the previous commands in self.commands
        #     (x, y, th, v) = self.from_conf(self.commands, -1)
        robot_config = SystemConfig.from_pos3(pos3=(x, y, th), v=v)
        self.planner_data = self.planner.optimize(
            robot_config, self.goal_config, sim_state_hist=self.sim_states
        )

        # TODO: make sure the planning control horizon is greater than the
        # simulator_joystick_update_ratio else it will not plan far enough

        # LQR feedback control loop
        self.commands = Trajectory.new_traj_clip_along_time_axis(
            self.planner_data["trajectory"],
            self.agent_params.control_horizon,
            repeat_second_to_last_speed=True,
        )

    def joystick_act(self):
        if self.joystick_on:
            num_cmds_per_step = self.simulator_joystick_update_ratio
            # runs through the entire planned horizon just with a cmds_step of the above
            num_steps = int(np.floor(self.commands.k / num_cmds_per_step))
            for j in range(num_steps):
                xytv_cmds = []
                for i in range(num_cmds_per_step):
                    idx = j * num_cmds_per_step + i
                    (x, y, th, v) = self.from_conf(self.commands, idx)
                    xytv_cmds.append((x, y, th, v))
                self.send_cmds(xytv_cmds, send_vel_cmds=False)

                # break if the robot finished
                if not self.joystick_on:
                    break
