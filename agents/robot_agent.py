from utils.utils import *
from agents.agent import Agent
from agents.robot_utils import *
from trajectory.trajectory import SystemConfig
from params.central_params import create_robot_params
import numpy as np
import time


class RobotAgent(Agent):
    def __init__(self, name, start_configs):
        self.name = name
        super().__init__(start_configs.get_start_config(),
                         start_configs.get_goal_config(),
                         name)
        self.joystick_inputs = []
        # josystick is ready once it has been sent an environment
        self.joystick_ready = False
        # To send the world state on the next joystick ping
        self.joystick_requests_world = -1
        # whether or not to repeat the last joystick input
        self.block_joystick = False  # gets updated in Simulator
        # told the joystick that the robot is powered off
        self.notified_joystick = False
        # amount of time the robot is blocking on the joystick
        self.block_time_total = 0
        # robot initially has no knowledge of the planning algorithm
        # this is (optionally) sent by the joystick
        self.algo_name = "UnknownAlgo"

    def simulation_init(self, sim_map, with_planner=False, keep_episode_running=False):
        # first initialize all the agent fields such as basic self.params
        super().simulation_init(sim_map,
                                with_planner=with_planner,
                                with_system_dynamics=True,
                                with_objectives=True,
                                keep_episode_running=keep_episode_running)
        # this robot agent does not have a "planner" since that is done through the joystick
        self.params.robot_params = create_robot_params()
        # NOTE: robot radius is not the same as regular Agents
        self.radius = self.params.robot_params.physical_params.radius
        # velocity bounds when teleporting to positions (if not using sys dynamics)
        self.v_bounds = self.params.system_dynamics_params.v_bounds
        self.w_bounds = self.params.system_dynamics_params.w_bounds
        # simulation update init
        self.num_executed = 0  # keeps track of the latest command that is to be executed
        # number of commands the joystick sends at once
        self.num_cmds_per_batch = 1
        # maximum number of times that the robot will repeat the last command if in asynch-mode
        self.remaining_repeats = self.params.robot_params.max_repeats

    def get_num_executed(self):
        return int(np.floor(len(self.joystick_inputs) / self.num_cmds_per_batch))

    def get_block_t_total(self):
        return self.block_time_total

    @staticmethod
    def generate_robot(configs, name=None, verbose=False):
        """
        Sample a new random robot agent from all required features
        """
        robot_name = "robot_agent"  # constant name for the robot since there will only ever be one
        # In order to print more readable arrays
        np.set_printoptions(precision=2)
        pos_2 = configs.get_start_config().to_3D_numpy()
        goal_2 = configs.get_goal_config().to_3D_numpy()
        if verbose:
            print("Robot", robot_name, "at", pos_2, "with goal", goal_2)
        return RobotAgent(robot_name, configs)

    @staticmethod
    def random_from_environment(environment):
        """
        Sample a new robot without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        from agents.humans.human_configs import HumanConfigs
        configs = HumanConfigs.generate_random_human_config(environment)
        return RobotAgent.generate_robot(configs)

    def check_termination_conditions(self):
        """use this to take in a world state and compute obstacles 
        (gen_agents/walls) to affect the robot"""
        # check for collisions with other gen_agents
        self.check_collisions(self.world_state)

        # enforce planning termination upon condition
        self.enforce_termination_conditions()

        if self.get_trajectory().k >= self.collision_point_k:
            self.end_acting = True

        if self.get_end_acting():
            self.power_off()

    def execute(self):
        self.check_termination_conditions()
        if(self.params.robot_params.use_system_dynamics):
            self.execute_velocity_cmds()
        else:
            self.execute_position_cmds()
        if (self.params.verbose):
            print(self.get_current_config().to_3D_numpy())
        # knowing that both executions took self.num_cmds_per_batch commands
        self.num_executed += self.num_cmds_per_batch

    def execute_velocity_cmds(self):
        for _ in range(self.num_cmds_per_batch):
            if(self.get_end_acting()):
                break
            current_config = self.get_current_config()
            # the command is indexed by self.num_executed and is safe due to the size constraints in the update()
            vel_cmd = self.joystick_inputs[self.num_executed]
            assert(len(vel_cmd) == 2)  # always a 2 tuple of v and w
            v = clip_vel(vel_cmd[0], self.v_bounds)
            w = clip_vel(vel_cmd[1], self.w_bounds)
            # NOTE: the format for the acceleration commands to the open loop for the robot is:
            # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
            command = np.array([[[v, w]]], dtype=np.float32)
            t_seg, _ = Agent.apply_control_open_loop(self, current_config,
                                                     command, 1,
                                                     sim_mode='ideal'
                                                     )
            self.trajectory.append_along_time_axis(
                t_seg, track_trajectory_acceleration=True)
            # act trajectory segment
            self.current_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    t_seg, t=-1)

    def execute_position_cmds(self):
        for _ in range(self.num_cmds_per_batch):
            if(self.get_end_acting()):
                break
            joystick_input = self.joystick_inputs[self.num_executed]
            assert(len(joystick_input) == 4)  # has x,y,theta,velocity
            new_pos3 = joystick_input[:3]
            new_v = joystick_input[3]
            old_pos3 = self.current_config.to_3D_numpy()
            # ensure the new position is reachable within velocity bounds
            new_pos3 = clip_posn(Agent.sim_dt, old_pos3,
                                 new_pos3, self.v_bounds)
            # move to the new position and update trajectory
            new_config = generate_config_from_pos_3(new_pos3, v=new_v)
            self.set_current_config(new_config)
            self.trajectory.append_along_time_axis(new_config,
                                                   track_trajectory_acceleration=True)

    def sense(self):
        # send a sim_state if it was requested by the joystick
        # self.joystick_requests_world is a 'countdown' where 0 => send sim_state
        # and -1 => do nothing (until receives command asking for one, else countdown
        if self.joystick_requests_world == 0:
            # has processed all prior commands
            send_sim_state(self)
        if self.block_joystick:
            # block simulation (world) progression on the act() commands sent from the joystick
            init_block_t = time.time()
            while not self.get_end_acting() and self.num_executed >= len(self.joystick_inputs):
                if self.num_executed == len(self.joystick_inputs):
                    if self.joystick_requests_world == 0:
                        send_sim_state(self)
                time.sleep(0.001)
            # capture how much time was spent blocking on joystick inputs
            self.block_time_total += time.time() - init_block_t

    def plan(self):
        # recall the planning is being done with YOUR social nagivation algorithm
        # and is being received through the joystick
        pass

    def act(self):
        # execute the next command in the queue
        num_cmds = len(self.joystick_inputs)
        if self.num_executed < num_cmds:
            # execute all the commands on the queue
            self.execute()
            # decrement counter
            if(self.joystick_requests_world > 0):
                self.joystick_requests_world -= 1
        elif not self.block_joystick and self.remaining_repeats > 0:
            # repeat the last n commands in the queue if running asynchronously
            # only if there is at least n>0 available commands to repeat
            if(num_cmds < 1):
                return
            repeats = self.joystick_inputs[-1:]
            self.joystick_inputs.extend(repeats)
            self.execute()
            # decrement counter
            if(self.joystick_requests_world > 0):
                self.joystick_requests_world -= 1
            # just executed one command, decrease from the counter
            self.remaining_repeats -= 1

    def update(self):
        if(self.get_end_acting()):
            return
        self.sense()
        self.plan()
        self.act()

    def power_off(self):
        # if the robot is already "off" do nothing
        print("\nRobot powering off, received",
              len(self.joystick_inputs), "commands")
        self.end_acting = True
        try:
            quit_message = self.world_state.to_json(
                robot_on=False,
                termination_cause=self.termination_cause
            )
            send_to_joystick(quit_message)
        except:
            return

    def listen_to_joystick(self):
        # send initial world state (specific episode metadata)
        send_to_joystick(self.world_state.to_json(send_metadata=True))
        while not self.get_end_acting():
            listen_once(self)

    def force_connect_self(self):
        force_connect()

    @staticmethod
    def establish_joystick_handshake(p):
        establish_handshake(p)

    @staticmethod
    def close_robot_sockets():
        close_sockets()
