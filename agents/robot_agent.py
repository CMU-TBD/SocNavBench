import json
import socket
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_robot_params
from trajectory.trajectory import SystemConfig
from utils.utils import conn_recv, generate_random_config

from agents.agent import Agent
from agents.robot_utils import (
    clip_posn,
    clip_vel,
    close_sockets,
    establish_handshake,
    force_connect,
    lock,
)


class RobotAgent(Agent):
    # TODO: add support for multiple robots
    # socket utils
    robot_receiver_socket: socket.socket = None
    robot_sender_socket: socket.socket = None
    robot_receiver_id: str = create_robot_params().recv_ID
    robot_sender_id: str = create_robot_params().send_ID

    def __init__(
        self, name: str, start_config: SystemConfig, goal_config: SystemConfig
    ):
        super().__init__(start_config, goal_config, name)
        # positional inputs are tuples of (x, y, theta, velocity)
        # acceleration inputs are tuples of (linear velocity, angular velocity)
        self.joystick_inputs: List[
            Tuple[float, float, float, float] or Tuple[float, float]
        ] = []
        # josystick is ready once it has been sent an environment
        self.joystick_ready: bool = False
        # To send the world state on the next joystick ping
        self.joystick_requests_world: int = -1
        # whether or not to repeat the last joystick input
        self.block_joystick: bool = False  # gets updated in Simulator
        # told the joystick that the robot is powered off
        self.notified_joystick: bool = False
        # amount of time the robot is blocking on the joystick
        self.block_time_total: float = 0
        # robot initially has no knowledge of the planning algorithm
        # this is (optionally) sent by the joystick
        self.algo_name: str = "UnknownAlgo"

    def simulation_init(
        self,
        sim_map: SBPDMap,
        with_planner: Optional[bool] = False,
        keep_episode_running: Optional[bool] = False,
    ) -> None:
        # first initialize all the agent fields such as basic self.params
        super().simulation_init(
            sim_map,
            with_planner=with_planner,
            with_system_dynamics=True,
            with_objectives=True,
            keep_episode_running=keep_episode_running,
        )
        # this robot agent does not have a "planner" since that is done through the joystick
        self.params.robot_params = create_robot_params()
        # NOTE: robot radius is not the same as regular Agents
        self.radius: float = self.params.robot_params.physical_params.radius
        # velocity bounds when teleporting to positions (if not using sys dynamics)
        self.v_bounds: Tuple[float, float] = self.params.system_dynamics_params.v_bounds
        self.w_bounds: Tuple[float, float] = self.params.system_dynamics_params.w_bounds
        # simulation update init
        self.num_executed: int = 0  # keeps track of the latest command that is to be executed
        # number of commands the joystick sends at once
        self.num_cmds_per_batch: int = 1
        # maximum number of times that the robot will repeat the last command if in asynch-mode
        self.remaining_repeats: int = self.params.robot_params.max_repeats

    def get_num_executed(self) -> int:
        return int(np.floor(len(self.joystick_inputs) / self.num_cmds_per_batch))

    def get_block_t_total(self) -> float:
        return self.block_time_total

    @classmethod
    def generate_robot(
        cls, start_goal: List[List[float]], verbose: Optional[bool] = False
    ):
        """
        Sample a new random robot agent from all required features
        """
        robot_name: str = "robot_agent"  # constant name for the robot since there will only ever be one
        start: SystemConfig = SystemConfig.from_pos3(start_goal[0])
        goal: SystemConfig = SystemConfig.from_pos3(start_goal[1])
        # In order to print more readable arrays
        np.set_printoptions(precision=2)
        if verbose:
            print("Robot", robot_name, "at", start, "with goal", goal)
        return cls(robot_name, start, goal)

    @classmethod
    def random_from_environment(
        cls, environment: Dict[str, float or int or np.ndarray]
    ):
        """
        Sample a new robot without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        start_goal: Tuple[SystemConfig, SystemConfig] = [
            generate_random_config(environment).position_and_heading_nk3(squeeze=True),
            generate_random_config(environment).position_and_heading_nk3(squeeze=True),
        ]
        return cls.generate_robot(start_goal, verbose=False)

    def check_termination_conditions(self) -> None:
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

    def execute(self) -> None:
        self.check_termination_conditions()
        if self.params.robot_params.use_system_dynamics:
            self.execute_velocity_cmds()
        else:
            self.execute_position_cmds()
        if self.params.verbose:
            print(self.get_current_config().position_and_heading_nk3(squeeze=True))
        # knowing that both executions took self.num_cmds_per_batch commands
        self.num_executed += self.num_cmds_per_batch

    def execute_velocity_cmds(self) -> None:
        # used when the robot executes acceleration commands and system dynamics carry it
        for _ in range(self.num_cmds_per_batch):
            if self.get_end_acting():
                break
            current_config = self.get_current_config()
            # the command is indexed by self.num_executed and is safe due to the size constraints in the update()
            vel_cmd = self.joystick_inputs[self.num_executed]
            assert len(vel_cmd) == 2  # always a 2 tuple of v and w
            v = clip_vel(vel_cmd[0], self.v_bounds)
            w = clip_vel(vel_cmd[1], self.w_bounds)
            # NOTE: the format for the acceleration commands to the open loop for the robot is:
            # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
            command = np.array([[[v, w]]], dtype=np.float32)
            t_seg, _ = Agent.apply_control_open_loop(
                self, current_config, command, 1, sim_mode="ideal"
            )
            self.trajectory.append_along_time_axis(
                t_seg, track_trajectory_acceleration=True
            )
            # act trajectory segment
            self.current_config = SystemConfig.init_config_from_trajectory_time_index(
                t_seg, idx=-1
            )

    def execute_position_cmds(self) -> None:
        # used when robot "teleports" to the next position
        for _ in range(self.num_cmds_per_batch):
            if self.get_end_acting():
                break
            joystick_input = self.joystick_inputs[self.num_executed]
            assert len(joystick_input) == 4  # has x,y,theta,velocity
            new_pos3 = joystick_input[:3]
            new_v = joystick_input[3]
            old_pos3 = self.current_config.position_and_heading_nk3(squeeze=True)
            # ensure the new position is reachable within velocity bounds
            new_pos3 = clip_posn(Agent.sim_dt, old_pos3, new_pos3, self.v_bounds)
            # move to the new position and update trajectory
            new_config = SystemConfig.from_pos3(new_pos3, v=new_v)
            self.set_current_config(new_config)
            self.trajectory.append_along_time_axis(
                new_config, track_trajectory_acceleration=True
            )

    def sense(self) -> None:
        # send a sim_state if it was requested by the joystick
        # self.joystick_requests_world is a 'countdown' where 0 => send sim_state
        # and -1 => do nothing (until receives command asking for one, else countdown
        if self.joystick_requests_world == 0:
            # has processed all prior commands
            self.send_sim_state()
        if self.block_joystick:
            # block simulation (world) progression on the act() commands sent from the joystick
            init_block_t = time.time()
            while not self.get_end_acting() and self.num_executed >= len(
                self.joystick_inputs
            ):
                if self.num_executed == len(self.joystick_inputs):
                    if self.joystick_requests_world == 0:
                        self.send_sim_state()
                time.sleep(0.001)
            # capture how much time was spent blocking on joystick inputs
            self.block_time_total += time.time() - init_block_t

    def plan(self) -> None:
        # recall the planning is being done with YOUR social nagivation algorithm
        # and is being received through the joystick
        pass

    def act(self) -> None:
        # execute the next command in the queue
        num_cmds = len(self.joystick_inputs)
        if self.num_executed < num_cmds:
            # execute all the commands on the queue
            self.execute()
            # decrement counter
            if self.joystick_requests_world > 0:
                self.joystick_requests_world -= 1
        elif not self.block_joystick and self.remaining_repeats > 0:
            # repeat the last n commands in the queue if running asynchronously
            # only if there is at least n>0 available commands to repeat
            if num_cmds < 1:
                return
            repeats = self.joystick_inputs[-1:]
            self.joystick_inputs.extend(repeats)
            self.execute()
            # decrement counter
            if self.joystick_requests_world > 0:
                self.joystick_requests_world -= 1
            # just executed one command, decrease from the counter
            self.remaining_repeats -= 1

    def update(self) -> None:
        if self.get_end_acting():
            return
        self.sense()
        self.plan()
        self.act()

    def power_off(self) -> None:
        # if the robot is already "off" do nothing
        print("\nRobot powering off, received", len(self.joystick_inputs), "commands")
        self.end_acting = True
        try:
            quit_message: str = self.world_state.to_json(
                robot_on=False, termination_cause=self.termination_cause
            )
            self.send_to_joystick(quit_message)
        except:
            return

    def listen_to_joystick(self) -> None:
        # send initial world state (specific episode metadata)
        self.send_to_joystick(self.world_state.to_json(send_metadata=True))
        while not self.get_end_acting():
            self.listen_once()

    """ BEGIN SOCKET UTILS """

    def send_sim_state(self) -> None:
        # send the (JSON serialized) world state per joystick's request
        if self.joystick_requests_world == 0:
            world_state: str = self.world_state.to_json(
                robot_on=not self.get_end_acting(),
                termination_cause=self.termination_cause,
            )
            self.send_to_joystick(world_state)
            # immediately note that the world has been sent:
            self.joystick_requests_world = -1

    def send_to_joystick(self, message: str) -> None:
        with lock:
            assert isinstance(message, str)
            # Create a TCP/IP socket
            RobotAgent.robot_sender_socket = socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            )
            # Connect the socket to the port where the server is listening
            try:
                RobotAgent.robot_sender_socket.connect(RobotAgent.robot_sender_id)
            except ConnectionRefusedError:
                # abort and dont send data
                return
            # Send data
            RobotAgent.robot_sender_socket.sendall(bytes(message, "utf-8"))
            RobotAgent.robot_sender_socket.close()

    def listen_once(self) -> None:
        """Constantly connects to the robot listener socket and receives information from the
        joystick about the input commands as well as the world requests
        """
        connection, _ = RobotAgent.robot_receiver_socket.accept()
        data_b, response_len = conn_recv(connection, buffr_amnt=128)
        # close connection to be reaccepted when the joystick sends data
        connection.close()
        if data_b is not b"" and response_len > 0:
            data_str = data_b.decode("utf-8")  # bytes to str
            if self.get_end_acting():
                self.joystick_requests_world = 0
            else:
                self.manage_data(data_str)

    def is_keyword(self, data_str: str) -> bool:
        # non json important keyword
        if data_str == "sense":
            self.joystick_requests_world = len(self.joystick_inputs) - (
                self.num_executed
            )
            return True
        elif data_str == "ready":
            self.joystick_ready = True
            return True
        elif "algo: " in data_str:
            self.algo_name = data_str[len("algo: ") :]
            return True
        elif data_str == "abandon":
            self.power_off()
            return True
        return False

    def manage_data(self, data_str: str) -> None:
        if not self.is_keyword(data_str):
            data = json.loads(data_str)
            joystick_input: list = data["j_input"]
            self.num_cmds_per_batch = len(joystick_input)
            # add input commands to queue to keep track of
            for i in range(self.num_cmds_per_batch):
                np_data = np.array(joystick_input[i], dtype=np.float32)
                self.joystick_inputs.append(np_data)

    def force_connect_self(self) -> None:
        force_connect(RobotAgent.robot_receiver_id)

    @staticmethod
    def establish_joystick_handshake(p: DotMap) -> None:
        if p.episode_params.without_robot:
            # lite-mode episode does not include a robot or joystick
            return
        socks = establish_handshake(
            p, RobotAgent.robot_sender_id, RobotAgent.robot_receiver_id
        )
        # assign the new sockets
        RobotAgent.robot_receiver_socket: socket.socket = socks[0]
        RobotAgent.robot_sender_socket: socket.socket = socks[1]

    @staticmethod
    def close_robot_sockets() -> None:
        close_sockets(
            [RobotAgent.robot_receiver_socket, RobotAgent.robot_sender_socket]
        )

