from random import randint
from typing import List, Tuple

import numpy as np

from joystick_py.joystick_base import JoystickBase


class JoystickRandom(JoystickBase):
    def __init__(self):
        # our 'positions' are modeled as (x, y, theta)
        self.robot_posn: np.ndarray = None  # current position of the robot
        super().__init__("RandomPlanner")  # parent class needs to know the algorithm

    def joystick_sense(self) -> None:
        # ping's the robot to request a sim state
        self.send_to_robot("sense")
        # listen to the robot's reply
        self.joystick_on = self.listen_once()
        # NOTE: self.sim_state_now is updated with the current world state
        # can get agent/robot position info from it, see simulators/sim_state.py
        self.robot_posn = (
            self.sim_state_now.get_robot()
            .get_current_config()
            .position_and_heading_nk3(squeeze=True)
        )

    def joystick_plan(self) -> None:
        # use the robot's current position from the newly updated self.sim_state_now
        # to plan random commands within viable range of the robot's system dynamics

        # frequency of actions per joystick refresh
        num_actions_per_dt: int = int(np.floor(self.sim_dt / self.joystick_params.dt))

        # send either posntional or velocity commands depending on param status
        if self.joystick_params.use_system_dynamics:
            self.input = self.random_vel_cmds(num_actions_per_dt)
        else:
            self.input = self.random_posn_cmds(num_actions_per_dt, self.robot_posn)

    def joystick_act(self) -> None:
        if not self.joystick_on:
            return
        # send random commands to the robot
        self.send_cmds(
            self.input, send_vel_cmds=self.joystick_params.use_system_dynamics
        )

    def update_loop(self) -> None:
        super().pre_update()  # pre-update initialization
        while self.joystick_on:
            # gather information about the world state based off the simulator
            self.joystick_sense()

            # create a plan for the next steps of the trajectory
            self.joystick_plan()

            # send a command to the robot
            self.joystick_act()

        self.finish_episode()

    """BEGIN RANDOM COMMAND FUNCTIONS"""

    def random_cmd(self, bounds: Tuple[float, float], precision: int = 3) -> int:
        return (
            randint(int(bounds[0] * precision), int(bounds[1] * precision)) / precision
        )

    def random_vel_cmds(self, freq: int) -> List[float]:
        velocity_cmds: List[float] = []
        for _ in range(freq):
            # add a random linear velocity command to send
            v = self.random_cmd(self.system_dynamics_params.v_bounds)
            # also add a random angular velocity command
            w = self.random_cmd(self.system_dynamics_params.w_bounds)
            velocity_cmds.append((v, w))
        # send the data in lists based off the simulator/joystick refresh rate
        return velocity_cmds

    def random_posn_cmds(
        self, freq: int, current_posn: List[float]
    ) -> List[Tuple[float, float, float, float]]:
        # generate a random position within range of viable velocity
        new_posns: List[Tuple[float, float, float, float]] = []
        for _ in range(freq):
            rand_vel = self.random_cmd(self.system_dynamics_params.v_bounds)
            rand_theta = self.random_cmd([-3.1415, 3.1415])  # full 360 degrees
            scaled_vel = self.joystick_params.dt * rand_vel  # accounting for dt
            new_x = scaled_vel * np.cos(rand_theta) + current_posn[0]
            new_y = scaled_vel * np.sin(rand_theta) + current_posn[1]
            new_posns.append((new_x, new_y, rand_theta, rand_vel))
        return new_posns
