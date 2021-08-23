from typing import Dict, List

import numpy as np
from agents.agent import Agent


class Episode:
    def __init__(
        self,
        name: str,
        environment: Dict[str, float or int or np.ndarray],
        agents: Dict[str, Agent],
        t_budget: float,
        r_start: List[float],
        r_goal: List[float],
    ):
        self.name = name
        self.environment = environment
        self.agents = agents
        self.time_budget = t_budget
        self.robot_start = r_start  # starting position of the robot
        self.robot_goal = r_goal  # goal position of the robot

    def get_name(self) -> str:
        return self.name

    def get_environment(self) -> Dict[str, float or int or np.ndarray]:
        return self.environment

    def get_agents(self) -> Dict[str, Agent]:
        return self.agents

    def get_time_budget(self) -> float:
        return self.time_budget

    def update(self, env, agents) -> None:
        if not (not env):  # only upate env if non-empty
            self.environment = env
        # update dict of agents
        self.agents = agents

    def get_robot_start(self) -> List[float]:
        return self.robot_start

    def get_robot_goal(self) -> List[float]:
        return self.robot_goal
