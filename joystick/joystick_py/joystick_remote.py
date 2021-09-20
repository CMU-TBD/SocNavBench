import socket
from time import sleep
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from trajectory.trajectory import SystemConfig

from joystick_py.joystick_base import JoystickBase
from utils.utils import color_text


class JoystickRemote(JoystickBase):
    """This is the base class for joystick implementations that are implemented with 
    message passing across local sockets. Eg. RVO, social-forces"""

    def __init__(
        self,
        algo_name: Optional[str] = "remote",
        HOST: Optional[str] = "127.0.0.1",
        PORT: Optional[int] = 2111,
    ):
        super().__init__(algo_name)
        self.agents = None
        self.agent_radius = None
        self.robot = None
        self.connect_to_remote_planner(algo_name=algo_name, HOST=HOST, PORT=PORT)

    def connect_to_remote_planner(self, algo_name: str, HOST: str, PORT: int) -> None:
        self.remote_planner_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print(
            'Waiting for {} executable at "{}:{}"...'.format(algo_name, HOST, PORT),
            end="",
            flush=True,
        )
        while True:
            try:
                self.remote_planner_sock.connect((HOST, PORT))
                print()
                break
            except ConnectionRefusedError or TimeoutError:
                sleep(1)
                print(".", end="", flush=True)
        print(
            "{}Connected to {} planner!{}".format(
                color_text["green"], algo_name, color_text["reset"]
            )
        )

    def init_obstacle_map(
        self, env: Dict[str, Any]
    ) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        scale = float(env["map_scale"])
        map_trav = env["map_traversible"]
        edges_raw = set()
        for i in range(len(map_trav)):
            for j in range(len(map_trav[0])):
                if map_trav[i][j] == False:
                    edges_cand = [
                        ((i, j), (i + 1, j)),
                        ((i, j), (i, j + 1)),
                        ((i + 1, j), (i + 1, j + 1)),
                        ((i, j + 1), (i + 1, j + 1)),
                    ]
                    for ed in edges_cand:
                        if ed in edges_raw:
                            edges_raw.remove(ed)
                        else:
                            edges_raw.add(ed)

        edges_set = set()
        while len(edges_raw) > 0:
            ed = edges_raw.pop()
            if (ed[1][0] - ed[0][0]) == 1:
                dir_vector = (1, 0)
            elif (ed[1][1] - ed[0][1]) == 1:
                dir_vector = (0, 1)
            else:
                raise Exception("Edge direction determination error!")
            end_pt1 = ed[0]
            end_pt2 = ed[1]
            end_pt = ed[0]
            end_pt_nxt = (end_pt[0] - dir_vector[0], end_pt[1] - dir_vector[1])
            while (end_pt_nxt, end_pt) in edges_raw:
                edges_raw.remove((end_pt_nxt, end_pt))
                end_pt1 = end_pt_nxt
                end_pt = end_pt_nxt
                end_pt_nxt = (end_pt[0] - dir_vector[0], end_pt[1] - dir_vector[1])
            end_pt = ed[1]
            end_pt_nxt = (end_pt[0] + dir_vector[0], end_pt[1] + dir_vector[1])
            while (end_pt, end_pt_nxt) in edges_raw:
                edges_raw.remove((end_pt, end_pt_nxt))
                end_pt2 = end_pt_nxt
                end_pt = end_pt_nxt
                end_pt_nxt = (end_pt[0] + dir_vector[0], end_pt[1] + dir_vector[1])
            end_pt1 = (end_pt1[0] * scale, end_pt1[1] * scale)
            end_pt2 = (end_pt2[0] * scale, end_pt2[1] * scale)
            edges_set.add((end_pt1, end_pt2))

        return edges_set

    def init_control_pipeline(self) -> None:
        self.goal_config = SystemConfig.from_pos3(self.get_robot_goal())

        env = self.current_ep.get_environment()
        self.environment = self.init_obstacle_map(env)
        self.robot = self.get_robot_start()
        self.agents = {}
        agents_info = self.current_ep.get_agents()
        for key in list(agents_info.keys()):
            agent = agents_info[key]
            self.agents[key] = np.squeeze(
                agent.get_current_config().position_and_heading_nk3()
            )
        self.commands = None

    def convert_to_string(self) -> str:
        # to be implemented in the children
        raise NotImplementedError

    def send_info_to_planner(self) -> None:
        info_string = self.convert_to_string()
        self.remote_planner_sock.sendall(info_string.encode("utf-8"))

    def joystick_sense(self) -> None:
        self.send_to_robot("sense")
        if not self.listen_once():
            self.joystick_on = False

        robot_prev = self.robot.copy()
        agents_prev = {}
        for key in list(self.agents.keys()):
            agent = self.agents[key]
            agents_prev[key] = agent.copy()

        # NOTE: self.sim_dt is available
        self.agents = {}
        self.agents_radius = {}
        agents_info = self.sim_state_now.get_all_agents()
        for key in list(agents_info.keys()):
            agent = agents_info[key]
            self.agents[key] = np.squeeze(
                agent.get_current_config().position_and_heading_nk3()
            )
            self.agents_radius[key] = agent.get_radius()
        robot_tmp = list(self.sim_state_now.get_robots().values())[0]
        self.robot = np.squeeze(
            robot_tmp.get_current_config().position_and_heading_nk3()
        )
        self.robot_radius = robot_tmp.get_radius()

        self.robot_v = (self.robot - robot_prev) / self.sim_dt
        self.agents_v = {}
        for key in list(self.agents.keys()):
            if key in agents_prev:
                v = (self.agents[key] - agents_prev[key]) / self.sim_dt
            else:
                v = np.array([0, 0, 0], dtype=np.float32)
            self.agents_v[key] = v

    def joystick_plan(self) -> None:
        horizon_scale = 10
        horizon = self.sim_dt * horizon_scale
        self.agents_goals = {}
        for key in list(self.agents.keys()):
            goal = self.agents[key] + self.agents_v[key] * horizon
            self.agents_goals[key] = goal

        self.send_info_to_planner()
        self.data = self.remote_planner_sock.recv(1024)

    def joystick_act(self) -> None:
        data_b = self.data.decode("utf-8")
        coordinate_str = data_b.split(",")
        x = float(coordinate_str[0])
        y = float(coordinate_str[1])
        print(
            "R:({:.3f}, {:.3f}, {:.3f}), xy:({:.3f}, {:.3f}), @ t:{:.3f}s".format(
                self.robot[0],
                self.robot[1],
                self.robot[2],
                x,
                y,
                self.sim_state_now.sim_t,
            )
        )
        th = np.arctan2(y - self.robot[1], x - self.robot[0])
        if self.joystick_on:
            self.send_cmds([(x, y, th, 0)], send_vel_cmds=False)

    def update_loop(self) -> None:
        self.robot_receiver_socket.listen(1)
        self.joystick_on = True
        while self.joystick_on:
            self.joystick_sense()
            self.joystick_plan()
            self.joystick_act()
        self.remote_planner_sock.sendall(b"OFF")
        self.finish_episode()

