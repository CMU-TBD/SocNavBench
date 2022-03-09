import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from agents.agent import Agent
from agents.humans.human import Human, HumanAppearance
from agents.robot_agent import RobotAgent
from dotmap import DotMap
from matplotlib import pyplot, lines
from params.central_params import get_path_to_socnav
from trajectory.trajectory import SystemConfig, Trajectory
from utils.utils import (
    color_text,
    euclidean_dist2,
    mkdir_if_missing,
    to_json_type,
    touch,
)

""" These are smaller "wrapper" classes that are visible by other
gen_agents/humans and saved during state deepcopies
NOTE: they are all READ-ONLY (only getters)
"""


class AgentState:
    def __init__(
        self,
        name: str,
        goal_config: SystemConfig,
        start_config: SystemConfig,
        current_config: SystemConfig,
        trajectory: Optional[Trajectory] = None,
        appearance: Optional[HumanAppearance] = None,
        collided: bool = False,
        end_acting: bool = False,
        collision_cooldown: int = -1,
        last_collision_t: Optional[float] = None,
        radius: int = 0,
        color: str = None,
    ):
        """Initialize an AgentState with either an Agent instance (a) or all the individual fields"""
        self.name: str = name
        self.goal_config: SystemConfig = goal_config
        self.start_config: SystemConfig = start_config
        self.current_config: SystemConfig = current_config
        self.trajectory: Trajectory = trajectory
        self.appearance: HumanAppearance = appearance
        self.collided: bool = collided
        self.end_acting: bool = end_acting
        self.collision_cooldown: int = collision_cooldown
        self.last_collision_t: float = last_collision_t
        self.radius: float = radius
        self.color: str = color

    @classmethod
    def from_agent(cls, a: Agent):
        appearance = None
        if isinstance(a, Human):  # only Humans have appearances
            appearance = a.get_appearance()
        return cls(
            name=a.get_name(),
            goal_config=a.get_goal_config(),
            start_config=a.get_start_config(),
            current_config=a.get_current_config(),
            trajectory=a.get_trajectory(deepcpy=True),
            appearance=appearance,
            collided=a.get_collided(),
            end_acting=a.get_end_acting(),
            collision_cooldown=a.get_collision_cooldown(),
            last_collision_t=a.last_collision_t,
            radius=a.get_radius(),
            color=a.get_color(),
        )

    def get_name(self) -> str:
        return self.name

    def get_current_config(self) -> SystemConfig:
        return self.current_config

    def get_start_config(self) -> SystemConfig:
        return self.start_config

    def get_goal_config(self) -> SystemConfig:
        return self.goal_config

    def get_trajectory(self) -> Optional[Trajectory]:
        return self.trajectory

    def get_appearance(self) -> HumanAppearance:
        return self.appearance

    def get_collided(self) -> bool:
        return self.collided

    def get_radius(self) -> float:
        return self.radius

    def get_color(self) -> str:
        return self.color

    def get_collision_cooldown(self) -> bool:
        return self.collision_cooldown

    def get_pos3(self) -> np.ndarray:
        return self.get_current_config().position_and_heading_nk3(squeeze=True)

    def render(self, ax: pyplot.Axes, p: DotMap) -> None:
        x, y, th = self.current_config.position_and_heading_nk3(squeeze=True)
        traj_mpl_kwargs: Dict[str, Any] = p.traj_mpl_kwargs
        if traj_mpl_kwargs["color"] is None:
            # overwrite the colour with the agent's colour. Does not affect p.traj_mpl_kwargs
            traj_mpl_kwargs = dict(traj_mpl_kwargs, **{"color": self.color})
        # only load the start/goal configs if they are present in the agent
        start_pos3 = None
        goal_pos3 = None
        if self.start_config is not None:
            start_pos3 = self.start_config.position_and_heading_nk3(squeeze=True)
            start_x, start_y, start_th = start_pos3
        if self.goal_config is not None:
            goal_pos3 = self.goal_config.position_and_heading_nk3(squeeze=True)
            goal_x, goal_y, goal_th = goal_pos3

        # draw trajectory
        if p.plot_trajectory and self.trajectory is not None:
            self.trajectory.render(
                ax,
                freq=p.traj_freq,
                plot_quiver=False,
                clip=p.max_traj_length,
                mpl_kwargs=traj_mpl_kwargs,
            )

        # draw agent body
        ax.plot(x, y, **p.body_normal_mpl_kwargs)

        # make the agent change colour when collided (only if specified in params)
        if p.body_collision_mpl_kwargs is not None and (
            self.collided or self.collision_cooldown > 0
        ):
            ax.plot(x, y, **p.body_collision_mpl_kwargs)
            ax.plot(x, y, **p.collision_mini_dot_mpl_kwargs)

        # draw start config
        if p.plot_start and start_pos3 is not None:
            ax.plot(start_x, start_y, **p.start_mpl_kwargs)

        # draw goal config
        if p.plot_goal and goal_pos3 is not None:
            ax.plot(goal_x, goal_y, **p.goal_mpl_kwargs)

        # draw quiver (heading arrow)
        if p.plot_quiver:

            s = 0.5  # scale

            def plot_quiver(xpos: float, ypos: float, theta: float) -> None:
                # TODO: add to kwargs params
                ax.quiver(
                    xpos,
                    ypos,
                    s * np.cos(theta),
                    s * np.sin(theta),
                    scale=1,
                    scale_units="xy",
                    zorder=1,  # behind the agent body
                )

            plot_quiver(x, y, th)  # plot agent body quiver

            # plot start quiver
            if p.plot_start and start_pos3 is not None:
                plot_quiver(start_x, start_y, start_th)

            # plot goal quiver
            if p.plot_goal and goal_pos3 is not None:
                plot_quiver(goal_x, goal_y, goal_th)

    def to_json_type(
        self, omit_fields: Optional[List[str]] = ["trajectory",],
    ) -> Dict[str, str]:
        json_dict: Dict[str, str] = {}
        json_dict["name"] = self.name  # always keep name

        def to_json_if_not_omitted(key: str, field: Any) -> None:
            if key not in omit_fields:
                json_dict[key] = to_json_type(field)

        to_json_if_not_omitted(
            "start_config",
            self.get_start_config().position_and_heading_nk3(squeeze=True),
        )
        to_json_if_not_omitted(
            "goal_config",
            self.get_goal_config().position_and_heading_nk3(squeeze=True),
        )
        to_json_if_not_omitted(
            "current_config",
            self.get_current_config().position_and_heading_nk3(squeeze=True),
        )
        to_json_if_not_omitted(
            "trajectory", self.get_trajectory().position_and_heading_nk3()
        )
        to_json_if_not_omitted("radius", self.radius)
        # no need to store 'collided' since collision cooldown contains more data
        to_json_if_not_omitted("collision_cooldown", self.collision_cooldown)
        to_json_if_not_omitted("last_collision_t", self.last_collision_t)
        return json_dict

    @classmethod
    def from_json(cls, json_dict: Dict[str, str]):
        assert "name" in json_dict
        name: str = json_dict["name"]
        start_config = (
            SystemConfig.from_pos3(json_dict["start_config"])
            if "start_config" in json_dict
            else None
        )
        goal_config = (
            SystemConfig.from_pos3(json_dict["goal_config"])
            if "start_config" in json_dict
            else None
        )
        current_config = (
            SystemConfig.from_pos3(json_dict["current_config"])
            if "current_config" in json_dict
            else None
        )
        radius = json_dict["radius"] if "radius" in json_dict else None
        coll_cooldown = (
            json_dict["collision_cooldown"]
            if "collision_cooldown" in json_dict
            else None
        )
        last_collision_t = (
            json_dict["last_collision_t"] if "last_collision_t" in json_dict else None
        )
        trajectory = None
        if "trajectory" in json_dict:
            np_repr: np.ndarray = np.array(json_dict["trajectory"])
            n, k, d = np_repr.shape
            assert n == 1 and d == 3  # 3 dimensional (pos3) and only 1 trajectory
            trajectory = Trajectory.from_pos3_array(np_repr)
        return cls(
            name=name,
            goal_config=goal_config,
            start_config=start_config,
            current_config=current_config,
            trajectory=trajectory,
            collided=bool(coll_cooldown > 0) if coll_cooldown is not None else False,
            end_acting=False,
            collision_cooldown=coll_cooldown,
            last_collision_t=last_collision_t,
            radius=radius,
            color=None,
        )


class SimState:
    def __init__(
        self,
        environment: Optional[Dict[str, float or int or np.ndarray]] = None,
        pedestrians: Optional[Dict[str, Agent]] = None,
        robots: Optional[Dict[str, RobotAgent]] = None,
        sim_t: Optional[float] = None,
        wall_t: Optional[float] = None,
        delta_t: Optional[float] = None,
        robot_on: Optional[bool] = True,
        episode_name: Optional[str] = None,
        max_time: Optional[float] = None,
        ped_collider: Optional[str] = "",
    ):
        self.environment: Dict[str, float or int or np.ndarray] = environment
        # no distinction between prerecorded and auto agents
        # new dict that the joystick will be sent
        self.pedestrians: Dict[str, AgentState] = pedestrians
        self.robots: Dict[str, AgentState] = robots
        self.sim_t: float = sim_t
        self.wall_t: float = wall_t
        self.delta_t: float = delta_t
        self.robot_on: bool = robot_on
        self.episode_name: str = episode_name
        self.episode_max_time: float = max_time
        self.ped_collider: str = ped_collider

    def get_environment(self) -> Dict[str, float or int or np.ndarray]:
        return self.environment

    def get_map(self) -> np.ndarray:
        return self.environment["map_traversible"]

    def get_pedestrians(self) -> Dict[str, Agent]:
        return self.pedestrians

    def get_robots(self) -> Dict[str, RobotAgent]:
        return self.robots

    def get_robot(
        self, index: Optional[int] = 0, name: Optional[str] = None
    ) -> RobotAgent:
        if name:  # index robot by name
            return self.robots[name]
        return list(self.robots.values())[index]  # index robot by posn

    def get_sim_t(self) -> float:
        return self.sim_t

    def get_wall_t(self) -> float:
        return self.wall_t

    def get_delta_t(self) -> float:
        return self.delta_t

    def get_robot_on(self) -> bool:
        return self.robot_on

    def get_episode_name(self) -> str:
        return self.episode_name

    def get_episode_max_time(self) -> float:
        return self.episode_max_time

    def get_collider(self) -> str:
        return self.ped_collider

    def get_all_agents(
        self, include_robot: Optional[bool] = False
    ) -> Dict[str, Agent or RobotAgent]:
        all_agents = {}
        all_agents.update(self.get_pedestrians())
        if include_robot:
            all_agents.update(self.get_robots())
        return all_agents

    def to_json(
        self,
        robot_on: Optional[bool] = True,
        send_metadata: Optional[bool] = False,
        termination_cause: Optional[str] = None,
        full_export: Optional[bool] = False,
    ) -> str:
        json_dict: Dict[str, float or int or np.ndarray] = {}
        json_dict["robot_on"] = to_json_type(robot_on)
        if robot_on:  # only send the world if the robot is ON
            json_args = {"omit_fields": ["trajectory"]}  # don't save trajectory
            if full_export:
                json_args = {"omit_fields": [""]}  # don't omit anything, serialize all!
                # NOTE: could add other configs here
            elif not send_metadata:
                json_args["omit_fields"].extend(["start_config", "goal_config"])
            robots_json: dict = to_json_type(self.get_robots(), json_args)
            if send_metadata:
                # NOTE: the robot(s) send their start/goal posn iff sending metadata
                # only include environment and episode name iff sending metadata
                json_dict["environment"] = to_json_type(self.get_environment())
                json_dict["episode_name"] = to_json_type(self.get_episode_name())
            # append other fields to the json dictionary
            json_dict["pedestrians"] = to_json_type(self.get_pedestrians(), json_args)
            json_dict["robots"] = robots_json
            json_dict["delta_t"] = to_json_type(self.get_delta_t())
            json_dict["episode_max_time"] = to_json_type(self.get_episode_max_time())
        else:
            json_dict["termination_cause"] = to_json_type(termination_cause)
        # sim_state should always have time
        json_dict["sim_t"] = to_json_type(self.get_sim_t())
        return json.dumps(json_dict)

    def export_to_file(
        self, out_dir: Optional[str] = None, export_metadata: Optional[bool] = False
    ) -> None:
        json_repr: str = self.to_json(
            robot_on=True,
            send_metadata=export_metadata,
            termination_cause=None,
            full_export=True,
        )
        filename: str = "sim_state"
        if out_dir is not None:
            mkdir_if_missing(out_dir)
            filename = os.path.join(out_dir, filename)
        out_filename: str = "{}_{:.4f}.json".format(filename, self.sim_t)
        with open(out_filename, "w") as out_file:
            out_file.write(json_repr)

    @classmethod
    def from_json(cls, json_str: Dict[str, str or int or float]):
        def try_loading(key: str) -> Optional[str or int or float]:
            if key in json_str:
                return json_str[key]
            return None

        return cls(
            environment=try_loading("environment"),
            pedestrians=SimState.init_agent_dict(json_str["pedestrians"]),
            robots=SimState.init_agent_dict(json_str["robots"]),
            sim_t=json_str["sim_t"],
            wall_t=None,
            delta_t=json_str["delta_t"],
            robot_on=json_str["robot_on"],
            episode_name=try_loading("episode_name"),
            max_time=json_str["episode_max_time"],
            ped_collider="",
        )

    @staticmethod
    def init_agent_dict(
        json_str_dict: Dict[str, Dict[str, str or float or int or dict]]
    ) -> Dict[str, Dict[str, AgentState]]:
        agent_dict: Dict[str, AgentState] = {}
        for agent_name in json_str_dict.keys():
            agent_dict[agent_name] = AgentState.from_json(json_str_dict[agent_name])
        return agent_dict

    def render(self, ax: pyplot.Axes, p: DotMap) -> None:
        """NOTE: this only renders the topview (schematic mode)"""
        """for the rgb & depth views we use the SocNavRenderer"""
        # Compute the real_world extent (in meters) of the traversible
        map_scale = self.environment["map_scale"]
        traversible = self.environment["map_traversible"]
        if not isinstance(traversible, np.ndarray):
            traversible = np.array(traversible)
        ax.set_xlim(0.0, traversible.shape[1] * map_scale)
        ax.set_ylim(0.0, traversible.shape[0] * map_scale)
        human_traversible = None
        if "human_traversible" in self.environment and p.draw_human_traversibles:
            assert p.render_3D
            human_traversible = self.environment["human_traversible"]
        extent = (
            np.array([0.0, traversible.shape[1], 0.0, traversible.shape[0]]) * map_scale
        )
        # plot the map traversible
        ax.imshow(
            traversible, extent=extent, cmap="gray", vmin=-0.5, vmax=1.5, origin="lower"
        )

        if human_traversible is not None:  # plot human traversible
            # NOTE: the human radius is only available given the openGL human modeling
            # Plot the 5x5 meter human radius grid atop the environment traversible
            alphas = np.empty(np.shape(human_traversible))
            for y in range(human_traversible.shape[1]):
                for x in range(human_traversible.shape[0]):
                    alphas[x][y] = not (human_traversible[x][y])
            ax.imshow(
                human_traversible,
                extent=extent,
                cmap="autumn_r",
                vmin=-0.5,
                vmax=1.5,
                origin="lower",
                alpha=alphas,
            )
            # alphas = np.all(np.logical_not(human_traversible))

        for human in self.pedestrians.values():
            human.render(ax, p.human_render_params)

        if (
            not p.draw_parallel_robots
            or len(p.draw_parallel_robots_params_by_algo) == 0
        ):
            for robot in self.robots.values():
                robot.render(ax, p.robot_render_params)
        # plot a small tick in the bottom left corner of schematic showing
        # how long a real world meter would be in the simulator world
        if p.plot_meter_tick:
            # plot other useful informational visuals in the topview
            # such as the key to the length of a "meter" unit
            plot_line_loc = self.environment["room_center"][:2] * 0.65
            start = [0, 0] + plot_line_loc
            end = [1, 0] + plot_line_loc
            gather_xs = [start[0], end[0]]
            gather_ys = [start[1], end[1]]
            col = "k-"
            h = 0.1  # height of the "ticks" of the key
            ax.plot(gather_xs, gather_ys, col)  # main line
            ax.plot(
                [start[0], start[0]], [start[1] + h, start[1] - h], col
            )  # tick left
            ax.plot([end[0], end[0]], [end[1] + h, end[1] - h], col)  # tick right
            if p.plot_meter_quiver:
                ax.text(
                    0.5 * (start[0] + end[0]) - 0.2,
                    start[1] + 0.5,
                    "1m",
                    fontsize=14,
                    verticalalignment="top",
                )
        if len(self.robots) > 0 or len(self.pedestrians) > 0:
            self.draw_legend(ax, p)

    def draw_legend(self, ax: pyplot.Axes, p: DotMap) -> None:
        # ensure no duplicate labels occur
        handles, labels = ax.get_legend_handles_labels()
        legend_keys: Dict[str, pyplot.Line2D] = {}
        assert len(handles) == len(labels)
        for i in range(len(handles)):
            label: str = labels[i]
            handle: pyplot.Line2D = handles[i]
            handle.set_linestyle("")  # no line passing through
            if p.collided_robots != DotMap() and label in p.collided_robots:
                assert p.robot_render_params.collision_mini_dot_mpl_kwargs is not None
                mark_of_shame: pyplot.Line2D = lines.Line2D(
                    xdata=[0],  # not plotted, xdata & ydata just have to be non-empty
                    ydata=[0],
                    **p.robot_render_params.collision_mini_dot_mpl_kwargs,
                )
                mark_of_shame.set_linestyle("")
                handle: Tuple[pyplot.Line2D] = (handle, mark_of_shame)  # mark in legend
            legend_keys[label] = handle
        if len(legend_keys) == 0:
            return  # do nothing, no label to print
        legend_order: Dict[str, int] = {
            "Pedestrian": 0,
            "Robot Start": 1,
            "Robot Goal": 2,
            # anything else is arbitrary
        }
        # organize the keys in the dictionary by the legend_order if present
        lgnd = ax.legend(
            *zip(
                *sorted(
                    # ensure each item is a tuple(line2D, str)
                    [item[::-1] for item in legend_keys.items()],
                    key=lambda k: legend_order[k[1]]
                    if k[1] in legend_order
                    else max(legend_order.values()) + 1,
                )
            ),
            loc=p.legend_loc,
        )

    @staticmethod
    def get_common_env(p: DotMap) -> Dict[str, int]:
        assert p.draw_parallel_robots is True
        socnav_dir: str = os.path.join(get_path_to_socnav(), "tests", "socnav")
        common_env: Dict[str, Any] = None
        for algo in list(p.draw_parallel_robots_params_by_algo.keys()):
            new_dir = os.path.join(socnav_dir, f"test_{algo}", f"{p.test_name}")
            if not os.path.exists(new_dir):
                break
            # find sim_state_data directory
            new_file: str = os.path.join(new_dir, "sim_state_data")
            min_sim_t: float = 1e8
            for sim_file in glob(new_file + "/sim_state_*.json"):
                try:
                    # try to infer the sim_t from just the filename (much faster)
                    sim_t = float(
                        sim_file.split("/")[-1]  # last one
                        .replace("sim_state_", "")  # no prefix
                        .replace(".json", "")  # no suffix
                    )
                except Exception as e:
                    with open(sim_file, "r") as f:
                        sim_state = SimState.from_json(json.load(f))
                        sim_t = sim_state.sim_t
                if sim_t < min_sim_t:
                    min_sim_t = sim_t
            first_sim_state_path = os.path.join(
                new_file, "sim_state_{:.4f}.json".format(min_sim_t)
            )
            with open(first_sim_state_path, "r") as f:
                sim_state = SimState.from_json(json.load(f))
                common_env = sim_state.environment
            if common_env is not None:  # only need to parse once
                break
        return common_env

    @staticmethod
    def get_max_parallel_sim_states(p: DotMap) -> Dict[str, int]:
        assert p.draw_parallel_robots is True
        max_states_per_algo: Dict[str, int] = {}
        socnav_dir: str = os.path.join(get_path_to_socnav(), "tests", "socnav")
        for algo in list(p.draw_parallel_robots_params_by_algo.keys()):
            new_dir = os.path.join(socnav_dir, f"test_{algo}", f"{p.test_name}")
            if not os.path.exists(new_dir):
                break
            # find sim_state_data directory
            new_file: str = os.path.join(new_dir, "sim_state_data")
            max_sim_t: float = 0.0
            for sim_file in glob(new_file + "/sim_state_*.json"):
                try:
                    # try to infer the sim_t from just the filename (much faster)
                    sim_t = float(
                        sim_file.split("/")[-1]  # last one
                        .replace("sim_state_", "")  # no prefix
                        .replace(".json", "")  # no suffix
                    )
                except Exception as e:
                    with open(sim_file, "r") as f:
                        sim_state = SimState.from_json(json.load(f))
                        sim_t = sim_state.sim_t
                if sim_t > max_sim_t:
                    max_sim_t = sim_t
            max_states_per_algo[algo] = max_sim_t
        return max_states_per_algo

    @staticmethod
    def get_sim_file_path(algo: str, test_name: str, sim_t: float) -> str:
        socnav_dir: str = get_path_to_socnav()
        return os.path.join(
            socnav_dir,
            "tests",
            "socnav",
            f"test_{algo}",
            f"{test_name}",
            "sim_state_data",
            f"sim_state_{sim_t:.4f}.json",
        )

    @staticmethod
    def draw_variants(
        env: Dict[str, Any],
        sim_t: float,
        ax: pyplot.Axes,
        max_algo_times: Dict[str, float],
        p: DotMap,
    ) -> None:
        out_dir: str = p.output_directory
        p.collided_robots = []  # track which robots collided across sim-states
        for algo in list(p.draw_parallel_robots_params_by_algo.keys()):
            test_name: str = out_dir.split("/")[-1]  # get name of this test
            exp_sim_state_file = SimState.get_sim_file_path(algo, test_name, sim_t)
            if not os.path.exists(exp_sim_state_file):
                if algo in max_algo_times and sim_t > 0:
                    # file does not exist bc the robot already terminated
                    max_sim_t: float = max_algo_times[algo]  # get last time
                    exp_sim_state_file = SimState.get_sim_file_path(
                        algo, test_name, max_sim_t
                    )
                    assert os.path.exists(exp_sim_state_file)
                else:
                    continue  # just skip alltogether, file does not exist

            with open(exp_sim_state_file, "r") as f:
                sim_state = SimState.from_json(json.load(f))
                if algo == max(max_algo_times, key=max_algo_times.get):
                    sim_state.environment = env
                    sim_state.render(ax, p)  # render the sim_state of the slowest
                rob = sim_state.get_robot()
                algo_params: DotMap = p.draw_parallel_robots_params_by_algo[algo]
                rob.render(ax, algo_params)
                if p.draw_mark_of_shame and sim_t >= rob.last_collision_t:
                    assert p.robot_render_params.collision_mini_dot_mpl_kwargs
                    x, y, _ = np.squeeze(rob.current_config.position_and_heading_nk3())
                    ax.plot(x, y, **p.robot_render_params.collision_mini_dot_mpl_kwargs)
                    rob_label: str = algo_params["body_normal_mpl_kwargs"]["label"]
                    p.collided_robots.append(rob_label)  # track this robot as "shamed"
                for pedestrian in sim_state.pedestrians.values():
                    if (
                        pedestrian.collision_cooldown is not None
                        and pedestrian.collision_cooldown > 0
                    ):
                        pedestrian.render(ax, p.human_render_params)
                # draw legend last (after all other agents/robots)
                sim_state.draw_legend(ax, p)
        # reset tracking which robots collided on this frame
        p.collided_robots = []

    @staticmethod
    def render_multi_robot(
        env: Dict[str, Any],
        sim_t: float,
        p: DotMap,
        max_algo_times: Dict[str, float],
        filename: str,
    ) -> None:
        if sim_t == 0:
            return  # bug where the first frame is not exported, just skip
        img_size: float = 10 * p.render_params.img_scale
        fig, ax = pyplot.subplots(1, 1, figsize=(1 * img_size, img_size))
        ax.set_title("Multi-robot Schematic View. t={:.3f}".format(sim_t), fontsize=14)
        ax.set_aspect("equal")
        # draw the SimStates
        SimState.draw_variants(env, sim_t, ax, max_algo_times, p.render_params)
        # save the fig to file
        full_file_name = os.path.join(p.output_directory, filename)
        if not os.path.exists(full_file_name):
            touch(full_file_name)  # Just as the bash command
        pyplot.tight_layout()
        fig.savefig(full_file_name, bbox_inches="tight", pad_inches=0)
        pyplot.close()


"""BEGIN SimState utils"""


def get_all_agents(
    sim_state: Dict[float, SimState], include_robot: Optional[bool] = False
) -> Dict[str, Agent or RobotAgent]:
    all_agents = {}
    all_agents.update(get_agents_from_type(sim_state, "pedestrians"))
    if include_robot:
        all_agents.update(get_agents_from_type(sim_state, "robots"))
    return all_agents


def get_agents_from_type(sim_state: SimState, agent_type: str) -> Dict[str, Agent]:
    if callable(getattr(sim_state, "get_" + agent_type, None)):
        getter_agent_type = getattr(sim_state, "get_" + agent_type, None)
        return getter_agent_type()
    return {}  # empty dict


def compute_next_vel(
    sim_state_prev: SimState, sim_state_now: SimState, agent_name: str
) -> float:
    old_agent = sim_state_prev.get_all_agents()[agent_name]
    old_pos = old_agent.get_current_config().position_and_heading_nk3(squeeze=True)
    new_agent = sim_state_now.get_all_agents()[agent_name]
    new_pos = new_agent.get_current_config().position_and_heading_nk3(squeeze=True)
    # calculate distance over time
    delta_t = sim_state_now.get_sim_t() - sim_state_prev.get_sim_t()
    return euclidean_dist2(old_pos, new_pos) / delta_t


def compute_agent_state_velocity(
    sim_states: List[SimState], agent_name: str
) -> List[float]:
    if len(sim_states) > 1:  # need at least two to compute differences in positions
        if agent_name in get_all_agents(sim_states[-1]):
            agent_velocities = []
            for i in range(len(sim_states)):
                if i > 0:
                    prev_sim_s = sim_states[i - 1]
                    now_sim_s = sim_states[i]
                    speed = compute_next_vel(prev_sim_s, now_sim_s, agent_name)
                    agent_velocities.append(speed)
                else:
                    agent_velocities.append(0.0)  # initial velocity is 0
            return agent_velocities
        else:
            print(
                "%sAgent" % color_text["red"],
                agent_name,
                "is not in the SimStates%s" % color_text["reset"],
            )
    else:
        return []


def compute_agent_state_acceleration(
    sim_states: List[SimState],
    agent_name: str,
    velocities: Optional[List[float]] = None,
) -> List[float]:
    if len(sim_states) > 1:  # need at least two to compute differences in velocities
        # optionally compute velocities as well
        if velocities is None:
            velocities = compute_agent_state_velocity(sim_states, agent_name)
        if agent_name in get_all_agents(sim_states[-1]):
            agent_accels = []
            for i, this_vel in enumerate(velocities):
                if i > 0:
                    # compute delta_t between sim states
                    sim_st_now = sim_states[i]
                    sim_st_prev = sim_states[i - 1]
                    delta_t = sim_st_now.get_sim_t() - sim_st_prev.get_sim_t()
                    # compute delta_v between velocities
                    last_vel = velocities[i - 1]
                    # calculate speeds over time
                    accel = (this_vel - last_vel) / delta_t
                    agent_accels.append(accel)
                    if i == len(sim_states) - 1:
                        # last element gets no acceleration
                        break
            return agent_accels
        else:
            print(
                "%sAgent" % color_text["red"],
                agent_name,
                "is not in the SimStates%s" % color_text["reset"],
            )
    else:
        return []


def compute_all_velocities(sim_states: List[SimState]) -> Dict[str, float]:
    all_velocities = {}
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert isinstance(agent_name, str)  # keyed by name
        all_velocities[agent_name] = compute_agent_state_velocity(
            sim_states, agent_name
        )
    return all_velocities


def compute_all_accelerations(sim_states: List[SimState]) -> Dict[str, float]:
    all_accels = {}
    # TODO: add option of providing precomputed velocities list
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert isinstance(agent_name, str)  # keyed by name
        all_accels[agent_name] = compute_agent_state_acceleration(
            sim_states, agent_name
        )
    return all_accels
