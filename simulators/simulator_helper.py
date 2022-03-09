import multiprocessing
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
from agents.agent import Agent
from agents.humans.recorded_human import PrerecordedHuman
from agents.robot_agent import RobotAgent
from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_simulator_params
from socnav.socnav_renderer import SocNavRenderer
from utils.image_utils import render_socnav, save_to_gif
from utils.utils import color_text

from simulators.sim_state import AgentState, SimState


class SimulatorHelper(object):
    def __init__(
        self,
        environment: Dict[str, float or int or np.ndarray],
        verbose: Optional[bool] = False,
    ):
        """ Initializer for the simulator helper
        Args:
            environment (dict): dictionary housing the obj map (bitmap) and more
            verbose (bool): flag to print debug prints
        """
        self.environment: Dict[str, float or int or np.ndarray] = environment
        self.params: DotMap = create_simulator_params(verbose)
        self.episode_params: DotMap = None
        # by default there is no robot (or algorithm)
        self.algo_name: Optional[str] = None
        self.obstacle_map: SBPDMap = None
        # keep track of all agents in dictionary with names as the key
        self.agents: Dict[str, Agent] = {}
        # keep track of all robots in dictionary with names as the key
        self.robots: Dict[str, RobotAgent] = {}
        # keep track of all prerecorded humans in a dictionary like the otherwise
        self.backstage_prerecs: Dict[str, PrerecordedHuman] = {}
        self.prerecs: Dict[str, PrerecordedHuman] = {}
        # keep a single (important) robot as a value
        self.robot: RobotAgent = None  # TODO: remove
        self.sim_states: Dict[int, SimState] = {}  # TODO: make a list
        self.wall_clock_time: float = 0
        self.sim_t: float = 0.0
        self.dt: float = self.params.dt  # will be updated in simulator based off dt
        # metadata of agents
        self.total_agents: int = 0
        self.num_collided_agents: int = 0
        self.num_completed_agents: int = 0
        self.num_timeout_agents: int = 0  # updated with (non-robot) add_agent
        # restart agent coloring on every instance of the simulator to be consistent across episodes
        Agent.restart_coloring()

    def add_agent(self, a: Agent) -> None:
        """Adds an agent member to the central simulator's pool of gen_agents
        NOTE: this function works for robots (RobotAgent), prerecorded gen_agents (PrerecordedHuman),
              and general gen_agents (Agent)
        Args:
            a (Agent/PrerecordedAgent/RobotAgent): The agent to be added to the simulator
        """
        assert self.obstacle_map is not None
        name: str = a.get_name()
        if isinstance(a, RobotAgent):
            # initialize the robot and add to simulator's known "robot" field
            a.simulation_init(
                sim_map=self.obstacle_map,
                with_planner=False,
                keep_episode_running=self.params.keep_episode_running,
            )
            self.robots[name] = a
            self.robot = a
        elif isinstance(a, PrerecordedHuman):
            # generic agent initializer but without a planner (already have trajectories)
            a.simulation_init(
                sim_map=self.obstacle_map,
                with_planner=False,
                with_system_dynamics=False,
                with_objectives=False,
                keep_episode_running=self.params.keep_episode_running,
            )
            # added to backstage prerecs which will add to self.prerecs when the time is right
            self.backstage_prerecs[name] = a
            self.num_timeout_agents += 1  # added one more non-robot agent
        else:
            # initialize agent and add to simulator
            a.simulation_init(
                sim_map=self.obstacle_map,
                with_planner=True,
                keep_episode_running=self.params.keep_episode_running,
            )
            self.agents[name] = a
            self.num_timeout_agents += 1  # added one more non-robot agent

    def exists_running_agent(self) -> bool:
        """Checks whether or not a generated agent is still running (acting)
        Returns:
            bool: True if there is at least one running agent, False otherwise
        """
        for a in self.agents.values():
            if not a.end_acting:  # if there is even just a single agent acting
                return True
        return False

    def exists_running_prerec(self) -> bool:
        """Checks whether or not a prerecorded agent is still running (acting)
        Returns:
            bool: True if there is at least one running prerec, False otherwise
        """
        # make sure there are still remaining pedestrians in the backstage
        return not (not self.backstage_prerecs)

    def init_obstacle_map(self, renderer: Optional[SocNavRenderer] = None) -> SBPDMap:
        """ Initializes the sbpd map."""
        p: DotMap = self.params.obstacle_map_params
        return p.obstacle_map(
            p,
            renderer,
            res=self.environment["map_scale"] * 100,
            map_trav=self.environment["map_traversible"],
        )

    def loop_condition(self) -> bool:
        raise NotImplementedError

    def simulate(self) -> None:
        raise NotImplementedError

    def print_sim_progress(self, rendered_frames: int) -> None:
        """prints an inline simulation progress message based off the current agent termination statuses
            TODO: account for agent<->agent collisions
        Args:
            rendered_frames (int): how many frames have been generated so far
        """
        print(
            "Agents:",
            self.total_agents,
            "%sSuccess:" % (color_text["green"]),
            self.num_completed_agents,
            "%sCollision:" % (color_text["red"]),
            self.num_collided_agents,
            "%sTimout:" % (color_text["blue"]),
            self.num_timeout_agents,
            "%sFrames:" % (color_text["reset"]),
            rendered_frames,
            "T=%.3fs" % (self.sim_t),
            "\r",
            end="",
        )

    def gather_robot_collisions(self, max_iter: int) -> List[str]:
        agent_collisions: List[str] = []
        for i in range(max_iter):
            collider = self.sim_states[i].get_collider()
            if collider != "":
                agent_collisions.append(collider)
        last_collider = self.robot.latest_collider
        if last_collider != "":
            agent_collisions.append(last_collider)
        return agent_collisions

    def loop_through_pedestrians(self, current_state: SimState) -> None:
        for a in list(self.agents.values()):
            if a.get_end_acting():
                if a.get_collided():
                    self.num_collided_agents += 1
                else:
                    self.num_completed_agents += 1
                self.num_timeout_agents -= 1  # decrement the timeout_agents counter
                del self.agents[a.get_name()]
                del a
            else:
                a.update(current_state)

        for a in list(self.backstage_prerecs.values()):
            if (not a.get_end_acting()) and (
                a.get_start_time() <= Agent.sim_t < a.get_end_time()
            ):
                # only add (or keep) agents in the time frame
                self.prerecs[a.get_name()] = a
                a.update(current_state)
                if self.robot is not None and a.just_collided_with_robot(self.robot):
                    self.num_collided_agents += 1  # add collisions with robot
            else:
                # remove agent since its not within the time frame or finished
                if a.get_name() in self.prerecs:
                    if a.get_end_acting() and a.get_collided():
                        self.num_collided_agents += 1
                    else:
                        self.num_completed_agents += 1
                    self.num_timeout_agents -= 1  # decrement the timeout_agents counter
                    self.prerecs.pop(a.get_name())
                    # also remove from back stage since they will no longer be used
                    del self.backstage_prerecs[a.get_name()]
                    del a

    def init_auto_agent_threads(
        self, current_state: SimState
    ) -> List[threading.Thread]:
        """Spawns a new agent thread for each agent (running or finished)
        Args:
            current_state (SimState): the most recent state of the world
        Returns:
            agent_threads (list): list of all spawned (not started) agent threads
        """
        agent_threads: List[threading.Thread] = []
        all_agents: List[Agent] = list(self.agents.values())
        for a in all_agents:
            if not a.end_acting:
                agent_threads.append(
                    threading.Thread(target=a.update, args=(current_state,))
                )
            else:
                if a.get_collided():
                    self.num_collided_agents += 1
                else:
                    self.num_completed_agents += 1
                del self.agents[a.get_name()]
                del a
        return agent_threads

    def init_prerec_agent_threads(
        self, current_state: SimState
    ) -> List[threading.Thread]:
        """Spawns a new prerec thread for each running prerecorded agent
        Args:
            current_state (SimState): the current state of the world
        Returns:
            prerec_threads (list): list of all spawned (not started) prerecorded agent threads
        """
        prerec_threads: List[threading.Thread] = []
        all_prerec_agents: List[PrerecordedHuman] = list(
            self.backstage_prerecs.values()
        )
        for a in all_prerec_agents:
            if (
                not a.end_acting
                and a.get_start_time() <= Agent.sim_t < a.get_end_time()
            ):
                # only add (or keep) agents in the time frame
                self.prerecs[a.get_name()] = a
                prerec_threads.append(
                    threading.Thread(target=a.update, args=(current_state,))
                )
            else:
                # remove agent since its not within the time frame or finished
                if a.get_name() in self.prerecs:
                    if a.get_collided():
                        self.num_collided_agents += 1
                    else:
                        self.num_completed_agents += 1
                    self.prerecs.pop(a.get_name())
                    # also remove from back stage since they will no longer be used
                    del self.backstage_prerecs[a.get_name()]
                    del a
        return prerec_threads

    def start_threads(self, thread_group: List[threading.Thread]) -> None:
        """Starts a group of threads at once
        Args:
            thread_group (list): a group of threads to be started
        """
        for t in thread_group:
            t.start()

    def join_threads(self, thread_group: List[threading.Thread]) -> None:
        """Joins a group of threads at once
        Args:
            thread_group (list): a group of threads to be joined
        """
        for t in thread_group:
            t.join()
            del t

    def render(
        self,
        renderer: SocNavRenderer,
        camera_pos_13: Optional[np.ndarray] = None,
        filename: Optional[str] = "sim_render",
    ) -> None:
        """Generates a png frame for each world state saved in self.sim_states. Note, based off the
        render_3D options, the function will generate the frames in multiple separate processes to
        optimize performance on multicore machines, else it can also be done sequentially.
        NOTE: the 3D renderer can currently only be run sequentially
        Args:
            filename (str, optional): name of each png frame (unindexed). Defaults to "obs".
        """
        if not self.params.render_params.render_movie:
            print(f"{color_text['orange']}Not rendering movie{color_text['reset']}")
            return

        # currently only single-threaded mode is supported for 3d rendering
        num_cores: int = 1 if self.params.render_3D else self.params.render_params.num_procs

        # Rendering movie
        fps: float = 1.0 / self.dt
        print(
            "Rendering movie with {}fps={}{} on {}{} processors{}".format(
                color_text["orange"],
                fps,
                color_text["reset"],
                color_text["blue"],
                num_cores,
                color_text["reset"],
            )
        )
        common_env: Dict[str, Any] = None

        # collect list of sim_states to render
        sim_state_bank = list(self.sim_states.values())

        # optionally (for multi-robot rendering) render this instead
        if self.params.render_params.draw_parallel_robots:
            if self.params.output_directory is None:
                self.params.output_directory = os.path.join(
                    self.params.socnav_params.socnav_dir,
                    "tests",
                    "socnav",
                    "test_multi_robot",
                    self.episode_params.name,
                )
                self.params.render_params.output_directory = (
                    self.params.output_directory
                )
                self.params.render_params.test_name = self.episode_params.name

            max_algo_times: Dict[str, float] = SimState.get_max_parallel_sim_states(
                self.params.render_params
            )
            max_sim_t = max(max_algo_times.values())
            num_states_per_proc = int(np.ceil((max_sim_t / self.dt) / num_cores))
            common_env = SimState.get_common_env(self.params.render_params)
            num_frames = num_states_per_proc * num_cores
        else:
            num_states_per_proc = int(np.ceil(len(self.sim_states) / num_cores))
            num_frames = int(np.ceil(len(sim_state_bank)))

        start_time = float(time.time())

        def worker_render_sim_states(procID: int) -> None:
            # runs an interleaved loop across sim_states in the bank
            mpl.use("Agg")  # for rendering without a display
            mpl.font_manager._get_font.cache_clear()
            for i in range(num_states_per_proc):
                sim_idx: int = procID + i * num_cores
                if self.params.render_params.draw_parallel_robots:
                    assert common_env is not None
                    SimState.render_multi_robot(
                        env=common_env,
                        sim_t=sim_idx * self.dt,
                        p=self.params,
                        max_algo_times=max_algo_times,
                        filename="{}_obs{:03d}.png".format(filename, sim_idx),
                    )
                elif sim_idx < len(sim_state_bank):
                    render_socnav(
                        sim_state=sim_state_bank[sim_idx],
                        renderer=renderer,
                        params=self.params,
                        camera_pos_13=camera_pos_13,
                        filename="{}_obs{:03d}.png".format(filename, sim_idx),
                    )
                print(
                    "Rendered frames: {}/{} ({:.2f}%)\r".format(
                        sim_idx, num_frames, 100.0 * min(1, sim_idx / num_frames),
                    ),
                    sep=" ",
                    end="",
                    flush=True,
                )  # inline print

        gif_processes: List[multiprocessing.Process] = []
        if num_cores > 1:
            # TODO: fix multiprocessing with Swiftshader engine!
            # currently only runs in single-process mode, deepcopying has a problem
            if self.params.render_3D == False:
                # optimized to use multiple processes
                for p in range(1, num_cores):
                    gif_processes.append(
                        multiprocessing.Process(
                            target=worker_render_sim_states, args=(p,)
                        )
                    )
                for proc in gif_processes:
                    proc.start()

        # run the renderer on the root processor
        worker_render_sim_states(0)

        # finish all the other processors if there are any
        for proc in gif_processes:
            proc.join()
        time_end = float(time.time())
        print(
            "Rendered frames: {}/{} ({:.2f}%)\n"
            "Finished rendering in {:.2f}s".format(
                num_frames, num_frames, 100.0, (time_end - start_time)
            )
        )  # make sure it says 100% at the end

        # convert all the generated frames into a gif file
        """NOTE: One can also save to mp4 using imageio-ffmpeg or this bash script:
              "ffmpeg -r 10 -i simulate_obs%01d.png -vcodec mpeg4 -y movie.mp4"
        Args:
            clear_old_files (bool, optional): Whether or not to clear old image files. Defaults to True.
        """
        if not self.params.render_params.render_movie:
            return
        save_to_gif(
            self.params.output_directory,
            fps=fps,
            filename="movie_{}".format(filename),
            use_ffmpeg=True,  # TODO: move to params
            clear_old_files=self.params.render_params.clear_files,
        )
        return


"""central sim - pandas utils"""


def sim_states_to_dataframe(sim) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Convert all states for non-robot agents into df
    :param sim:
    :return:
    """
    from simulators.simulator import Simulator

    if isinstance(sim, Simulator):
        all_states: List[SimState] = sim.states
    elif isinstance(sim, dict):
        all_states = sim

    cols = ["sim_step", "agent_name", "x", "y", "theta"]
    agent_info: Dict[str, List[float]] = {}  # later store traversibles
    df = pd.DataFrame(columns=cols, dtype=np.float64)

    # TODO: vectorize!!
    for sim_step, sim_state in all_states.items():
        for agent_name, agent in sim_state.get_all_agents(True).items():

            assert isinstance(agent, AgentState)
            traj = agent.current_config.position_and_heading_nk3(squeeze=True)
            agent_info[agent_name] = [agent.radius]

            if len(traj) == 0:
                continue
            # elif len(traj) > 1 and len(traj.shape) > 1:
            #     traj = traj[-1]
            #
            if len(traj) != 3:
                print(sim_step, traj)
                raise NotImplementedError

            x: float = traj[0]
            y: float = traj[1]
            th: float = traj[2]

            df.loc[len(df)] = [sim_step, agent_name, x, y, th]

    return df, agent_info


def add_sim_state_to_dataframe(
    sim_step: int, sim_state: SimState, df: pd.DataFrame, agent_info: Dict[str, List]
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
        append agents at sim_step*delta_t into df
        :param sim:
        :return:
    """
    for agent_name, agent in sim_state.items():
        if not isinstance(agent, dict):
            traj = np.squeeze(agent.vehicle_trajectory.position_and_heading_nk3())
            if agent_name not in agent_info:
                agent_info[agent_name] = [agent.get_radius()]
        else:
            traj = np.squeeze(agent["trajectory"])
            if agent_name not in agent_info:
                agent_info[agent_name] = [agent["radius"]]

        x, y, th = traj
        df.append([sim_step, agent_name, x, y, th])

    return df, agent_info
