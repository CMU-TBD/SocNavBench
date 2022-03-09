import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from agents.agent import Agent
from agents.robot_agent import RobotAgent
from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from socnav.socnav_renderer import SocNavRenderer
from utils.utils import (
    absmax,
    color_text,
    euclidean_dist2,
    iter_print,
    termination_cause_to_color,
    touch,
)

from simulators.sim_state import AgentState, SimState
from simulators.simulator_helper import SimulatorHelper


class Simulator(SimulatorHelper):
    """The centralized simulator of SocNavBench """

    def __init__(
        self,
        environment: Dict[str, int or float or np.ndarray],
        renderer: Optional[SocNavRenderer] = None,
        episode_params: Optional[DotMap] = None,
        verbose: Optional[bool] = True,
    ):
        """ Initializer for the central simulator
        Args:
            environment (dict): dictionary housing the obj map (bitmap) and more
            renderer (optional): OpenGL renderer for 3D models. Defaults to None
            episode_params (str, optional): Name of the episode test that the simulator runs
        """
        # init SimulatorHelper base class
        super().__init__(environment, verbose)
        # init Simulator implementation
        self.episode_params: DotMap = episode_params
        # output directory is updated again if there is a robot (and algorithm) in the simulator
        self.params.output_directory = None
        self.params.render_params.output_directory = None
        if environment is not None:
            self.obstacle_map: SBPDMap = self.init_obstacle_map(renderer)
        self.r: SocNavRenderer = renderer
        # only export the metadata for sim_states on the first one
        self.first_export_metadata = True

    def init_sim_data(self, verbose: Optional[bool] = True) -> None:
        self.total_agents: int = len(self.agents) + len(self.backstage_prerecs)
        # Create pre-simulation metadata
        if verbose:
            print("Running simulation on", self.total_agents, "agents")
        # scale the simulator time
        self.dt: float = self.params.delta_t_scale * self.params.dt
        # update the baseline agents' simulation refresh rate
        Agent.set_sim_dt(self.dt)
        Agent.set_sim_t(self.sim_t)
        # add the first (when t=0) agents to the self.prerecs dict
        self.init_prerec_agent_threads(current_state=None)
        # save initial state before the simulator is spawned
        self.sim_t = 0.0
        if self.dt < self.params.dt:
            print(
                "%sSimulation dt is too small; either lower the gen_agents' dt's"
                % color_text["red"],
                self.params.dt,
                "or increase simulation delta_t%s" % color_text["reset"],
            )
            exit(1)

    def loop_condition(self) -> bool:
        if self.robot:
            # stop the simulation if the robot has exited
            return not self.robot.end_acting
        # else just run until there are no more agents
        return self.exists_running_agent() or self.exists_running_prerec()

    def simulate(self) -> None:
        """ A function that simulates an entire episode. The gen_agents are updated with simultaneous
        threads running their update() functions and updating the robot with commands from the
        external joystick process.
        """
        # initialize pre-simulation metadata
        self.init_sim_data()
        # keep track of wall-time in the simulator
        start_time: float = time.time()
        # get initial state
        current_state: SimState = self.save_state()
        # initialize robot update thread
        r_t: threading.Thread = self.init_robot_listener_thread(current_state)
        # start iteration
        iteration: int = 0
        self.print_sim_progress(iteration)
        # run simulation
        while self.sim_t <= self.episode_params.max_time and self.loop_condition():
            wall_t: float = time.time()
            # update the time for all agents
            Agent.set_sim_t(self.sim_t)
            # initiate thread operations
            self.pedestrians_update(current_state)
            if self.robot is not None:
                # calls a single iteration of the robot update
                self.robot.update()
            # update simulator time
            self.sim_t += self.dt
            # capture time after all the gen_agents have updated
            # Takes screenshot of the new simulation state
            current_state = self.save_state(wall_t - start_time)
            if self.robot:
                self.robot.update_world(current_state)
            # update iteration count
            iteration += 1
            # print simulation progress
            self.print_sim_progress(iteration)
            # synchronize time with real-world if running in asynchronous mode
            self.synchronize(wall_t)
        # finish the simulate
        self.conclude_simulation(start_time, iteration, r_t)

    def synchronize(self, wall_t: float) -> None:
        # get time difference between NOW and when the wall_t was last updated
        # (occurs at the start of every simulate() cycle )
        w_dt: float = time.time() - wall_t
        # TODO: note there is danger if w_dt takes longer than self.dt
        if not self.params.block_joystick:
            if w_dt > self.dt:
                print(
                    "%sSim-cycle took %.3fs > %.3fs%s"
                    % (color_text["red"], w_dt, self.dt, color_text["reset"])
                )
                return
            # sleep to run in as-close-as-possible to real-time
            time.sleep(self.dt - w_dt)

    def conclude_simulation(
        self, start_time: float, iteration: int, r_t: threading.Thread
    ) -> None:
        # free all the gen_agents
        for a in self.agents.values():
            del a
        # free all the prerecs
        for p in self.prerecs.values():
            del p
        # turn off the robot if it is still on
        # capture final wall clock (completion) time
        self.sim_wall_clock = time.time() - start_time
        print("\nSimulation completed in %.4f real world seconds" % self.sim_wall_clock)
        # decommission_robot
        if self.robot is not None:
            if not self.robot.get_end_acting():
                self.robot.power_off()
            self.robot_collisions = self.gather_robot_collisions(iteration)
            c = termination_cause_to_color(self.robot.termination_cause)
            print(
                "Robot termination cause: %s%s%s"
                % (color_text[c], self.robot.termination_cause, color_text["reset"])
            )
        if self.episode_params.write_episode_log:
            self.generate_sim_log()
        if self.robot is not None:
            # TODO generate + write the score report
            from simulators.simulator_helper import sim_states_to_dataframe

            self.sim_df, self.agent_info = sim_states_to_dataframe(self.sim_states)
            self.generate_episode_score_report()
            # finally close the robot listener thread
            self.decommission_robot(r_t)

    def save_state(self, wall_t: Optional[float] = 0.0) -> SimState:
        """Captures the current state of the world to be saved to self.sim_states
        Args:
            sim_t (float): the current time in the simulator in seconds
            delta_t (float): the timestep size in the simulator in seconds
            wall_t (float): the current wall clock time
        Returns:
            current_state (SimState): the most recent state of the world
        """
        # NOTE: when using a modular environment, make saved_env a deepcopy
        saved_env = self.environment
        pedestrians: Dict[str, AgentState] = {}
        for a in self.agents.values():
            pedestrians[a.get_name()] = AgentState.from_agent(a)
        # deepcopy all prerecorded gen_agents
        for a in self.prerecs.values():
            pedestrians[a.get_name()] = AgentState.from_agent(a)
        # Save all the robots
        saved_robots: Dict[str, RobotAgent] = {}
        last_robot_collision: str = ""
        if self.robot:
            saved_robots[self.robot.get_name()] = AgentState.from_agent(self.robot)
            last_robot_collision = self.robot.latest_collider
        current_state = SimState(
            environment=saved_env,
            pedestrians=pedestrians,
            robots=saved_robots,
            sim_t=self.sim_t,
            wall_t=wall_t,
            delta_t=self.dt,
            episode_name=self.episode_params.name,
            max_time=self.episode_params.max_time,
            ped_collider=last_robot_collision,
        )
        # Save current state to a class dictionary indexed by simulator time
        sim_t_step: int = round(self.sim_t / self.dt)
        self.sim_states[sim_t_step] = current_state
        if self.algo_name is not None:
            current_state.export_to_file(
                out_dir=os.path.join(self.params.output_directory, "sim_state_data"),
                export_metadata=self.first_export_metadata,
            )
            self.first_export_metadata = False
        # debug prints
        return current_state

    """ BEGIN SCORING UTILS """

    def generate_episode_score_report(
        self, filename: Optional[str] = "episode_score"
    ) -> None:
        # should do this in some formal format
        # json? pandas? how to aggregate per episode?
        # TODO how to have a list of metrics? for now hardcoded
        # different analysis based on success and failure
        metrics_list: List[str] = [
            # meta
            "success",
            "termination_cause",
            "map",
            "total_sim_time_taken",
            "sim_time_budget",
            "wall_wait_time",
            # motion
            "robot_speed",
            "robot_motion_energy",
            "robot_acceleration",
            "robot_jerk",
            # path
            "path_length",
            "path_length_ratio",
            "goal_traversal_ratio",
            "path_irregularity",
            # individual
            "personal_space_cost",
            "closest_pedestrian_distance",
            "time_to_collision",
            # others
            # Time to collision
            # Planning-based?
        ]

        # fail_metrics = [
        #     "goal_traversal_ratio"
        # ]

        # score_df = pd.DataFrame(columns=metrics_list)

        ep_params = self.episode_params
        filename = "episode_score_%s.pkl" % ep_params.name
        abs_filename = os.path.join(self.params.output_directory, filename)
        touch(abs_filename)
        metrics_out = {}

        from metrics import metrics_sim_utils

        for metric in metrics_list:
            try:
                metric_fn = eval("metrics_sim_utils." + metric)
            except (AttributeError, NameError):
                import logging

                logging.info(
                    "The metric %s is not implemented yet" % metric
                )  # will not print anything
                continue
            metrics_out[metric] = metric_fn(self)

        # other ROBOT INFO

        metrics_out["num_recv_joystick"] = len(self.robot.joystick_inputs)
        metrics_out["num_exec_robot"] = self.robot.num_executed

        try:
            with open(abs_filename, "wb") as f:
                # f.write(metrics_out)
                import pickle

                pickle.dump(metrics_out, f)

            print(
                "%sSuccessfully wrote episode metrics to %s%s"
                % (color_text["green"], abs_filename, color_text["reset"])
            )
        except:
            print(
                "%sWriting episode metrics failed%s"
                % (color_text["red"], color_text["reset"])
            )
        return

    def generate_sim_log(self, filename: Optional[str] = "episode_log.txt") -> None:
        abs_filename: str = os.path.join(self.params.output_directory, filename)
        touch(abs_filename)  # create if dosent already exist
        ep_params: DotMap = self.episode_params
        data: str = ""
        data += "****************EPISODE INFO****************\n"
        data += "Episode name: %s\n" % ep_params.name
        data += "Building name: %s\n" % ep_params.map_name
        data += "Robot start: %s\n" % str(ep_params.robot_start_goal[0])
        data += "Robot goal: %s\n" % str(ep_params.robot_start_goal[1])
        data += "Time budget: %.3f\n" % ep_params.max_time
        # data += "Prerec start indx: %d\n" % ep_params.prerec_start_indxs[0]
        data += "Total agents in scene: %d\n" % self.total_agents
        data += "****************SIMULATOR INFO****************\n"
        data += "Simulator refresh rate (s): %0.3f\n" % self.dt
        data += "Total duration of simulation (s): %0.3f\n" % self.sim_wall_clock
        num_successful = self.num_completed_agents
        data += "Num Successful agents: %d\n" % num_successful
        num_collision = self.num_collided_agents
        data += "Num Collided agents: %d\n" % num_collision
        num_timeout = self.total_agents - (num_successful + num_collision)
        data += "Num Timeout agents: %d\n" % num_timeout

        if self.robot:
            data += "****************ROBOT INFO****************\n"
            data += "Robot termination cause: %s\n" % self.robot.termination_cause
            data += "Robot collided with %d agent(s)\n" % len(self.robot_collisions)
            if len(self.robot_collisions) != 0:
                data += "Collided with: %s\n" % iter_print(self.robot_collisions)
            data += "Num commands received from joystick: %d\n" % len(
                self.robot.joystick_inputs
            )
            data += (
                "Total time blocking for joystick input (s): %0.3f\n"
                % self.robot.get_block_t_total()
            )
            data += "Num commands executed by robot: %d\n" % self.robot.num_executed
            rob_displacement = euclidean_dist2(
                ep_params.robot_start_goal[0],
                self.robot.get_current_config().position_and_heading_nk3(squeeze=True),
            )
            data += "Robot displacement (m): %0.3f\n" % rob_displacement
            data += "Max robot velocity (m/s): %0.3f\n" % absmax(
                self.robot.get_trajectory().speed_nk1()
            )
            data += "Max robot acceleration: %0.3f\n" % absmax(
                self.robot.get_trajectory().acceleration_nk1()
            )
            data += "Max robot angular velocity: %0.3f\n" % absmax(
                self.robot.get_trajectory().angular_speed_nk1()
            )
            data += "Max robot angular acceleration: %0.3f\n" % absmax(
                self.robot.get_trajectory().angular_acceleration_nk1()
            )
        try:
            with open(abs_filename, "w") as f:
                f.write(data)
                f.close()
            print(
                "%sSuccessfully wrote episode log to %s%s"
                % (color_text["green"], filename, color_text["reset"])
            )
        except:
            print(
                "%sWriting episode log failed%s"
                % (color_text["red"], color_text["reset"])
            )

    """ BEGIN ROBOT UTILS """

    def init_robot_listener_thread(
        self, current_state: SimState, power_on: Optional[bool] = True
    ) -> Optional[threading.Thread]:
        """Initializes the robot listener by establishing socket connections to
        the joystick, transmitting the (constant) obstacle map (environment),
        and starting the robot thread.
        Args:
            power_on (bool, optional): Whether or not the robot should start on. Defaults to True.
        Returns:
            Thread: The robot's update thread if it exists in the simulator, else None
        """
        r_listener_thread = None
        if self.robot is None:
            print(
                "%sNo robot in simulator%s" % (color_text["red"], color_text["reset"])
            )
            self.algo_name = "none" # no robot present
        else:
            # wait for joystick connection to be established
            # give the robot knowledge of the initial world
            self.robot.block_joystick = self.params.block_joystick
            self.robot.update_world(current_state)
            # initialize the robot to establish joystick connection
            assert self.robot.world_state is not None
            # send first transaction to the joystick
            print("Sending episode data to joystick...")
            r_listener_thread = threading.Thread(target=self.robot.listen_to_joystick)
            if power_on:
                r_listener_thread.start()
            # wait until joystick is ready
            while not self.robot.joystick_ready:
                # wait until joystick receives the environment (once)
                time.sleep(0.01)
            # either "Unknown" if the robot did not receive an algorithm title
            # or the name of the planning algorithm used by the joystick
            self.algo_name = self.robot.algo_name
            print(
                "Robot powering on with algorithm: {}{}{}".format(
                    color_text["orange"], self.algo_name, color_text["reset"]
                )
            )
        # name of the directory to output everything
        self.params.output_directory = os.path.join(
            self.params.socnav_params.socnav_dir,
            "tests/socnav/",
            "test_" + self.algo_name,
            self.episode_params.name,
        )
        self.params.render_params.output_directory = self.params.output_directory
        self.params.algo_name = self.algo_name
        return r_listener_thread

    def decommission_robot(self, r_listener_thread: threading.Thread) -> None:
        """Turns off the robot and joins the robot's update thread
        Args:
            r_listener_thread (Thread): the robot update thread to join
        """
        if r_listener_thread is not None:
            # close robot listener threads
            if r_listener_thread.is_alive():
                # connect to its own socket to break the accept() loop
                self.robot.force_connect_self()
                r_listener_thread.join()
            del r_listener_thread

    def pedestrians_update(self, current_state: SimState) -> None:
        if self.params.use_multithreading:
            agent_threads = self.init_auto_agent_threads(current_state)
            prerec_threads = self.init_prerec_agent_threads(current_state)
            pedestrian_threads = agent_threads + prerec_threads
            # start agent threads
            self.start_threads(pedestrian_threads)
            # join all thread groups
            self.join_threads(pedestrian_threads)
        else:
            self.loop_through_pedestrians(current_state)

    def add_agents(self, agents: List[Agent]) -> None:
        """
        Add existing agents to the simulator
        """
        if self.params.render_3D:
            self.environment["human_traversible"] = self.r.building.human_traversible
        for agent in agents:
            self.add_agent(agent)
