import numpy as np
import multiprocessing
import threading
from utils.utils import *
from utils.image_utils import *
from agents.agent import Agent
from simulators.sim_state import SimState
from socnav.socnav_renderer import SocNavRenderer
from params.central_params import create_simulator_params
import pandas as pd


class SimulatorHelper(object):

    def __init__(self, environment: dict):
        """ Initializer for the central simulator
        Args:
            environment (dict): dictionary housing the obj map (bitmap) and more
            renderer (optional): OpenGL renderer for 3D models. Defaults to None
            episode_params (str, optional): Name of the episode test that the simulator runs
        """
        self.environment = environment
        self.params = create_simulator_params()
        self.algo_name = "lite"  # by default there is no robot (or algorithm)
        self.obstacle_map = None
        # keep track of all agents in dictionary with names as the key
        self.agents = {}
        # keep track of all robots in dictionary with names as the key
        self.robots = {}
        # keep track of all prerecorded humans in a dictionary like the otherwise
        self.backstage_prerecs = {}
        self.prerecs = {}
        # keep a single (important) robot as a value
        self.robot = None
        self.sim_states = {}
        self.wall_clock_time: float = 0
        self.sim_t: float = 0.0
        self.dt: float = 0  # will be updated in simulator based off dt
        # metadata of agents
        self.total_agents: int = 0
        self.num_collided_agents: int = 0
        self.num_completed_agents: int = 0
        # restart agent coloring on every instance of the simulator to be consistent across episodes
        Agent.restart_coloring()

    def add_agent(self, a):
        """Adds an agent member to the central simulator's pool of gen_agents
        NOTE: this function works for robots (RobotAgent), prerecorded gen_agents (PrerecordedHuman),
              and general gen_agents (Agent)
        Args:
            a (Agent/PrerecordedAgent/RobotAgent): The agent to be added to the simulator
        """
        assert(self.obstacle_map is not None)
        name = a.get_name()
        from agents.robot_agent import RobotAgent
        from agents.humans.recorded_human import PrerecordedHuman
        if isinstance(a, RobotAgent):
            # initialize the robot and add to simulator's known "robot" field
            a.simulation_init(sim_map=self.obstacle_map,
                              with_planner=False,
                              keep_episode_running=self.params.keep_episode_running)
            self.robots[name] = a
            self.robot = a
        elif isinstance(a, PrerecordedHuman):
            # generic agent initializer but without a planner (already have trajectories)
            a.simulation_init(sim_map=self.obstacle_map,
                              with_planner=False,
                              with_system_dynamics=False,
                              with_objectives=False,
                              keep_episode_running=self.params.keep_episode_running)
            # added to backstage prerecs which will add to self.prerecs when the time is right
            self.backstage_prerecs[name] = a
        else:
            # initialize agent and add to simulator
            a.simulation_init(sim_map=self.obstacle_map,
                              with_planner=True,
                              keep_episode_running=self.params.keep_episode_running)
            self.agents[name] = a

    def exists_running_agent(self):
        """Checks whether or not a generated agent is still running (acting)
        Returns:
            bool: True if there is at least one running agent, False otherwise
        """
        for a in self.agents.values():
            if (not a.end_acting):  # if there is even just a single agent acting
                return True
        return False

    def exists_running_prerec(self):
        """Checks whether or not a prerecorded agent is still running (acting)
        Returns:
            bool: True if there is at least one running prerec, False otherwise
        """
        # make sure there are still remaining pedestrians in the backstage
        return not (not self.backstage_prerecs)

    def init_obstacle_map(self, renderer=None, ):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p, renderer,
                              res=self.environment["map_scale"] * 100,
                              map_trav=self.environment["map_traversible"])

    def loop_condition(self):
        raise NotImplementedError

    def simulate(self):
        raise NotImplementedError

    def print_sim_progress(self, rendered_frames: int):
        """prints an inline simulation progress message based off agent planning termination
            TODO: account for agent<->agent collisions
        Args:
            rendered_frames (int): how many frames have been generated so far
        """
        num_timeout = self.total_agents - \
            self.num_completed_agents - self.num_collided_agents
        print("A:", self.total_agents,
              "%sSuccess:" % (color_green), self.num_completed_agents,
              "%sCollide:" % (color_red), self.num_collided_agents,
              "%sTime:" % (color_blue), num_timeout,
              "%sFrames:" % (color_reset), rendered_frames,
              "T = %.3f" % (self.sim_t),
              "\r", end="")

    def gather_robot_collisions(self, max_iter: int):
        agent_collisions = []
        for i in range(max_iter):
            collider = self.sim_states[i].get_collider()
            if(collider != ""):
                agent_collisions.append(collider)
        last_collider = self.robot.latest_collider
        if(last_collider != ""):
            agent_collisions.append(last_collider)
        return agent_collisions

    def init_auto_agent_threads(self, current_state: SimState):
        """Spawns a new agent thread for each agent (running or finished)
        Args:
            current_state (SimState): the most recent state of the world
        Returns:
            agent_threads (list): list of all spawned (not started) agent threads
        """
        agent_threads = []
        all_agents = list(self.agents.values())
        for a in all_agents:
            if(not a.end_acting):
                agent_threads.append(threading.Thread(target=a.update,
                                                      args=(current_state,)))
            else:
                if(a.get_collided()):
                    self.num_collided_agents += 1
                else:
                    self.num_completed_agents += 1
                del(self.agents[a.get_name()])
                del(a)
        return agent_threads

    def init_prerec_agent_threads(self, current_state: SimState):
        """Spawns a new prerec thread for each running prerecorded agent
        Args:
            current_state (SimState): the current state of the world
        Returns:
            prerec_threads (list): list of all spawned (not started) prerecorded agent threads
        """
        prerec_threads = []
        all_prerec_agents = list(self.backstage_prerecs.values())
        for a in all_prerec_agents:
            if(not a.end_acting and a.get_start_time() <= Agent.sim_t < a.get_end_time()):
                # only add (or keep) agents in the time frame
                self.prerecs[a.get_name()] = a
                prerec_threads.append(threading.Thread(target=a.update,
                                                       args=(current_state,)))
            else:
                # remove agent since its not within the time frame or finished
                if(a.get_name() in self.prerecs.keys()):
                    if(a.get_collided()):
                        self.num_collided_agents += 1
                    else:
                        self.num_completed_agents += 1
                    self.prerecs.pop(a.get_name())
                    # also remove from back stage since they will no longer be used
                    del(self.backstage_prerecs[a.get_name()])
                    del(a)
        return prerec_threads

    def start_threads(self, thread_group):
        """Starts a group of threads at once
        Args:
            thread_group (list): a group of threads to be started
        """
        for t in thread_group:
            t.start()

    def join_threads(self, thread_group):
        """Joins a group of threads at once
        Args:
            thread_group (list): a group of threads to be joined
        """
        for t in thread_group:
            t.join()
            del(t)

    def render(self, renderer, camera_pose, filename: str = "obs"):
        """Generates a png frame for each world state saved in self.sim_states. Note, based off the
        render_3D options, the function will generate the frames in multiple separate processes to
        optimize performance on multicore machines, else it can also be done sequentially.
        NOTE: the 3D renderer can currently only be run sequentially
        Args:
            filename (str, optional): name of each png frame (unindexed). Defaults to "obs".
        """
        if self.params.fps_scale_down == 0 or not self.params.record_video:
            print("%sNot rendering movie%s" % (color_orange, color_reset))
            return

        # Rendering movie
        fps = (1.0 / self.dt) * self.params.fps_scale_down
        print("%sRendering movie with fps=%d%s" %
              (color_orange, fps, color_reset))
        num_frames = \
            int(np.ceil(len(self.sim_states) * self.params.fps_scale_down))
        np.set_printoptions(precision=3)

        if not self.params.render_3D:
            # optimized to use multiple processes
            # TODO: put a limit on the maximum number of processes that can be run at once
            gif_processes = []
            skip = 0
            frame = 0
            for p, s in enumerate(self.sim_states.values()):
                if skip == 0:
                    # pool.apply_async(self.render_sim_state, args=(s, filename + str(p) + ".png"))
                    frame += 1
                    gif_processes.append(multiprocessing.Process(
                        target=self.render_sim_state,
                        args=(renderer, camera_pose, s, filename + str(p) + ".png"))
                    )
                    gif_processes[-1].start()
                    print("Started processes: %d out of %d, %.3f%% \r" %
                          (frame, num_frames, 100.0 * (frame / num_frames)), end="")
                    # reset skip counter for frames
                    skip = int(1.0 / self.params.fps_scale_down) - 1
                else:
                    # skip certain other frames as directed by the fps_scale_down
                    skip -= 1
            print()  # not overwrite next line
            for frame, p in enumerate(gif_processes):
                p.join()
                print("Finished processes: %d out of %d, %.3f%% \r" %
                      (frame + 1, num_frames, 100.0 * ((frame + 1) / num_frames)), end="")
            print()  # not overwrite next line
        else:
            # generate frames sequentially (non multiproceses)
            skip = 0
            frame = 0
            for s in self.sim_states.values():
                if skip == 0:
                    self.render_sim_state(renderer, camera_pose, s,
                                          filename + str(frame) + ".png")
                    frame += 1
                    print("Generated Frames: %d out of %d, %.3f%% \r" %
                          (frame, num_frames, 100.0 * (frame / num_frames)), end="")
                    del s  # free the state from memory
                    skip = int(1.0 / self.params.fps_scale_down) - 1
                else:
                    skip -= 1.0
            print()

        # convert all the generated frames into a gif file
        self.save_frames_to_gif(filename=self.episode_params.name)

    def render_sim_state(self, renderer: SocNavRenderer, camera_pose: list,
                         state: SimState, filename: str):
        """Converts a state into an image to be later converted to a gif movie
        Args:
            state (SimState): the state of the world to convert to an image
            filename (str): the name of the resulting image (unindexed)
        """
        if self.robot:
            robot = list(state.get_robots().values())[0]
            camera_pos_13 = robot.get_current_config().to_3D_numpy()
        else:
            robot = None
            if camera_pose is not None:
                camera_pos_13 = camera_pose
            else:
                camera_pos_13 = state.get_environment()["room_center"]

        rgb_image_1mk3 = None
        depth_image_1mk1 = None
        # NOTE: 3d renderer can only be used with sequential plotting, much slower
        if self.params.render_3D:
            # TODO: Fix multiprocessing for properly deepcopied renderers
            # only when rendering with opengl
            assert("human_traversible" in state.get_environment().keys())
            # remove the "old" humans
            renderer.remove_all_humans()
            # update pedestrians humans
            for a in state.get_pedestrians().values():
                renderer.update_human(a)
            # Update human traversible
            # NOTE: this is technically not R-O since it modifies the human trav
            # TODO: use a separate variable to keep SimStates as R-O
            # state.get_environment()["human_traversible"] = \
            #     renderer.get_human_traversible()
            # compute the rgb and depth images
            rgb_image_1mk3, depth_image_1mk1 = \
                render_rgb_and_depth(renderer, np.array([camera_pos_13]),
                                     state.get_environment()["map_scale"],
                                     human_visible=True)
        # plot the rbg, depth, and topview images if applicable
        render_scene(self.params, rgb_image_1mk3, depth_image_1mk1,
                     state.get_environment(), camera_pos_13,
                     state.get_pedestrians(), state.get_robots(),
                     state.get_sim_t(), state.get_wall_t(), filename)
        # Delete state to save memory after frames are generated
        del state

    def save_frames_to_gif(self, filename=""):
        """Convert a directory full of png's to a gif movie
        NOTE: One can also save to mp4 using imageio-ffmpeg or this bash script:
              "ffmpeg -r 10 -i simulate_obs%01d.png -vcodec mpeg4 -y movie.mp4"
        Args:
            clear_old_files (bool, optional): Whether or not to clear old image files. Defaults to True.
        """
        if self.params.fps_scale_down == 0 or not self.params.record_video:
            return
        duration = self.dt * (1.0 / self.params.fps_scale_down)
        # sequentially
        gif_filename = "movie_%s" % filename
        save_to_gif(self.params.output_directory, duration,
                    gif_filename=gif_filename,
                    clear_old_files=self.params.clear_files)


"""central sim - pandas utils"""


def sim_states_to_dataframe(sim):
    """
    Convert all states for non-robot agents into df
    :param sim:
    :return:
    """
    from simulators.central_simulator import CentralSimulator

    if isinstance(sim, CentralSimulator):
        all_states = sim.states
    elif isinstance(sim, dict):
        all_states = sim

    cols = ["sim_step", "agent_name", "x", "y", "theta"]
    agent_info = {}  # for now store radius, later store traversibles
    df = pd.DataFrame(columns=cols)

    for sim_step, sim_state in all_states.items():
        for agent_name, agent in sim_state.get_all_agents(True).items():

            if not isinstance(agent, dict):
                # traj = np.squeeze(agent.vehicle_trajectory.position_and_heading_nk3())
                traj = np.squeeze(
                    agent.current_config.position_and_heading_nk3())
                agent_info[agent_name] = [agent.get_radius()]
            else:
                traj = np.squeeze(agent["trajectory"])
                agent_info[agent_name] = [agent["radius"]]

            if len(traj) == 0:
                continue
            # elif len(traj) > 1 and len(traj.shape) > 1:
            #     traj = traj[-1]
            #
            if len(traj) != 3:
                print(sim_step, traj)
                raise NotImplementedError

            x, y, th = traj

            df.loc[len(df)] = [sim_step, agent_name, x, y, th]

    return df, agent_info


def add_sim_state_to_dataframe(sim_step, sim_state, df, agent_info):
    """
        append agents at sim_step*delta_t into df
        :param sim:
        :return:
    """
    for agent_name, agent in sim_state.items():
        if not isinstance(agent, dict):
            traj = np.squeeze(
                agent.vehicle_trajectory.position_and_heading_nk3())
            if not agent_name in agent_info.keys():
                agent_info[agent_name] = [agent.get_radius()]
        else:
            traj = np.squeeze(agent["trajectory"])
            if not agent_name in agent_info.keys():
                agent_info[agent_name] = [agent["radius"]]

        x, y, th = traj
        df.append([sim_step, agent_name, x, y, th])

    return df, agent_info
