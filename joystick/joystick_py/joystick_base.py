import socket
import json
import os
import random
from utils.utils import *
from params.central_params import create_joystick_params, get_path_to_socnav, create_robot_params, get_seed
from simulators.sim_state import SimState
from simulators.episode import Episode


# seed the random number generator
random.seed(get_seed())


class JoystickBase():
    def __init__(self, algorithm_name: str = "socnav"):
        self.joystick_params = create_joystick_params()
        self.algorithm_name = algorithm_name
        print("Joystick running %s algorithm" % self.algorithm_name)
        # include the system dynamics for both posn & velocity commands
        from params.central_params import create_system_dynamics_params
        self.system_dynamics_params = create_system_dynamics_params()
        if self.joystick_params.use_system_dynamics:
            print("Joystick using system dynamics")
        else:
            print("Joystick using positional dynamics")
        # episode fields
        self.episode_names = []
        self.current_ep = None
        self.sim_state_now = None
        # main fields
        self.joystick_on = True  # status of the joystick
        # socket fields
        self.robot_sender_socket = None    # the socket for sending commands to the robot
        self.robot_receiver_socket = None  # world info receiver socket
        # flipped bc joystick-recv = robot-send & vice versa
        self.send_ID = create_robot_params().recv_ID
        self.recv_ID = create_robot_params().send_ID
        print("Initiated joystick locally (AF_UNIX) at \"%s\" & \"%s\"" %
              (self.send_ID, self.recv_ID))
        # potentially add more fields based off the params
        self.param_based_init()

    def param_based_init(self):
        # data tracking with pandas
        if self.joystick_params.write_pandas_log:
            import pandas as pd
            global pd
            self.pd_df = None    # pandas dataframe for writing to a csv
            self.agent_log = {}  # log of all the agents as updated by sensing

        # keeping an explicit log of every sim_state indexed by time
        if self.joystick_params.track_sim_states:
            self.sim_states = {}  # log of simulator states indexed by time

        # tracking velocity and acceleration of the agents from the sim states
        if self.joystick_params.track_vel_accel:
            global velocities
            from simulators.sim_state import compute_all_velocities as velocities
            global accelerations
            from simulators.sim_state import compute_all_accelerations as accelerations
            self.velocities = {}     # velocities of all agents as sensed by the joystick
            self.accelerations = {}  # accelerations of all agents as sensed by the joystick

        # plotting the sim states received from the robot
        if self.joystick_params.generate_movie:
            import matplotlib as mpl
            mpl.use('Agg')  # for rendering without a display
            import matplotlib.pyplot as plt
            from utils.image_utils import save_to_gif

    def get_episodes(self):
        return self.episode_names

    def get_robot_start(self):
        return self.current_ep.get_robot_start()

    def get_robot_goal(self):
        return self.current_ep.get_robot_goal()

    def init_obstacle_map(self):
        raise NotImplementedError

    def init_control_pipeline(self):
        pass

    def send_cmds(self, cmds, send_vel_cmds: bool = True):
        assert(send_vel_cmds == self.joystick_params.use_system_dynamics)
        if send_vel_cmds:
            # needs v, w
            for command_grp in cmds:
                if len(command_grp) != 2:
                    print("%sERROR: joystick expecting (v, w) for velocity commands. Got \"(%s)\"%s" % (
                        color_red, iter_print(command_grp), color_reset))
                assert(len(command_grp) == 2)
        else:
            # needs x, y, th, v
            for command_grp in cmds:
                if len(command_grp) != 4:
                    print("%sERROR: joystick expecting (x, y, theta, v) for positional commands. Got \"(%s)\"%s" % (
                        color_red, iter_print(command_grp), color_reset))
                assert(len(command_grp) == 4)
        json_dict = {}
        json_dict["j_input"] = cmds
        serialized_cmds = json.dumps(json_dict, indent=1)
        self.send_to_robot(serialized_cmds)

    def joystick_sense(self):
        raise NotImplementedError

    def joystick_plan(self):
        raise NotImplementedError

    def joystick_act(self):
        raise NotImplementedError

    def update_loop(self):
        raise NotImplementedError

    def pre_update(self):
        assert(self.sim_dt is not None)
        self.robot_receiver_socket.listen(1)  # init robot listener socket
        self.joystick_on = True

    def finish_episode(self):
        # finished this episode
        print("%sFinished episode:" % color_green,
              self.current_ep.get_name(), "%s" % color_reset)
        # If current episode is the last one, the joystick is done
        if self.current_ep.get_name() == self.episode_names[-1]:
            self.close_recv_socket()
            print("Finished all episodes\n\n")
        else:
            self.current_ep = None

    """ BEGIN LISTEN UTILS """

    def get_all_episode_names(self):
        # sets the data to look for all the episode names
        return self.listen_once(0)

    def get_episode_metadata(self):
        # sets data_type to look for the episode metadata
        return self.listen_once(1)

    def listen_once(self, data_type: int = 2):
        """Runs a single instance of listening to the receiver socket
        to obtain information about the world and episode metadata
        Args:
            data_type (int): 0 if obtaining all episode names,
                             1 if obtaining specific episode metadata,
                             2 if obtaining simulator info from a sim_state.
                             Defaults to 2.
        Returns:
            [bool]: True if the listening was successful, False otherwise
        """
        connection, _ = self.robot_receiver_socket.accept()
        data_b, response_len = conn_recv(connection)
        # quickly close connection to open up for the next input
        connection.close()
        if self.joystick_params.verbose:
            print("%sreceived" % color_blue, response_len,
                  "bytes from robot%s" % color_reset)
        if response_len > 0:
            data_str = data_b.decode("utf-8")  # bytes to str
            data_json = json.loads(data_str)
            if (data_type == 0):
                return self.manage_episodes_name_data(data_json)
            elif (data_type == 1):
                return self.manage_episode_data(data_json)
            else:
                return self.manage_sim_state_data(data_json)
        # received no information from joystick
        self.joystick_on = False
        return False

    def manage_episodes_name_data(self, episode_names_json: dict):
        # case where there is no simulator yet, just episodes
        assert('episodes' in episode_names_json.keys())
        self.episode_names = episode_names_json['episodes']
        print("Received episodes:", self.episode_names)
        assert(len(self.episode_names) > 0)
        return True  # valid parsing of the data

    def manage_episode_data(self, initial_sim_state_json: dict):
        current_world = SimState.from_json(initial_sim_state_json)
        # not empty dictionary
        assert(not (not current_world.get_environment()))
        self.update_knowledge_from_episode(current_world, init_ep=True)
        # send the algorithm name to the robot/simulator
        self.send_to_robot("algo: " + self.algorithm_name)
        # ping the robot that the joystick received the episode (keyword)
        self.send_to_robot("ready")
        return True

    def manage_sim_state_data(self, sim_state_json: dict):
        # case where the robot sends a power-off signal
        if not sim_state_json['robot_on']:
            term_status = sim_state_json['termination_cause']
            term_color = color_print(termination_cause_to_color(term_status))
            print("\npowering off joystick, robot terminated with: %s%s%s" %
                  (term_color, term_status, color_reset))
            self.joystick_on = False
            return False  # robot is off, do not continue
        else:
            self.sim_state_now = SimState.from_json(sim_state_json)
            # only update the SimStates for non-environment configs
            self.update_knowledge_from_episode(self.sim_state_now)

            # update the history of past sim_states if requested
            if self.joystick_params.track_sim_states:
                self.sim_states[self.sim_state_now.get_sim_t()] = \
                    self.sim_state_now

            print("Updated state of the world for time = %.3f out of %.3f\r" %
                  (self.sim_state_now.get_sim_t(),
                   self.current_ep.get_time_budget()),
                  sep=' ', end="", flush=True)

            # self.track_vel_accel(self.sim_state_now)  # TODO: remove

            if self.joystick_params.write_pandas_log:
                # used for file IO such as pandas logging
                # NOTE: this MUST match the directory name in Simulator
                self.dirname = 'tests/socnav/' + "test_" + self.algorithm_name + "/" +\
                    self.current_ep.get_name() + "/joystick_data"
                # Write the Agent's trajectory data into a pandas file
                self.update_logs(self.sim_state_now)
                self.write_pandas()
        return True

    def update_knowledge_from_episode(self, current_world, init_ep: bool = False):
        name = current_world.get_episode_name()
        env = current_world.get_environment()
        # get all the agents in the scene except for the robot
        agents = current_world.get_pedestrians()
        max_t = current_world.get_episode_max_time()
        # gather robot information
        robots = list(current_world.get_robots().values())
        # only one robot is supported
        assert(len(robots) == 1)
        robot = robots[0]
        # episode data
        if init_ep:
            # only update start/goal when creating an episode
            r_start = robot.get_start_config().to_3D_numpy()
            r_goal = robot.get_goal_config().to_3D_numpy()
            # creates a new instance of the episode for further use
            self.current_ep = \
                Episode(name, env, agents, max_t, r_start, r_goal)
            print("%sRunning test for %s%s" %
                  (color_yellow, self.current_ep.get_name(), color_reset))
            assert(self.current_ep.get_name() in self.episode_names)
        else:
            # option to update the env and agents in the existing (running) episode
            self.current_ep.update(env, agents)
        # update the delta_t of the simulator, which we wont assume is consistent
        self.sim_dt = current_world.get_delta_t()

    """ END LISTEN UTILS """

    def track_vel_accel(self, current_world):
        assert(self.joystick_params.track_vel_accel)
        # TODO: ensure the sim_states are sorted by their time, so that traversing
        # the hash map maintains relative order of when they entered
        sim_state_list = list(self.sim_states.values())
        sim_t = current_world.get_sim_t()
        self.velocities[sim_t] = velocities(sim_state_list)
        self.accelerations[sim_t] = accelerations(sim_state_list)

    """ BEGIN PANDAS UTILS """

    def update_logs(self, world_state: SimState):
        self.update_log_of_type('robots', world_state)
        self.update_log_of_type('gen_agents', world_state)
        self.update_log_of_type('prerecs', world_state)

    def update_log_of_type(self, agent_type: str, world_state: SimState):
        from simulators.sim_state import get_agents_from_type
        agents_of_type = get_agents_from_type(world_state, agent_type)
        for a in agents_of_type.keys():
            if a not in self.agent_log.keys():
                # initialize dict for a specific agent if dosent already exist
                self.agent_log[a] = {}
            self.agent_log[a][world_state.get_sim_t()] = \
                agents_of_type[a].get_current_config().to_3D_numpy()

    def write_pandas(self):
        assert self.joystick_params.write_pandas_log
        pd_df = pd.DataFrame(self.agent_log)
        abs_path = \
            os.path.join(get_path_to_socnav(), self.dirname, 'agent_data.csv')
        if not os.path.exists(abs_path):
            touch(abs_path)  # Just as the bash command
        pd_df.to_csv(abs_path)
        if self.joystick_params.verbose:
            print("%sUpdated pandas dataframe%s" %
                  (color_green, color_reset))

    """ END PANDAS UTILS """

    """ BEGIN SOCKET UTILS """

    def close_recv_socket(self):
        if self.joystick_on:
            # connect to the socket, closing it, and continuing the thread to completion
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.recv_ID)
            except:
                print("%sClosing listener socket%s" % (color_red, color_reset))
            self.robot_receiver_socket.close()

    def send_to_robot(self, message: str):
        # Create a TCP/IP socket
        self.robot_sender_socket = \
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Connect the socket to the location where the server is listening
        assert(isinstance(message, str))
        try:
            self.robot_sender_socket.connect(self.send_ID)
        except:  # used to turn off the joystick
            return
        # Send data
        self.robot_sender_socket.sendall(bytes(message, "utf-8"))
        self.robot_sender_socket.close()
        if self.joystick_params.print_data:
            print("sent", message)

    def init_send_conn(self):
        """Creates the initial handshake between the joystick and the robot to
        have a communication channel with the external robot process """
        self.robot_sender_socket = \
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.robot_sender_socket.connect(self.send_ID)
        except:
            print("%sUnable to connect to robot%s" % (color_red, color_reset))
            print("Make sure you have a simulation instance running")
            exit(1)
        print("%sRobot <-- Joystick (sender) connection established%s" %
              (color_green, color_reset))
        assert(self.robot_sender_socket)

    def init_recv_conn(self):
        """Creates the initial handshake between the joystick and the meta test
        controller that sends information about the episodes as well as the 
        RobotAgent that sends it's SimStates serialized through json as a 'sense'"""
        self.robot_receiver_socket = \
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.robot_receiver_socket.bind(self.recv_ID)
        # wait for a connection
        self.robot_receiver_socket.listen(1)
        connection, client = self.robot_receiver_socket.accept()
        print("%sRobot --> Joystick (receiver) connection established%s" %
              (color_green, color_reset))
        return connection, client

    """ END SOCKET UTILS """
