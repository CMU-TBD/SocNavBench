import numpy as np
from copy import deepcopy
import json
# from simulators.agent import Agent
# from humans.human import Human
from trajectory.trajectory import Trajectory
from utils.utils import *

""" These are smaller "wrapper" classes that are visible by other
gen_agents/humans and saved during state deepcopies
NOTE: they are all READ-ONLY (only getters)
"""


class AgentState():
    def __init__(self, a=None, deepcpy=False):
        if(a):
            self.name = a.get_name()
            self.start_config = a.get_start_config(deepcpy=deepcpy)
            self.goal_config = a.get_goal_config(deepcpy=deepcpy)
            self.current_config = a.get_current_config(deepcpy=deepcpy)
            self.trajectory = a.get_trajectory(deepcpy=deepcpy)
            self.collided = a.get_collided()
            self.end_acting = a.end_acting
            self.collision_cooldown = a.get_collision_cooldown()
            self.radius = a.get_radius()
            self.color = a.get_color()

    def get_name(self):
        return self.name

    def get_current_config(self):
        return self.current_config

    def get_start_config(self):
        return self.start_config

    def get_goal_config(self):
        return self.goal_config

    def get_trajectory(self):
        return self.trajectory

    def get_collided(self):
        return self.collided

    def get_radius(self):
        return self.radius

    def get_color(self):
        return self.color

    def get_collision_cooldown(self):
        return self.collision_cooldown

    def get_pos3(self):
        return self.get_current_config().to_3D_numpy()

    def to_json(self, include_start_goal=False):
        name_json = SimState.to_json_type(deepcopy(self.name))
        # NOTE: the configs are just being serialized with their 3D positions
        if include_start_goal:
            start_json = SimState.to_json_type(
                self.get_start_config().to_3D_numpy())
            goal_json = SimState.to_json_type(
                self.get_goal_config().to_3D_numpy())
        current_json = SimState.to_json_type(
            deepcopy(self.get_current_config().to_3D_numpy()))
        # trajectory_json = "None"
        radius_json = deepcopy(self.radius)
        json_dict = {}
        json_dict['name'] = name_json
        # NOTE: the start and goal (of the robot) are only sent when the environment is sent
        if include_start_goal:
            json_dict['start_config'] = start_json
            json_dict['goal_config'] = goal_json
        json_dict['current_config'] = current_json
        # json_dict['trajectory'] = trajectory_json
        json_dict['radius'] = radius_json
        # returns array (python list) to be json'd in_simstate
        return json_dict

    @ staticmethod
    def from_json(json_str: dict):
        new_state = AgentState()
        new_state.name = json_str['name']
        if('start_config' in json_str.keys()):
            new_state.start_config = \
                generate_config_from_pos_3(json_str['start_config'])
        if('goal_config' in json_str.keys()):
            new_state.goal_config = \
                generate_config_from_pos_3(json_str['goal_config'])
        new_state.current_config = \
            generate_config_from_pos_3(json_str['current_config'])
        new_state.vehicle_trajectory = Trajectory(dt=0.05, n=1, k=0)  # default
        new_state.radius = json_str['radius']
        new_state.collided = False
        new_state.end_acting = False
        new_state.color = None
        return new_state


class HumanState(AgentState):
    def __init__(self, human, deepcpy=False):
        self.appearance = human.get_appearance()
        # Initialize the agent state class
        super().__init__(human, deepcpy=deepcpy)

    def get_appearance(self):
        return self.appearance


class SimState():

    # environment = None

    def __init__(self, environment: dict = None, pedestrians: dict = None,
                 robots: dict = None, sim_t: float = None, wall_t: float = None,
                 delta_t: float = None, episode_name: str = None, max_time: float = None,
                 ped_collider: str = ""):
        self.environment = environment
        # no distinction between prerecorded and auto agents
        self.pedestrians = pedestrians  # new dict that the joystick will be sent
        self.robots = robots
        self.sim_t = sim_t
        self.wall_t = wall_t
        self.delta_t = delta_t
        self.robot_on = True  # TODO: why keep this if not using explicitly?
        self.episode_name = episode_name
        self.episode_max_time = max_time
        self.ped_collider = ped_collider

    def get_environment(self):
        return self.environment

    def get_map(self):
        return self.environment["map_traversible"]

    def get_pedestrians(self):
        return self.pedestrians

    def get_robots(self):
        return self.robots

    def get_robot(self):
        return list(self.robots.values())[0]

    def get_sim_t(self):
        return self.sim_t

    def get_wall_t(self):
        return self.wall_t

    def get_delta_t(self):
        return self.delta_t

    def get_robot_on(self):
        return self.robot_on

    def get_episode_name(self):
        return self.episode_name

    def get_episode_max_time(self):
        return self.episode_max_time

    def get_collider(self):
        return self.ped_collider

    def get_all_agents(self, include_robot=False):
        all_agents = {}
        all_agents.update(self.get_pedestrians())
        if include_robot:
            all_agents.update(self.get_robots())
        return all_agents

    def to_json(self, robot_on=True, send_metadata=False, termination_cause=None):
        json_dict = {}
        json_dict['robot_on'] = deepcopy(robot_on)  # true or false
        sim_t_json = deepcopy(self.get_sim_t())
        if robot_on:  # only send the world if the robot is ON
            if send_metadata:
                environment_json = \
                    SimState.to_json_dict(deepcopy(self.get_environment()))
                episode_json = deepcopy(self.get_episode_name())
                episode_max_time_json = deepcopy(self.get_episode_max_time())
            else:
                environment_json = {}  # empty dictionary
                episode_json = {}
                episode_max_time_json = {}
            # serialize all other fields
            ped_json = \
                SimState.to_json_dict(deepcopy(self.get_pedestrians()))
            # NOTE: the robot only includes its start/goal posn if sending metadata
            robots_json = \
                SimState.to_json_dict(deepcopy(self.get_robots()),
                                      include_start_goal=send_metadata)
            delta_t_json = deepcopy(self.get_delta_t())
            # append them to the json dictionary
            json_dict['environment'] = environment_json
            json_dict['pedestrians'] = ped_json
            json_dict['robots'] = robots_json
            json_dict['delta_t'] = delta_t_json
            json_dict['episode_name'] = episode_json
            json_dict['episode_max_time'] = episode_max_time_json
        else:
            json_dict['termination_cause'] = deepcopy(termination_cause)
        # sim_state should always have time
        json_dict['sim_t'] = sim_t_json
        return json.dumps(json_dict, indent=1)

    @ staticmethod
    def init_agent_dict(json_str_dict):
        agent_dict = {}
        for d in json_str_dict.keys():
            agent_dict[d] = AgentState.from_json(json_str_dict[d])
        return agent_dict

    @ staticmethod
    def from_json(json_str: dict):
        new_state = SimState()
        new_state.environment = json_str['environment']
        new_state.pedestrians = \
            SimState.init_agent_dict(json_str['pedestrians'])
        new_state.robots = SimState.init_agent_dict(json_str['robots'])
        new_state.sim_t: float = json_str['sim_t']
        new_state.delta_t: float = json_str['delta_t']
        new_state.robot_on: bool = json_str['robot_on']
        new_state.episode_name: str = json_str['episode_name']
        new_state.episode_max_time: str = json_str['episode_max_time']
        new_state.wall_t = None
        new_state.ped_collider = ""
        return new_state

    @ staticmethod
    def to_json_type(elem, include_start_goal=False):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            # recursive for dictionaries within dictionaries
            return SimState.to_json_dict(elem, include_start_goal=include_start_goal)
        if isinstance(elem, AgentState):
            return elem.to_json(include_start_goal=include_start_goal)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)

    @ staticmethod
    def to_json_dict(param_dict, include_start_goal=False):
        """ Converts params_dict to a json serializable dict."""
        for key in param_dict.keys():
            param_dict[key] = SimState.to_json_type(
                param_dict[key], include_start_goal=include_start_goal)
        return param_dict


"""BEGIN SimState utils"""


def get_all_agents(sim_state: dict, include_robot=False):
    all_agents = {}
    all_agents.update(get_agents_from_type(sim_state, "pedestrians"))
    if include_robot:
        all_agents.update(get_agents_from_type(sim_state, "robots"))
    return all_agents


def get_agents_from_type(sim_state, agent_type: str):
    if callable(getattr(sim_state, 'get_' + agent_type, None)):
        getter_agent_type = getattr(sim_state, 'get_' + agent_type, None)
        return getter_agent_type()
    return {}  # empty dict


def compute_next_vel(sim_state_prev, sim_state_now, agent_name: str):
    old_agent = sim_state_prev.get_all_agents()[agent_name]
    old_pos = old_agent.get_current_config().to_3D_numpy()
    new_agent = sim_state_now.get_all_agents()[agent_name]
    new_pos = new_agent.get_current_config().to_3D_numpy()
    # calculate distance over time
    delta_t = sim_state_now.get_sim_t() - sim_state_prev.get_sim_t()
    return euclidean_dist2(old_pos, new_pos) / delta_t


def compute_agent_state_velocity(sim_states: list, agent_name: str):
    if(len(sim_states) > 1):  # need at least two to compute differences in positions
        if(agent_name in get_all_agents(sim_states[-1]).keys()):
            agent_velocities = []
            for i in range(len(sim_states)):
                if(i > 0):
                    prev_sim_s = sim_states[i - 1]
                    now_sim_s = sim_states[i]
                    speed = compute_next_vel(prev_sim_s, now_sim_s, agent_name)
                    agent_velocities.append(speed)
                else:
                    agent_velocities.append(0.0)  # initial velocity is 0
            return agent_velocities
        else:
            print("%sAgent" % color_red, agent_name,
                  "is not in the SimStates%s" % color_reset)
    else:
        return []


def compute_agent_state_acceleration(sim_states: list, agent_name: str, velocities: list = None):
    if(len(sim_states) > 1):  # need at least two to compute differences in velocities
        # optionally compute velocities as well
        if(velocities is None):
            velocities = compute_agent_state_velocity(sim_states, agent_name)
        if(agent_name in get_all_agents(sim_states[-1]).keys()):
            agent_accels = []
            for i, this_vel in enumerate(velocities):
                if(i > 0):
                    # compute delta_t between sim states
                    sim_st_now = sim_states[i]
                    sim_st_prev = sim_states[i - 1]
                    delta_t = sim_st_now.get_sim_t() - sim_st_prev.get_sim_t()
                    # compute delta_v between velocities
                    last_vel = velocities[i - 1]
                    # calculate speeds over time
                    accel = (this_vel - last_vel) / delta_t
                    agent_accels.append(accel)
                    if(i == len(sim_states) - 1):
                        # last element gets no acceleration
                        break
            return agent_accels
        else:
            print("%sAgent" % color_red, agent_name,
                  "is not in the SimStates%s" % color_reset)
    else:
        return []


def compute_all_velocities(sim_states: list):
    all_velocities = {}
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert(isinstance(agent_name, str))  # keyed by name
        all_velocities[agent_name] = \
            compute_agent_state_velocity(sim_states, agent_name)
    return all_velocities


def compute_all_accelerations(sim_states: list):
    all_accels = {}
    # TODO: add option of providing precomputed velocities list
    for agent_name in get_all_agents(sim_states[-1]).keys():
        assert(isinstance(agent_name, str))  # keyed by name
        all_accels[agent_name] = compute_agent_state_acceleration(
            sim_states, agent_name)
    return all_accels
