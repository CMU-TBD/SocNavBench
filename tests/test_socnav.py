import numpy as np
import random
from dotmap import DotMap
# Humanav
from agents.humans.human import Human
from agents.humans.recorded_human import PrerecordedHuman
from agents.robot_agent import RobotAgent
# Planner + Simulator:
from simulators.simulator import Simulator
from params.central_params import get_seed, create_socnav_params
from utils.utils import construct_environment, generate_config_from_pos_3

# seed the random number generator
random.seed(get_seed())


def create_params():
    p = create_socnav_params()

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See params/central_params.py for more information
    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.0
    p.camera_params.fov_horizontal = 75.0

    # Introduce the robot params
    from params.central_params import create_robot_params
    p.robot_params = create_robot_params()

    # Introduce the episode params
    from params.central_params import create_episodes_params, create_datasets_params
    p.episode_params = create_episodes_params()

    # not testing robot, only simulator + agents
    p.episode_params.without_robot = False

    # overwrite tests with custom basic test
    p.episode_params.tests = {}
    default_name = "test_socnav_univ"
    p.episode_params.tests[default_name] = \
        DotMap(name=default_name,
               map_name='Univ',
               pedestrian_datasets=create_datasets_params(["univ"]),
               datasets_start_t=[0.],
               ped_ranges=[(0, 100)],
               #    agents_start=[[8, 8, 0]], agents_end=[[17.5, 13, 0.]],
               agents_start=[], agents_end=[],
               robot_start_goal=[[10, 3, 0], [15.5, 8, 0.7]],
               max_time=30,
               write_episode_log=True
               )

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.physical_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth if the host has an available display
        p.camera_params.modalities = ['rgb', 'disparity']
    else:
        p.camera_params.modalities = ['occupancy_grid']
    return p


def generate_auto_humans(starts, goals, simulator, environment, p, r):
    """
    Generate and add num_humans number of randomly generated humans to the simulator
    """
    num_gen_humans = min(len(starts), len(goals))
    print("Generated Auto Humans:", num_gen_humans)
    from agents.humans.human_configs import HumanConfigs
    for i in range(num_gen_humans):
        start_config = generate_config_from_pos_3(starts[i])
        goal_config = generate_config_from_pos_3(goals[i])
        start_goal_configs = HumanConfigs(start_config, goal_config)
        # Generates a random human from the environment
        new_human_i = Human.generate_human_with_configs(
            start_goal_configs,
            generate_appearance=p.render_3D
        )
        # update renderer and get human traversible if it exists
        if p.render_3D:
            r.add_human(new_human_i)
            environment["human_traversible"] = \
                np.array(r.get_human_traversible())

        # Input human fields into simulator
        simulator.add_agent(new_human_i)


def test_socnav():
    """
    Code for loading random humans into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params()  # used to instantiate the camera and its parameters

    RobotAgent.establish_joystick_handshake(p)

    for test in list(p.episode_params.tests.keys()):
        episode = p.episode_params.tests[test]
        """Create the environment and renderer for the episode"""
        environment, r = construct_environment(p, test, episode)
        """
        Creating planner, simulator, and control pipelines for the framework
        of a human trajectory and pathfinding.
        """
        simulator = Simulator(environment, renderer=r, episode_params=episode)
        """Generate the autonomous human agents from the episode"""
        generate_auto_humans(episode.agents_start, episode.agents_end,
                             simulator, environment, p, r)

        """Add the prerecorded humans to the simulator"""
        for i, dataset in enumerate(episode.pedestrian_datasets):
            dataset_start_t = episode.datasets_start_t[i]
            dataset_ped_range = episode.ped_ranges[i]
            PrerecordedHuman.generate_pedestrians(simulator, p,
                                                  max_time=episode.max_time,
                                                  start_t=dataset_start_t,
                                                  ped_range=dataset_ped_range,
                                                  dataset=dataset
                                                  )

        """Generate the robot(s) for the simulator"""
        if not p.episode_params.without_robot:
            if(len(episode.robot_start_goal) == 0):
                # randomly generate robot
                robot_agent = RobotAgent.random_from_environment(environment)
            else:
                # create constant start/goal for testing purposes
                r_start = episode.robot_start_goal[0]
                r_goal = episode.robot_start_goal[1]
                from agents.humans.human_configs import HumanConfigs
                start_conf = generate_config_from_pos_3(r_start)
                goal_conf = generate_config_from_pos_3(r_goal)
                configs = HumanConfigs(start_conf, goal_conf)
                robot_agent = RobotAgent.generate_robot(configs)
            simulator.add_agent(robot_agent)
        # run simulation
        simulator.simulate()
        # render the simulation result
        simulator.render(r, None, filename=episode.name + "_obs")

    if not p.episode_params.without_robot:
        RobotAgent.close_robot_sockets()


if __name__ == '__main__':
    # run basic room test with variable # of human
    test_socnav()
