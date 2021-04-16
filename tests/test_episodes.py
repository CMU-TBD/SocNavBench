import numpy as np
import random
# Humanav
from agents.humans.human import Human
from agents.robot_agent import RobotAgent
from agents.humans.recorded_human import PrerecordedHuman
# Planner + Simulator:
from simulators.simulator import Simulator
from params.central_params import get_seed, create_socnav_params
from utils.utils import construct_environment

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
    from params.central_params import create_episodes_params
    p.episode_params = create_episodes_params()

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.physical_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth if the host has an available display
        p.camera_params.modalities = ['rgb', 'disparity']
    else:
        p.camera_params.modalities = ['occupancy_grid']

    return p


def test_episodes():
    """
    Code for loading a random human into the environment
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
        Human.generate(simulator, p, episode.agents_start, episode.agents_end,
                       environment, r)

        """Generate the robot in the simulator"""
        if not p.episode_params.without_robot:
            RobotAgent.generate(simulator, p, episode.robot_start_goal)

        """Add the prerecorded humans to the simulator"""
        for i, dataset in enumerate(episode.pedestrian_datasets):
            dataset_start_t = episode.datasets_start_t[i]
            dataset_ped_range = episode.ped_ranges[i]
            PrerecordedHuman.generate(simulator, p, environment, r,
                                      max_time=episode.max_time,
                                      start_t=dataset_start_t,
                                      ped_range=dataset_ped_range,
                                      dataset=dataset
                                      )

        # run simulation
        simulator.simulate()
        # render the simulation result
        simulator.render(r, None, filename=episode.name + "_obs")

    if not p.episode_params.without_robot:
        RobotAgent.close_robot_sockets()


if __name__ == '__main__':
    # run basic room test with variable # of human
    test_episodes()
