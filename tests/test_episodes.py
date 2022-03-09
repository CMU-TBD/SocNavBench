import random
from typing import Dict, Tuple

import numpy as np

# Humanav
from agents.humans.human import Human
from agents.humans.recorded_human import PrerecordedHuman
from agents.robot_agent import RobotAgent
from dotmap import DotMap
from params.central_params import create_socnav_params, get_seed

# Planner + Simulator:
from simulators.simulator import Simulator
from socnav.socnav_renderer import SocNavRenderer
from trajectory.trajectory import SystemConfig
from utils.socnav_utils import construct_environment

# seed the random number generator
random.seed(get_seed())


def create_params() -> DotMap:
    p: DotMap = create_socnav_params()

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
        p.camera_params.modalities = ["rgb", "disparity"]
    else:
        p.camera_params.modalities = ["occupancy_grid"]

    return p


def test_episodes() -> None:
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p: DotMap = create_params()  # used to instantiate the camera and its parameters

    RobotAgent.establish_joystick_handshake(p)

    for test in list(p.episode_params.tests.keys()):
        episode = p.episode_params.tests[test]

        """Create the environment and renderer for the episode"""
        env_r = construct_environment(p, test, episode)
        environment: Dict[str, float or int or np.ndarray] = env_r[0]
        r: SocNavRenderer = env_r[1]

        """
        Creating planner, simulator, and control pipelines for the framework
        of a human trajectory and pathfinding. 
        """
        simulator = Simulator(environment, renderer=r, episode_params=episode)
        simulator.params.render_params.draw_parallel_robots = False  # force false

        """Generate the autonomous human agents from the episode"""
        new_humans = Human.generate_humans(p, episode.agents_start, episode.agents_end)
        simulator.add_agents(new_humans)

        """Generate the robot in the simulator"""
        if not p.episode_params.without_robot:
            robot_agent = RobotAgent.generate_robot(episode.robot_start_goal)
            simulator.add_agent(robot_agent)

        """Add the prerecorded humans to the simulator"""
        for i, dataset in enumerate(episode.pedestrian_datasets):
            dataset_start_t = episode.datasets_start_t[i]
            dataset_ped_range = episode.ped_ranges[i]
            new_prerecs = PrerecordedHuman.generate_humans(
                p,
                max_time=episode.max_time,
                start_t=dataset_start_t,
                ped_range=dataset_ped_range,
                dataset=dataset,
            )
            simulator.add_agents(new_prerecs)

        # run simulation & render
        simulator.simulate()
        simulator.render(r, filename=episode.name + "_obs")

    if not p.episode_params.without_robot:
        RobotAgent.close_robot_sockets()


if __name__ == "__main__":
    # run basic room test with variable # of human
    test_episodes()
