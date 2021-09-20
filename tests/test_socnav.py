import random
from typing import Dict, Tuple

import numpy as np

# Humanav
from agents.humans.human import Human
from agents.humans.recorded_human import PrerecordedHuman
from agents.robot_agent import RobotAgent
from dotmap import DotMap
from params.central_params import (
    create_datasets_params,
    create_episodes_params,
    create_robot_params,
    create_socnav_params,
    get_seed,
)

# Planner + Simulator:
from simulators.simulator import Simulator
from socnav.socnav_renderer import SocNavRenderer
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
    p.robot_params = create_robot_params()

    # Introduce the episode params
    p.episode_params = create_episodes_params()

    # not testing robot, only simulator + agents
    p.episode_params.without_robot = False

    # overwrite tests with custom basic test
    p.episode_params.tests = {}
    default_name = "test_socnav_univ"
    p.episode_params.tests[default_name] = DotMap(
        name=default_name,
        map_name="Univ",
        pedestrian_datasets=create_datasets_params(["univ"]),
        datasets_start_t=[0.0],
        ped_ranges=[(0, 100)],
        #    agents_start=[[8, 8, 0]], agents_end=[[17.5, 13, 0.]],
        agents_start=[],
        agents_end=[],
        robot_start_goal=[[2.5, 4, 0], [15.5, 8, 0.7]],
        max_time=30,
        write_episode_log=True,
    )

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.physical_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth if the host has an available display
        p.camera_params.modalities = ["rgb", "disparity"]
    else:
        p.camera_params.modalities = ["occupancy_grid"]
    return p


def test_socnav() -> None:
    """
    Code for loading random humans into the environment
    and rendering topview, rgb, and depth images.
    """
    p: DotMap = create_params()  # used to instantiate the camera and its parameters

    RobotAgent.establish_joystick_handshake(p)

    for test in list(p.episode_params.tests.keys()):
        episode: DotMap = p.episode_params.tests[test]
        """Create the environment and renderer for the episode"""
        env_r = construct_environment(p, test, episode)
        environment: Dict[str, float or int or np.ndarray] = env_r[0]
        r: SocNavRenderer = env_r[1]

        """
        Creating planner, simulator, and control pipelines for the framework
        of a human trajectory and pathfinding.
        """
        simulator = Simulator(environment, renderer=r, episode_params=episode)
        simulator.params.render_params.draw_parallel_robots = False # force false
        """Generate the autonomous human agents from the episode"""
        new_humans = Human.generate_humans(p, episode.agents_start, episode.agents_end)
        simulator.add_agents(new_humans)

        """Add the prerecorded humans to the simulator"""
        for i, dataset in enumerate(episode.pedestrian_datasets):
            dataset_start_t: float = episode.datasets_start_t[i]
            dataset_ped_range: Tuple[int, int] = episode.ped_ranges[i]
            new_prerecs = PrerecordedHuman.generate_humans(
                p,
                max_time=episode.max_time,
                start_t=dataset_start_t,
                ped_range=dataset_ped_range,
                dataset=dataset,
            )
            simulator.add_agents(new_prerecs)

        """Generate the robot(s) for the simulator"""
        if not p.episode_params.without_robot:
            if len(episode.robot_start_goal) == 0:
                # randomly generate robot
                robot_agent = RobotAgent.random_from_environment(environment)
                simulator.add_agent(robot_agent)
            else:
                robot_agent = RobotAgent.generate_robot(episode.robot_start_goal)
                simulator.add_agent(robot_agent)

        # run simulation & render
        simulator.simulate()
        simulator.render(r, filename=episode.name + "_obs")

    if not p.episode_params.without_robot:
        RobotAgent.close_robot_sockets()


if __name__ == "__main__":
    # run basic room test with variable # of human
    test_socnav()
