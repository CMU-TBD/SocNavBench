import numpy as np
import matplotlib.pyplot as plt
from obstacles.sbpd_map import SBPDMap
from objectives.personal_space_cost import PersonalSpaceCost
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.objective_function import ObjectiveFunction
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from utils.utils import *

from params.simulator.sbpd_simulator_params import create_params as create_sim_params
from params.renderer_params import get_seed
from params.renderer_params import create_params as create_base_params
from simulators.central_simulator import CentralSimulator

from humans.human import Human
from humanav.humanav_renderer_multi import HumANavRendererMulti

from simulators.sim_state import SimState, get_pos3


def create_params():
    p = create_base_params()

    # Set any custom parameters

    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.
    p.camera_params.fov_horizontal = 75.

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See humanav/renderer_params.py for more information

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth then host pc has an available display
        p.camera_params.modalities = ['rgb', 'disparity']
    else:
        p.camera_params.modalities = ['occupancy_grid']

    return p


def create_renderer_params():
    from params.central_params import get_traversible_dir, get_sbpd_data_dir, create_base_params
    p = DotMap()
    p.dataset_name = 'sbpd'
    p.building_name = create_base_params().building_name
    p.flip = False

    p.camera_params = DotMap(modalities=['occupancy_grid'],  # occupancy_grid, rgb, or depth
                             width=64,
                             height=64)

    # The robot is modeled as a solid cylinder
    # of height, 'height', with radius, 'radius',
    # base at height 'base' above the ground
    # The robot has a camera at height
    # 'sensor_height' pointing at
    # camera_elevation_degree degrees vertically
    # from the horizontal plane.
    p.robot_params = DotMap(radius=18,
                            base=10,
                            height=100,
                            sensor_height=80,
                            camera_elevation_degree=-45,  # camera tilt
                            delta_theta=1.0)

    # Traversible dir
    p.traversible_dir = get_traversible_dir()

    # SBPD Data Directory
    p.sbpd_data_dir = get_sbpd_data_dir()

    return p


def create_test_params():
    p = DotMap()
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=.5,
                                        power=2,
                                        obstacle_cost=25.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=25.0)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=25.0,
                                       goal_margin=0.0)

    p.objective_fn_params = DotMap(obj_type='mean')
    p.obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                   map_origin_2=[0, 0],
                                   sampling_thres=2,
                                   plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()

    p.personal_space_params = DotMap(
        power=1,
        psc_scale=10
    )

    return p


def test_personal_cost_function(sim_state: SimState, plot=False):
    """
    Creating objective points maually, plotting them in the ObjectiveFunction
    class, and then asserting that combined, their sum adds up to the same
    objective cost as the sum of the individual trajectories
    """
    # Create parameters
    p = create_test_params()
    from humanav.humanav_renderer_multi import HumANavRendererMulti
    r = HumANavRendererMulti.get_renderer(
        p.obstacle_map_params.renderer_params, deepcpy=False)
    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()

    obstacle_map = SBPDMap(p.obstacle_map_params,
                           renderer=0, res=dx_cm, trav=traversible)
    # obstacle_map = SBPDMap(p.obstacle_map_params)
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
    map_size_2 = obstacle_occupancy_grid.shape[::-1]

    # Define a goal position and compute the corresponding fmm map
    goal_pos_n2 = np.array([[9., 15]])
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                           map_size_2=map_size_2,
                                                           dx=0.05,
                                                           map_origin_2=[
                                                               0., 0.],
                                                           mask_grid_mn=obstacle_occupancy_grid)
    # Define the cost function
    objective_function = ObjectiveFunction(p.objective_fn_params)
    objective_function.add_objective(ObstacleAvoidance(
        params=p.avoid_obstacle_objective, obstacle_map=obstacle_map))
    objective_function.add_objective(GoalDistance(
        params=p.goal_distance_objective, fmm_map=fmm_map))
    objective_function.add_objective(AngleDistance(
        params=p.goal_angle_objective, fmm_map=fmm_map))

    # Define each objective separately
    objective1 = ObstacleAvoidance(
        params=p.avoid_obstacle_objective, obstacle_map=obstacle_map)
    objective2 = GoalDistance(
        params=p.goal_distance_objective, fmm_map=fmm_map)
    objective3 = AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map)

    # Define cost function for personal state
    objectiveP = PersonalSpaceCost(
        params=p.personal_space_objective)

    # Define a set of positions and evaluate objective
    pos_nk2 = np.array(
        [[[8., 12.5], [8., 16.], [18., 16.5]]], dtype=np.float32)
    heading_nk2 = np.array([[[np.pi / 2.0], [0.1], [0.1]]], dtype=np.float32)
    trajectory = Trajectory(
        dt=0.1, n=1, k=3, position_nk2=pos_nk2, heading_nk1=heading_nk2)

    # Compute the objective function
    values_by_objective = objective_function.evaluate_function_by_objective(
        trajectory)
    overall_objective = objective_function.evaluate_function(trajectory)

    # Expected objective values
    expected_objective1 = objective1.evaluate_objective(trajectory)
    expected_objective2 = objective2.evaluate_objective(trajectory)
    expected_objective3 = objective3.evaluate_objective(trajectory)
    expected_overall_objective = np.mean(
        expected_objective1 + expected_objective2 + expected_objective3, axis=1)
    assert len(values_by_objective) == 3
    assert values_by_objective[0][1].shape == (1, 3)
    assert overall_objective.shape == (1,)
    # assert np.allclose(overall_objective.numpy(), expected_overall_objective.numpy(), atol=1e-2)
    assert np.allclose(overall_objective,
                       expected_overall_objective, atol=1e-2)

    # Use sim_state from main
    ps_cost = objectiveP.evaluate_objective(trajectory, sim_state)
    print("Personal space cost ", ps_cost)
    print("Obstacle avoidance cost", expected_objective1)
    print("Goal distance cost", expected_objective2)
    print("Angle distance cost", expected_objective3)

    # Optionally visualize the traversable and the points on which
    # we compute the objective function
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        obstacle_map.render(ax)

        # plot agent start
        ax.plot(pos_nk2[0, :, 0], pos_nk2[0, :, 1], 'r.')
        # plot agent goal
        ax.plot(goal_pos_n2[0, 0], goal_pos_n2[0, 1], 'k*')

        agents = sim_state.get_all_agents()

        for agent_name, agent_vals in agents.items():
            agent_pos3 = get_pos3(agent_vals)  # (x,y,th)
            theta = agent_pos3[2]
            ax.plot(agent_pos3[0], agent_pos3[1], 'g.')

        # plot non ego agents
        fig.savefig('../test_psc_function.png',
                    bbox_inches='tight', pad_inches=0)


def main_test():
    p = create_params()  # used to instantiate the camera and its parameters
    # TODO: can optimize HumANavRendererMulti renderer when not rendering humans
    # get the renderer from the camera p
    r = HumANavRendererMulti.get_renderer(p, deepcpy=False)
    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()

    #  TODO this env dict creation should be some sort of function
    environment = {}
    environment["map_scale"] = 5 / 100  # dx_m
    environment["room_center"] = np.array([14, 14., 0.])  # room_center
    environment["traversibles"] = np.array([traversible])

    # construct simulator
    sim_params = create_sim_params(render_3D=False)
    simulator = CentralSimulator(sim_params, environment)

    # generate and add random humans
    human_list = []
    for i in range(4):
        # Generates a random human from the environment
        new_human_i = Human.generate_random_human_from_environment(
            environment,
            generate_appearance=False,
            radius=5
        )
        # Or specify a human's initial configs with a HumanConfig instance
        # Human.generate_human_with_configs(Human, fixed_start_goal)
        human_list.append(new_human_i)

        # Load a random human at a specified state and speed
        # update human traversible
        environment["traversibles"] = np.array([traversible])

        # Input human fields into simulator
        simulator.add_agent(new_human_i)
        print("Generated Random Humans:", i + 1, "\r", end="")

    # get initial state
    sim_delT = 0.05
    sim_state = simulator.save_state(0, sim_delT, 0)
    test_personal_cost_function(sim_state, plot=True)
    print("%sCost function tests passed!%s" % (color_green, color_reset))


if __name__ == '__main__':
    main_test()
