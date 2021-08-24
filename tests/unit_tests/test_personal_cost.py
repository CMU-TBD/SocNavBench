import matplotlib.pyplot as plt
import numpy as np
from agents.humans.human import Human
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.objective_function import ObjectiveFunction
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.personal_space_cost import PersonalSpaceCost
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_socnav_params, create_test_map_params
from simulators.sim_state import SimState
from simulators.simulator import Simulator
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from utils.socnav_utils import construct_environment, load_building
from utils.utils import *


def create_params() -> DotMap:
    p = create_socnav_params()

    p.render_3D = False  # only test without renderer
    p.building_params.load_meshes = False  # don't load any meshes

    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1, angle_cost=25.0)
    p.obstacle_map_params = DotMap(
        obstacle_map=SBPDMap,
        map_origin_2=[0.0, 0.0],
        sampling_thres=2,
        plotting_grid_steps=100,
    )
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(
        obstacle_margin0=0.3, obstacle_margin1=0.5, power=2, obstacle_cost=25.0
    )
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1, angle_cost=25.0)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2, goal_cost=25.0, goal_margin=0.0)

    p.objective_fn_params = DotMap(obj_type="mean")
    p.obstacle_map_params = DotMap(
        obstacle_map=SBPDMap,
        map_origin_2=[0, 0],
        sampling_thres=2,
        plotting_grid_steps=100,
    )

    p.personal_space_params = DotMap(power=1, psc_scale=10)
    # Introduce the robot params
    from params.central_params import create_robot_params

    p.robot_params = create_robot_params()

    # Introduce the episode params
    from params.central_params import create_datasets_params, create_episodes_params

    p.episode_params = create_episodes_params()

    # not testing robot, only simulator + agents
    p.episode_params.without_robot = True

    # overwrite tests with custom basic test
    p.episode_params.tests = {}
    default_name = "test_psc"
    p.episode_params.tests[default_name] = DotMap(
        name=default_name,
        map_name="Univ",
        pedestrian_datasets=create_datasets_params([]),
        datasets_start_t=[],
        ped_ranges=[],
        agents_start=[],
        agents_end=[],
        robot_start_goal=[[10, 3, 0], [15.5, 8, 0.7]],
        max_time=30,
        write_episode_log=False,
    )
    # definitely wont be rendering this
    p.render_3D = False
    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.physical_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth if the host has an available display
        p.camera_params.modalities = ["rgb", "disparity"]
    else:
        p.camera_params.modalities = ["occupancy_grid"]

    return create_test_map_params(p)


def test_personal_cost_function(sim_state: SimState, plot=False, verbose=False) -> None:
    """
    Creating objective points maually, plotting them in the ObjectiveFunction
    class, and then asserting that combined, their sum adds up to the same
    objective cost as the sum of the individual trajectories
    """
    # Create parameters
    p = create_params()
    r, dx_cm, traversible = load_building(p)

    obstacle_map = SBPDMap(
        p.obstacle_map_params, renderer=0, res=dx_cm, map_trav=traversible
    )
    # obstacle_map = SBPDMap(p.obstacle_map_params)
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
    map_size_2 = obstacle_occupancy_grid.shape[::-1]

    # Define a goal position and compute the corresponding fmm map
    goal_pos_n2 = np.array([[9.0, 10]])
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(
        goal_positions_n2=goal_pos_n2,
        map_size_2=map_size_2,
        dx=0.05,
        map_origin_2=[0.0, 0.0],
        mask_grid_mn=obstacle_occupancy_grid,
    )
    # Define the cost function
    objective_function = ObjectiveFunction(p.objective_fn_params)
    objective_function.add_objective(
        ObstacleAvoidance(params=p.avoid_obstacle_objective, obstacle_map=obstacle_map)
    )
    objective_function.add_objective(
        GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map)
    )
    objective_function.add_objective(
        AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map)
    )

    # Define each objective separately
    objective1 = ObstacleAvoidance(
        params=p.avoid_obstacle_objective, obstacle_map=obstacle_map
    )
    objective2 = GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map)
    objective3 = AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map)

    # Define cost function for personal state
    objectiveP = PersonalSpaceCost(params=p.personal_space_params)

    # Define a set of positions and evaluate objective
    pos_nk2 = np.array([[[8.0, 12.5], [8.0, 16.0], [18.0, 16.5]]], dtype=np.float32)
    heading_nk2 = np.array([[[np.pi / 2.0], [0.1], [0.1]]], dtype=np.float32)
    trajectory = Trajectory(
        dt=0.1, n=1, k=3, position_nk2=pos_nk2, heading_nk1=heading_nk2
    )

    # Compute the objective function
    values_by_objective = objective_function.evaluate_function_by_objective(trajectory)
    overall_objective = objective_function.evaluate_function(trajectory)

    # Expected objective values
    expected_objective1 = objective1.evaluate_objective(trajectory)
    expected_objective2 = objective2.evaluate_objective(trajectory)
    expected_objective3 = objective3.evaluate_objective(trajectory)
    expected_overall_objective = np.mean(
        expected_objective1 + expected_objective2 + expected_objective3, axis=1
    )
    assert len(values_by_objective) == 3
    assert values_by_objective[0][1].shape == (1, 3)
    assert overall_objective.shape == (1,)
    # assert np.allclose(overall_objective.numpy(), expected_overall_objective.numpy(), atol=1e-2)
    assert np.allclose(overall_objective, expected_overall_objective, atol=1e-2)

    # Use sim_state from main
    sim_state_hist = {}
    sim_state_hist[0] = sim_state
    ps_cost = objectiveP.evaluate_objective(trajectory, sim_state_hist)
    if verbose:
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
        ax.plot(pos_nk2[0, :, 0], pos_nk2[0, :, 1], "r.")
        # plot agent goal
        ax.plot(goal_pos_n2[0, 0], goal_pos_n2[0, 1], "k*")

        agents = sim_state.get_all_agents()

        for agent_name, agent_vals in agents.items():
            agent_pos3 = agent_vals.get_pos3()  # (x,y,th)
            theta = agent_pos3[2]
            ax.plot(agent_pos3[0], agent_pos3[1], "g.")

        # plot non ego agents
        fig.savefig("../test_psc_function.png", bbox_inches="tight", pad_inches=0)


def main_test() -> None:
    p = create_params()  # used to instantiate the camera and its parameters
    test = "test_psc"
    episode = p.episode_params.tests[test]

    environment, r = construct_environment(p, test, episode, verbose=False)

    # construct simulator
    simulator = Simulator(
        environment, renderer=r, episode_params=episode, verbose=False
    )
    # generate and add random humans
    for _ in range(4):
        # Generates a random human from the environment
        new_human_i = Human.generate_human(
            environment=environment, generate_appearance=False
        )
        simulator.add_agent(new_human_i)
    # initialize simulator fields
    simulator.init_sim_data(verbose=False)
    # get initial state
    sim_state = simulator.save_state(0)
    test_personal_cost_function(sim_state, plot=True)
    print("%sGoal-psc tests passed!%s" % (color_text["green"], color_text["reset"]))


if __name__ == "__main__":
    main_test()
