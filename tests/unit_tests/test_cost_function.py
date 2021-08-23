import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.objective_function import ObjectiveFunction
from objectives.obstacle_avoidance import ObstacleAvoidance
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_socnav_params, create_test_map_params
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from utils.socnav_utils import load_building
from utils.utils import color_text


def create_params() -> DotMap:
    p = create_socnav_params()
    
    p.render_3D = False # only test without renderer
    p.building_params.load_meshes = False # don't load any meshes

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
    return create_test_map_params(p)


def test_cost_function(plot=False) -> None:
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
    goal_pos_n2 = p.goal_pos_n2
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

    # Define a set of positions and evaluate objective
    pos_nk2 = p.pos_nk2
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
    # expected_overall_objective = tf.reduce_mean(
    #     expected_objective1 + expected_objective2 + expected_objective3, axis=1)
    expected_overall_objective = np.mean(
        expected_objective1 + expected_objective2 + expected_objective3, axis=1
    )
    assert len(values_by_objective) == 3
    assert values_by_objective[0][1].shape == (1, 3)
    assert overall_objective.shape == (1,)
    # assert np.allclose(overall_objective.numpy(), expected_overall_objective.numpy(), atol=1e-2)
    assert np.allclose(overall_objective, expected_overall_objective, atol=1e-2)

    # Optionally visualize the traversable and the points on which
    # we compute the objective function
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        obstacle_map.render(ax)
        ax.plot(pos_nk2[0, :, 0].numpy(), pos_nk2[0, :, 1].numpy(), "r.")
        ax.plot(goal_pos_n2[0, 0], goal_pos_n2[0, 1], "k*")
        fig.savefig(
            "./tests/cost/test_cost_function.png", bbox_inches="tight", pad_inches=0
        )


def main_test() -> None:
    test_cost_function(plot=False)
    print(
        "%sCost function tests passed!%s" % (color_text["green"], color_text["reset"])
    )


if __name__ == "__main__":
    main_test()
