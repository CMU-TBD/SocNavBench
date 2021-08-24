import numpy as np
from dotmap import DotMap
from objectives.angle_distance import AngleDistance
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_socnav_params, create_test_map_params
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from utils.socnav_utils import load_building
from utils.utils import color_text


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
    return create_test_map_params(p)


def test_goal_angle_distance() -> None:
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
    # goal_pos_n2 = np.array([[9., 15.]])
    goal_pos_n2 = p.goal_pos_n2
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(
        goal_positions_n2=goal_pos_n2,
        map_size_2=map_size_2,
        dx=0.05,
        map_origin_2=[0.0, 0.0],
        mask_grid_mn=obstacle_occupancy_grid,
    )

    # Define the objective
    objective = AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map)

    # Define a set of positions and evaluate objective
    # pos_nk2 = np.array([[[8., 16.], [8., 12.5], [18., 16.5]]], dtype=np.float32)
    pos_nk2 = p.pos_nk2
    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=pos_nk2)

    # Compute the objective
    objective_values_13 = objective.evaluate_objective(trajectory)
    assert objective_values_13.shape == (1, 3)

    # Expected objective values
    angle_map = fmm_map.fmm_angle_map.voxel_function_mn
    idxs_xy_n2 = pos_nk2[0] / 0.05
    idxs_yx_n2 = idxs_xy_n2[:, ::-1].astype(np.int32)
    expected_angles = np.array(
        [
            angle_map[idxs_yx_n2[0][0], idxs_yx_n2[0][1]],
            angle_map[idxs_yx_n2[1][0], idxs_yx_n2[1][1]],
            angle_map[idxs_yx_n2[2][0], idxs_yx_n2[2][1]],
        ],
        dtype=np.float32,
    )
    expected_objective = 25.0 * abs(expected_angles)

    assert np.allclose(objective_values_13[0], expected_objective, atol=1e-2)
    # hardcoded results to match the given inputs
    assert np.allclose(objective_values_13[0], p.test_goal_ang_obj_ans, atol=1e-2)


def main_test() -> None:
    test_goal_angle_distance()
    print(
        "%sGoal-angle objective tests passed!%s"
        % (color_text["green"], color_text["reset"])
    )


if __name__ == "__main__":
    main_test()
