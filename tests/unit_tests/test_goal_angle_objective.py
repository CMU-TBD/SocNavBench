import numpy as np
from obstacles.sbpd_map import SBPDMap
from objectives.angle_distance import AngleDistance
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from dotmap import DotMap
from utils.utils import *
from params.central_params import create_map_params


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


def create_params():
    p = create_map_params()
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=25.0)
    p.obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                   map_origin_2=[0., 0.],
                                   sampling_thres=2,
                                   plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()

    return p


def test_goal_angle_distance():
    # Create parameters
    p = create_params()

    # Create an SBPD Map
    from socnav.socnav_renderer import SocNavRenderer
    r = SocNavRenderer.get_renderer(
        p.obstacle_map_params.renderer_params, deepcpy=False)
    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()

    obstacle_map = SBPDMap(p.obstacle_map_params,
                           renderer=0, res=dx_cm, map_trav=traversible)
    # obstacle_map = SBPDMap(p.obstacle_map_params)
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
    map_size_2 = obstacle_occupancy_grid.shape[::-1]

    # Define a goal position and compute the corresponding fmm map
    # goal_pos_n2 = np.array([[9., 15.]])
    goal_pos_n2 = p.goal_pos_n2
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                           map_size_2=map_size_2,
                                                           dx=0.05,
                                                           map_origin_2=[
                                                               0., 0.],
                                                           mask_grid_mn=obstacle_occupancy_grid)

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
    idxs_xy_n2 = pos_nk2[0] / .05
    idxs_yx_n2 = idxs_xy_n2[:, ::-1].astype(np.int32)
    expected_angles = np.array([angle_map[idxs_yx_n2[0][0], idxs_yx_n2[0][1]],
                                angle_map[idxs_yx_n2[1][0], idxs_yx_n2[1][1]],
                                angle_map[idxs_yx_n2[2][0], idxs_yx_n2[2][1]]],
                               dtype=np.float32)
    expected_objective = 25. * abs(expected_angles)

    assert np.allclose(objective_values_13[
                       0], expected_objective, atol=1e-2)
    # hardcoded results to match the given inputs
    assert np.allclose(
        objective_values_13[0], p.test_goal_ang_obj_ans, atol=1e-2)


def main_test():
    test_goal_angle_distance()
    print("%sGoal-angle objective tests passed!%s" %
          (color_green, color_reset))


if __name__ == '__main__':
    main_test()
