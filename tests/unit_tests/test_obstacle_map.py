import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
from trajectory.trajectory import Trajectory
from obstacles.sbpd_map import SBPDMap
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
    p.obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                   map_origin_2=[0., 0.],
                                   sampling_thres=2,
                                   plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()

    return p


def test_sbpd_map(visualize=False):
    np.random.seed(seed=1)

    # Define a set of positions and evaluate objective
    pos_nk2 = np.array(
        [[[8., 16.], [8., 12.5], [18., 16.5]]], dtype=np.float32)
    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=pos_nk2)

    p = create_params()

    # Create an SBPD Map
    from socnav.socnav_renderer import SocNavRenderer
    r = SocNavRenderer.get_renderer(
        p.obstacle_map_params.renderer_params, deepcpy=False)
    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()

    obstacle_map = SBPDMap(p.obstacle_map_params,
                           renderer=0, res=dx_cm, trav=traversible)

    obs_dists_nk = obstacle_map.dist_to_nearest_obs(trajectory.position_nk2())

    assert(np.allclose(obs_dists_nk[0], p.test_obst_map_ans))

    if visualize:
        #occupancy_grid_nn = obstacle_map.create_occupancy_grid(trajectory.position_nk2())

        fig = plt.figure()
        ax = fig.add_subplot(121)
        obstacle_map.render(ax)

        ax = fig.add_subplot(122)
        # ax.imshow(occupancy_grid_nn, cmap='gray', origin='lower')
        ax.set_axis_off()
        # plt.show()
        fig.savefig('./tests/obstacles/test_obstacle_map.png',
                    bbox_inches='tight', pad_inches=0)


def main_test():
    test_sbpd_map(visualize=False)
    print("%sObstacle map tests passed!%s" % (color_green, color_reset))


if __name__ == '__main__':
    main_test()
