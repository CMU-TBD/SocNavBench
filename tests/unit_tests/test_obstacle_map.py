import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_socnav_params, create_test_map_params
from trajectory.trajectory import Trajectory
from utils.socnav_utils import load_building
from utils.utils import color_text


def create_params() -> DotMap:
    p = create_socnav_params()

    p.render_3D = False  # only test without renderer
    p.building_params.load_meshes = False  # don't load any meshes

    p.obstacle_map_params = DotMap(
        obstacle_map=SBPDMap,
        map_origin_2=[0.0, 0.0],
        sampling_thres=2,
        plotting_grid_steps=100,
    )
    return create_test_map_params(p)


def test_sbpd_map(visualize=False) -> None:
    np.random.seed(seed=1)
    p = create_params()

    r, dx_cm, traversible = load_building(p)

    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=p.pos_nk2)

    obstacle_map = SBPDMap(
        p.obstacle_map_params, renderer=0, res=dx_cm, map_trav=traversible
    )

    obs_dists_nk = obstacle_map.dist_to_nearest_obs(trajectory.position_nk2())

    assert np.allclose(obs_dists_nk[0], p.test_obst_map_ans)

    if visualize:
        # occupancy_grid_nn = obstacle_map.create_occupancy_grid(trajectory.position_nk2())

        fig = plt.figure()
        ax = fig.add_subplot(121)
        obstacle_map.render(ax)

        ax = fig.add_subplot(122)
        # ax.imshow(occupancy_grid_nn, cmap='gray', origin='lower')
        ax.set_axis_off()
        # plt.show()
        fig.savefig(
            "./tests/obstacles/test_obstacle_map.png", bbox_inches="tight", pad_inches=0
        )


def main_test() -> None:
    test_sbpd_map(visualize=False)
    print("%sObstacle map tests passed!%s" % (color_text["green"], color_text["reset"]))


if __name__ == "__main__":
    main_test()
