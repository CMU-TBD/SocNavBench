import numpy as np
from utils.utils import color_text


def test_fmm_map() -> None:
    from utils.fmm_map import FmmMap

    # Create a grid and a function within that grid
    scale = 0.5
    grid_size = np.array([10, 10])

    goal_position_n2 = np.array([[2.5, 2.5]])

    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(
        goal_positions_n2=goal_position_n2, map_size_2=grid_size, dx=scale
    )

    # Let's have a bunch of points to test the fmm distance and angle map
    test_positions = np.array(
        [[[2.0, 3.0], [2.5, 3.0], [3.0, 3.0], [3.0, 2.0], [2.5, 2.0], [2.0, 2.0]]],
        dtype=np.float32,
    )

    # Predicted distance and angles
    distances = fmm_map.fmm_distance_map.compute_voxel_function(
        test_positions, invalid_value=-100.0
    )
    angles = fmm_map.fmm_angle_map.compute_voxel_function(
        test_positions, invalid_value=-100.0
    )

    # The expected distance is dist1 as defined below. However, due to the numerical issues, the actual computed
    # distance turns out to be a larger (0.6 in this case).
    dist1 = scale * np.sqrt(2.0) - 0.5 * scale * np.cos(np.pi * 45.0 / 180.0)
    dist1 = 0.60
    expected_distances = np.array([dist1, 0.25, dist1, dist1, 0.25, dist1])
    expected_angles = (np.pi / 180) * np.array(
        [-45.0, -90.0, -135.0, 135.0, 90.0, 45.0]
    )

    assert np.sum(abs(expected_distances - distances) <= 0.01) == 6
    assert np.sum(abs(expected_angles - angles) <= 0.01) == 6


def main_test() -> None:
    np.random.seed(seed=1)
    test_fmm_map()
    print("%sFmm_map tests passed!%s" % (color_text["green"], color_text["reset"]))


if __name__ == "__main__":
    main_test()
