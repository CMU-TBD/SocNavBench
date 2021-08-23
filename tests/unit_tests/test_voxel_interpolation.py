import numpy as np
from scipy import interpolate
from utils.utils import color_text
from utils.voxel_map_utils import VoxelMap


def test_voxel_interpolation() -> None:
    # Create a grid and a function within that grid
    scale = 0.1
    grid_size = np.array([31, 31])
    grid_origin = np.array([5.0, 6.0])

    x = np.linspace(
        grid_origin[0], grid_origin[0] + scale * (grid_size[0] - 1), grid_size[0]
    )
    y = np.linspace(
        grid_origin[1], grid_origin[1] + scale * (grid_size[1] - 1), grid_size[1]
    )

    xv, yv = np.meshgrid(x, y, sparse=False, indexing="xy")
    zv = 3 * xv + 5 * yv

    voxel_map = VoxelMap(
        scale=scale,
        origin_2=np.array(grid_origin / scale, dtype=np.float32),
        map_size_2=np.array(grid_size, dtype=np.float32),
        function_array_mn=np.array(zv, dtype=np.float32),
    )

    # Let's have a bunch of points to test the interpolation
    test_positions = np.array(
        [
            [[5.02, 6.01], [5.67, 7.73], [6.93, 6.93]],
            [[9.1, 7.2], [7.889, 8.22], [7.1, 8.1]],
        ],
        dtype=np.float32,
    )

    # Get the interpolated output of the voxel map
    interpolated_values = voxel_map.compute_voxel_function(
        test_positions, invalid_value=-1.0
    )
    interpolated_values = np.reshape(interpolated_values, [-1])

    # Expected interpolated values
    expected_interpolated_values = (
        3 * test_positions[:, :, 0] + 5 * test_positions[:, :, 1]
    )
    expected_interpolated_values = np.reshape(expected_interpolated_values, [-1])
    expected_interpolated_values[3] = -1.0

    # Scipy Interpolated values
    f_scipy = interpolate.RectBivariateSpline(y, x, zv, kx=1, ky=1)
    scipy_interpolated_values = f_scipy.ev(
        np.reshape(test_positions[:, :, 1], [-1]),
        np.reshape(test_positions[:, :, 0], [-1]),
    )
    scipy_interpolated_values[3] = -1.0

    assert np.sum(abs(expected_interpolated_values - interpolated_values) <= 0.01) == 6
    assert np.sum(abs(scipy_interpolated_values - interpolated_values) <= 0.01) == 6


def main_test() -> None:
    np.random.seed(seed=1)
    test_voxel_interpolation()
    print(
        "%sVoxel interpolation tests passed!%s"
        % (color_text["green"], color_text["reset"])
    )


if __name__ == "__main__":
    main_test()
