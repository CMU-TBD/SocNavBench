import numpy as np


class VoxelMap(object):
    """
    A voxel map object that allows to compute the function value at any arbitrary point in the voxel grid.
    """

    def __init__(self, scale, origin_2, map_size_2, function_array_mn):
        """
        Args:
            scale: The scale (dx) of the grid.
            origin_2: The origin of the voxel grid in the voxel space.
            map_size_2: The number of voxels in the voxel grid.
            function_array_mn: The function stored in the voxel grid. The size of the grid is assumed to be mxn, where
            m is the size in the y-dimension and n is in the x-dimension.
        """
        self.map_scale = np.array(scale, dtype=np.float32)
        self.map_origin_2 = origin_2
        self.map_size_int32_2 = map_size_2.astype(np.int32)
        self.map_size_float32_2 = map_size_2.astype(np.float32)
        self.voxel_function_mn = function_array_mn

    def compute_voxel_function(self, position_nk2, invalid_value=100.):
        """
        Compute the voxel function at the specified positions.
        """
        # Compute the position in the voxel space.
        voxel_space_position_nk2 = self.grid_world_to_voxel_world(
            position_nk2) - self.map_origin_2

        # Define the lower and upper voxels. Mod is done to make sure that the invalid voxels have been
        # assigned a valid voxel. However, these invalid voxels will be discarded later.
        lower_voxel_indices_nk2_xy = np.mod(
            np.floor(voxel_space_position_nk2).astype(np.int32), self.map_size_int32_2)
        upper_voxel_indices_nk2_xy = np.mod(
            lower_voxel_indices_nk2_xy + 1, self.map_size_int32_2)

        lower_voxel_float_nk2 = lower_voxel_indices_nk2_xy.astype(np.float32)
        upper_voxel_float_nk2 = upper_voxel_indices_nk2_xy.astype(np.float32)

        # Voxel indices for 4 corner voxels. Note that indices are stacked out of order for voxel_indices11 to make
        # sure that the first element along axis2 represents y-value (since the voxel map's first dimension is y and
        # not x).
        voxel_indices_nk4 = np.concatenate(
            [lower_voxel_indices_nk2_xy, upper_voxel_indices_nk2_xy], axis=2)

        # TODO: Casting as int64 because tfe.DEVICE_PLACEMENT_SILENT
        # is not working to place non-gpu ops (i.e. gather for (int32, int 32)) on
        # the cpu. This is potentially slowing things down as it copies to cpu
        voxel_indices_int64_nk4 = voxel_indices_nk4.astype(np.int32)

        def gather_nd(data, indices):
            # very similar to tf.gather_nd but with numpy slicing
            return data[indices[:, :, 0], indices[:, :, 1]]
        # Voxel function values at corner points
        data = self.voxel_function_mn
        indices_1 = voxel_indices_int64_nk4[:, :, [1, 0]]
        data11_nk = gather_nd(data, indices_1)
        indices_2 = voxel_indices_int64_nk4[:, :, [1, 2]]
        data21_nk = gather_nd(data, indices_2)
        # equivalent to np.take (since its a 3D matrix)
        indices_3 = voxel_indices_int64_nk4[:, :, [3, 0]]
        data12_nk = gather_nd(data, indices_3)
        indices_4 = voxel_indices_int64_nk4[:, :, [3, 2]]
        data22_nk = gather_nd(data, indices_4)

        # Define gammas for x interpolation
        gamma1 = upper_voxel_float_nk2[:, :, 0] - \
            voxel_space_position_nk2[:, :, 0]
        gamma2 = voxel_space_position_nk2[:, :, 0] - \
            lower_voxel_float_nk2[:, :, 0]

        # Define betas for y interpolation
        beta1 = upper_voxel_float_nk2[:, :, 1] - \
            voxel_space_position_nk2[:, :, 1]
        beta2 = voxel_space_position_nk2[:, :, 1] - \
            lower_voxel_float_nk2[:, :, 1]

        # Interpolation in the x-direction
        fx1_nk = gamma1 * data11_nk + gamma2 * data21_nk
        fx2_nk = gamma1 * data12_nk + gamma2 * data22_nk

        # Interpolation in the y-direction
        f_nk = beta1 * fx1_nk + beta2 * fx2_nk

        # Discard the invalid voxels
        valid_voxels_nk = self.is_valid_voxel(position_nk2)

        return np.where(valid_voxels_nk, f_nk, np.ones_like(f_nk) * invalid_value)

    def grid_world_to_voxel_world(self, position_nk2):
        """
        Convert the positions in the global world to the voxel world coordinates.
        """
        return position_nk2 / self.map_scale

    def is_valid_voxel(self, position_nk2):
        """
        Check if a given set of positions are within the voxel map or not.

        """
        voxel_space_position_nk2 = self.grid_world_to_voxel_world(
            position_nk2) - self.map_origin_2
        return np.logical_and(np.all(voxel_space_position_nk2 >= 0., axis=2),
                              np.all(voxel_space_position_nk2 < (self.map_size_float32_2 - 1.), axis=2))
