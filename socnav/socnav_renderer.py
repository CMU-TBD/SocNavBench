from mp_env import map_utils as mu
from utils import depth_utils as du
from sbpd import sbpd
from utils.utils import mkdir_if_missing
import numpy as np
import sys
import os
import copy
import pickle


class SocNavRenderer():
    """
    An image renderer to render images from the
    SBPD dataset, now with support for multiple humans
    """
    renderer = None
    # Computed by looking at a distribution of walking human radii
    default_human_radius = .5

    def __init__(self, params):
        self.p = params

        # Making use of a dictionary mapping Human's names to humans
        self.humans = {}  # to keep track of the Humans
        self.human_traversible = None

        if self.p.building_params.load_meshes:
            self.d = sbpd.get_dataset(self.p.building_params.dataset_name, 'all',
                                      data_dir=self.p.sbpd_data_dir,
                                      surreal_params=self.p.surreal)
            self.building = self.d.load_data(self.p.building_params.building_name,
                                             self.p.robot_params.physical_params,
                                             self.p.building_params.flip)
            self.humans_loaded = False
            # Instantiating a camera/ shader object is only needed
            # for rgb and depth images
            if 'rgb' in self.p.camera_params.modalities or 'disparity' in self.p.camera_params.modalities:
                from mp_env.render import swiftshader_renderer as sr
                # Note: Resizing disparity images messes up the depth measurement.
                # So if we are rendering disparity we restrict resize to be 1.0.
                if 'disparity' in self.p.camera_params.modalities:
                    assert(self.p.camera_params.im_resize == 1.)
                r_obj = sr.get_r_obj(self.p.camera_params)
                self.building.set_r_obj(r_obj)
                self.building.load_building_into_scene()
            elif 'occupancy_grid' in self.p.camera_params.modalities:
                # MP Env only allows for square top views to be generated currently
                assert(self.p.camera_params.width ==
                       self.p.camera_params.height)
            else:
                assert(False)
        else:
            self.human_radius = self.default_human_radius

    @classmethod
    def get_renderer(cls, params, deepcpy=False, ensure_only_one=False):
        """
        Used to instantiate a renderer object. Ensures that only one renderer
        object ever exists as they are very memory intensive.
        """
        if(ensure_only_one):
            r = cls.renderer
            if r is not None:
                dn, bn, f, c = r.p.dataset_name, r.p.building_name, r.p.flip, r.p.modalities
                if dn == params.dataset_name and bn == params.building_name and f == params.flip and c == params.modalities:
                    return r
                else:
                    assert False, "Renderer settings are different than previously instantiated renderer"

        cls.renderer = cls(params)
        if(not deepcpy):
            return cls.renderer
        return copy.deepcopy(cls.renderer)

    def render_images(self, starts_n2, thetas_n1, crop_size=None, human_visible=True):
        """
        Render the corresponding image from
        the x, y positions in starts_2n facing heading
        thetas_1n
        """
        p = self.p.camera_params
        if 'occupancy_grid' in p.modalities:
            if crop_size is None:
                crop_size = [p.width, p.height]
            imgs = self._get_topview(starts_n2, thetas_n1, crop_size=crop_size)
        elif 'rgb' in p.modalities:
            imgs = self._get_rgb_image(
                starts_n2, thetas_n1, human_visible=human_visible)
        elif 'disparity' in p.modalities:
            return
        else:
            assert(False)
        return np.array(imgs)

    def add_human(self, human, only_sample_human_identity=False):
        """
        Inserts a human mesh where their positional
        arguments specify
        """
        if self.p.building_params.load_meshes:
            if not only_sample_human_identity:
                # Load the human mesh into the scene
                self.building.load_human_into_scene(human)
                self.humans[human.get_name()] = human
                # Log that there is a human in the environment
                self.human_traversible = self.building.human_traversible
                # If updating the human traversible a radius will be dynamically
                # computed for each human position, else the
                try:
                    self.human_radius = self.building.map._human_radius
                except AttributeError:
                    self.human_radius = self.default_human_radius

    def get_human_traversible(self):
        return self.human_traversible

    def remove_human(self, name):
        """
        Remove the human with identity ID
        - Must be done from self.building and self.humans
        """
        if self.p.building_params.load_meshes:
            self.building.remove_human(name)
            self.humans.pop(name)

    def remove_all_humans(self):
        """
        Remove all humans in the scene
        """
        for name, _ in list(self.humans.items()):
            self.remove_human(name)

    def update_human(self, human):
        """
        Updates an existing human within a building 
        """
        if self.p.building_params.load_meshes:
            self.humans[human.get_name()] = human
            self.building.update_human(human)
            self.human_traversible = self.building.human_traversible

    def _get_rgb_image(self, starts_n2, thetas_n1, human_visible):
        """
        Render rgb image(s) from the x, y, theta
        location in starts and thetas.
        """
        if self.p.building_params.load_meshes:
            # Scale thetas by 1/delta_theta as the building object
            # internally scales theta by delta_theta
            nodes_n3 = np.concatenate([starts_n2 * 1.,
                                       thetas_n1 / self.building.robot.delta_theta], axis=1)
            imgs_nmk3 = self.building.render_nodes(
                nodes_n3, modality='rgb', human_visible=human_visible)
        else:
            width = self.p.camera_params.width
            height = self.p.camera_params.height
            resize = self.p.camera_params.im_resize
            width = int(width * resize)
            height = int(height * resize)
            n = len(starts_n2)
            imgs_nmk3 = np.zeros((n, width, height, 3), dtype=np.float32)
        return np.array(imgs_nmk3)

    def _get_topview(self, starts_n2, thetas_n1, crop_size=[64, 64]):
        """
        Render crop_size  topview(s) from the x, y, theta locations
        in starts and thetas.
        """
        # SBPD only supports square top views currently
        assert(crop_size[0] == crop_size[1])

        traversible_map = self.building.map.traversible * 1.

        # In the topview the positive x axis points to the right and
        # the positive y axis points up. The robot is located at
        # (0, (crop_size[0]-1)/2) (in pixel coordinates) facing directly to the right
        x_axis_n2 = np.concatenate(
            [np.cos(thetas_n1), np.sin(thetas_n1)], axis=1)
        y_axis_n2 = -np.concatenate([np.cos(thetas_n1 + np.pi / 2.),
                                     np.sin(thetas_n1 + np.pi / 2.)], axis=1)
        robot_loc_2 = np.array([0, (crop_size[0] - 1.) / 2.])

        crops_nmk = mu.generate_egocentric_maps([traversible_map], [1.0], [crop_size[0]],
                                                starts_n2, x_axis_n2, y_axis_n2, dst_theta=0.,
                                                dst_loc=robot_loc_2)[0]

        # Invert the crops so that 1.0 corresponds to occupied space
        # and 0.0 corresponds to free space
        crops_nmk1 = [np.logical_not(
            crop_mk[:, :, None]) * 1.0 for crop_mk in crops_nmk]
        return np.array(crops_nmk1)

    def _get_depth_image(self, starts_n2, thetas_n1, xy_resolution, map_size, pos_3, human_visible=True):
        """
        Render analytically projected depth images at the locations in
        starts, thetas. Bin data inside bins in a resolution of xy_resolution along x and y axis and
        z_bins in the z direction. Z Direction is the vertical z = 0 is floor. """
        r_obj = self.building.r_obj
        robot = self.building.robot
        z_bins = [-10, robot.base, robot.base + robot.height]

        nodes_n3 = np.concatenate(
            [starts_n2 * 1., thetas_n1 / self.building.robot.delta_theta], axis=1)
        # Disparity in centimeters
        disparity_imgs_cm = np.array(self.building.render_nodes(
            nodes_n3, 'disparity', human_visible=human_visible))

        depth_imgs_meters = 100. / \
            (disparity_imgs_cm[..., 1] + 0.000001)  # no divide by 0 error

        # Optionally Clip Depth Readings
        if self.p.camera_params.max_depth_meters < np.inf:
            inf_mask = np.isinf(depth_imgs_meters)
            max_depth_mask = depth_imgs_meters >= self.p.camera_params.max_depth_meters
            mask = np.logical_and(np.logical_not(inf_mask), max_depth_mask)
            depth_imgs_meters[mask] = self.p.camera_params.max_depth_meters

        assert (r_obj.fov_horizontal == r_obj.fov_vertical)
        # Generate a Point Cloud from the Depth Image
        # (In the Camera Coordinate System)
        cm = du.get_camera_matrix(
            r_obj.width, r_obj.height, r_obj.fov_vertical)
        XYZ = du.get_point_cloud_from_z(depth_imgs_meters, cm)
        XYZ = XYZ * 100.  # convert to centimeters

        # Transform from the camera coordinate system
        # to the geocentric coordinate system (align the point cloud to the ground plane)
        XYZ = du.make_geocentric(
            XYZ, robot.sensor_height, robot.camera_elevation_degree)

        # Note: Added here to get the depth image in the current frame
        # Transform from the ground plane to the robots current
        # location in the map
        XYZ = self.transform_to_current_frame(XYZ[0], pos_3)
        XYZ = XYZ[None, :, :, :]

        count, isvalid = du.bin_points(
            XYZ * 1., map_size, z_bins, xy_resolution)
        count = [x[0, ...] for x in np.split(count, count.shape[0], 0)]
        isvalid = [x[0, ...] for x in np.split(isvalid, isvalid.shape[0], 0)]

        return disparity_imgs_cm, count, isvalid

    def transform_to_current_frame(self, XYZ, current_loc):
        R = du.get_r_matrix([0., 0., 1.], angle=current_loc[2] - np.pi / 2.)
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
        XYZ[:, :, 0] = XYZ[:, :, 0] + current_loc[0] * \
            100.  # convert to centimeters
        XYZ[:, :, 1] = XYZ[:, :, 1] + current_loc[1] * \
            100.  # convert to centimeters
        return XYZ

    def get_config(self):
        """
        Return the resolution and traversible of the SBPD building. If python version
        is 2.7 return a precomputed traversible (from python 3.6) as some of the
        loading libraries do not match currently.
        """

        traversible_dir = self.p.traversible_dir
        traversible_dir = os.path.join(traversible_dir,
                                       self.p.building_params.building_name)

        if self.p.building_params.load_traversible_from_pickle_file or not self.p.building_params.load_meshes:
            filename = os.path.join(traversible_dir, 'data.pkl')
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            resolution = data['resolution']
            traversible = data['traversible']
        else:
            assert sys.version[0] == '3'
            resolution, traversible = self.building.env.resolution, self.building.traversible

            mkdir_if_missing(traversible_dir)

            filenames = os.listdir(traversible_dir)
            if 'data.pkl' not in filenames:
                data = {'resolution': resolution,
                        'traversible': traversible}
                with open(os.path.join(traversible_dir, 'data.pkl'), 'wb') as f:
                    # Save with protocol = 2 for python2.7
                    pickle.dump(data, f, protocol=2)

        return resolution, traversible
