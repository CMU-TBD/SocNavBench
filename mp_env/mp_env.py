from trajectory.trajectory import SystemConfig
from typing import Dict, List, Optional
import numpy as np
from mp_env.map_utils import (
    make_map,
    compute_traversibility,
    add_human_to_traversible,
    pick_largest_cc,
)
from agents.humans.human import HumanAppearance
from dotmap import DotMap
from mp_env.map_utils import Foo
from mp_env.render.swiftshader_renderer import Shape, SwiftshaderRenderer
from simulators.sim_state import AgentState


class Building:
    def __init__(
        self,
        dataset,  # StanfordBuildingParserDataset
        name: str,
        robot_params: DotMap,
        env: Foo,
        flip: Optional[bool] = False,
    ):
        self.restrict_to_largest_cc: bool = True
        self.robot: DotMap = robot_params
        self.env: Foo = env
        self.name: str = name

        # Load the building meta data.

        self.env_paths: Dict[str, str] = dataset.load_building(self.name)
        self.materials_scale: float = 1.0

        # load the shape of the entire building
        self.shapess: List[Shape] = dataset.load_building_meshes(
            self.env_paths, materials_scale=1.0
        )

        if flip:
            for shapes in self.shapess:
                shapes.flip_shape()

        vs: List[np.ndarray] = []
        for shapes in self.shapess:
            vs.append(shapes.get_vertices()[0])
        vs: np.ndarray = np.concatenate(vs, axis=0)

        self.map: Foo = make_map(env.padding, env.resolution, vertex=vs, sc=100.0)
        self.map = compute_traversibility(
            self.map,
            robot_params.base,
            robot_params.height,
            robot_params.radius,
            env.valid_min,
            env.valid_max,
            env.num_point_threshold,
            shapess=self.shapess,
            sc=100.0,
            n_samples_per_face=env.n_samples_per_face,
        )

        # The map object has _traversible (only the SBPD building)
        # and traversible (the current traversible which may include
        # space occupied by any humans in the environment)
        self.traversible: np.ndarray = self.map.traversible * 1
        self.human_traversible: np.ndarray = self.map._human_traversible * 1

        self.flipped: bool = flip
        self.renderer_entitiy_ids: List[int] = []
        if self.restrict_to_largest_cc:
            self.traversible = pick_largest_cc(self.traversible)
            self.map._traversible = self.traversible
            self.map.traversible = self.traversible

        # Instance variable for storing human information and humans
        self.named_humans = set()
        self.individual_human_traversibles = {}

    def set_r_obj(self, r_obj: SwiftshaderRenderer) -> None:
        self.r_obj = r_obj

    def load_building_into_scene(self, dedup_tbo: Optional[bool] = False) -> None:
        assert self.shapess is not None
        # Loads the scene.
        self.renderer_entitiy_ids += self.r_obj.load_shapes(self.shapess, dedup_tbo)
        # Free up memory, we dont need the mesh or the materials anymore.
        self.shapess: List[Shape] = None

    def _transform_to_ego(
        self, world_vertices_n3: np.ndarray, pos_3: np.ndarray
    ) -> np.ndarray:
        """
        Transforms the world_vertices_n3 ([x, y, z])
        to the ego frame where pos_3 [x, y, theta]
        is the location of the origin in the world frame.
        """
        # Translate by X,Y
        ego_vertices_xy_n2 = world_vertices_n3[:, :2] - pos_3[:2]

        # Rotate by theta
        theta = -pos_3[2]
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        ego_vertices_xy_n2 = ego_vertices_xy_n2.dot(R)

        # Join the changed X, Y coordinates with
        # the unchanged Z coordinates
        ego_vertices_n3 = np.concatenate(
            [ego_vertices_xy_n2, world_vertices_n3[:, 2:3]], axis=1
        )

        return ego_vertices_n3

    def _transform_to_world(
        self, ego_vertices_n3: np.ndarray, pos_3: np.ndarray
    ) -> np.ndarray:
        """
        Transforms the ego_vertices_n3 ([x, y, z])
        to the world frame where pos_3 [x, y, theta]
        is the location of the origin in the world frame.
        """
        ego_vertices_xy_n2 = ego_vertices_n3[:, :2]

        theta = pos_3[2]

        # Rotate by theta
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        world_vertices_xy_n2 = ego_vertices_xy_n2.dot(R)

        # Translate by X,Y
        world_vertices_xy_n2 += pos_3[:2]

        # Join the changed X, Y coordinates with
        # the unchanged Z coordinates
        world_vertices_n3 = np.concatenate(
            [world_vertices_xy_n2, ego_vertices_n3[:, 2:3]], axis=1
        )

        return world_vertices_n3

    def _traversible_world_to_vertex_world(self, pos_3: np.ndarray) -> np.ndarray:
        """
        Convert an [x, y, theta] coordinate specified on
        the traversible map to a [x, y, theta] coordinate
        in the same coordinate frame as the meshes themselves
        """
        # Divide by 100 to convert origin back to meters
        # The mesh is stored in units of meters
        # So we just offset the human by the desired # meters
        xy_offset_map = self.map.origin / 100.0 + pos_3[:2]

        return np.array([xy_offset_map[0], xy_offset_map[1], pos_3[2]])

    def create_human_key(self, name: str) -> str:
        return "human_{}".format(name)

    def load_human_into_scene(
        self,
        human,  # technically an AgentState
        dedup_tbo: Optional[bool] = False,
        allow_repeat_humans: Optional[bool] = False,
    ) -> None:
        """
        Load a gendered human mesh with associated body shape, texture, and human_materials
        into a building at their current config and speed in the static building.
        """
        return self.load_bulk_humans_into_scene([human], dedup_tbo, allow_repeat_humans)

    def load_bulk_humans_into_scene(
        self,
        humans: List[AgentState],
        dedup_tbo: Optional[bool] = False,
        allow_repeat_humans: Optional[bool] = False,
    ) -> None:
        """loads many human meshes into the scene at once (SLOW)"""
        all_human_shapes: List = []  # list of HumanShapes
        for human in humans:
            # Add human to dictionary in building
            human_appearance: HumanAppearance = human.get_appearance()
            # StanfordBuildingParserDataset
            appearance_dataset = human_appearance.dataset
            current_config: SystemConfig = human.get_current_config()
            name: str = human.get_name()
            pos_3 = current_config.position_and_heading_nk3(squeeze=True)
            speed = current_config.speed_nk1()
            gender = human_appearance.get_gender()
            human_materials = human_appearance.get_texture()
            body_shape = human_appearance.get_shape()
            rng = human_appearance.get_mesh_rng()
            self.named_humans.add(name)

            # Load the human mesh
            shapess, center_pos_3, _ = appearance_dataset.load_random_human(
                speed, gender, human_materials, body_shape, rng
            )
            assert len(shapess) == 1  # only loads a single human at a time

            # Make sure the human's feet are actually on the ground in SBPD
            # (i.e. the minimum z coordinate is 0)

            z_offset = shapess[0].meshes[0].vertices[:, 2].min()
            shapess[0].meshes[0].vertices = np.concatenate(
                [
                    shapess[0].meshes[0].vertices[:, :2],
                    shapess[0].meshes[0].vertices[:, 2:3] - z_offset,
                ],
                axis=1,
            )

            # Make sure the human is in the canonical position
            # The centerpoint between the left and right foot should be (0, 0)
            # and the heading (average of the direction of the two feet) should be 0
            shapess[0].meshes[0].vertices = self._transform_to_ego(
                shapess[0].meshes[0].vertices, center_pos_3
            )
            human_ego_vertices = shapess[0].meshes[0].vertices * 1.0

            # Move the human to the desired location
            pos_3 = self._traversible_world_to_vertex_world(pos_3)
            shapess[0].meshes[0].vertices = self._transform_to_world(
                shapess[0].meshes[0].vertices, pos_3
            )
            shapess[0].meshes[0].name = self.create_human_key(name)

            # append this shape to the list of all shapes
            all_human_shapes.append(shapess[0])

        # load all shapes at once
        self.renderer_entitiy_ids += self.r_obj.load_shapes(
            all_human_shapes, dedup_tbo, allow_repeat_humans=allow_repeat_humans
        )

        # Update The Human Traversible
        if appearance_dataset.surreal_params.compute_human_traversible:
            map = self.map
            env = self.env
            robot: DotMap = self.robot
            map = add_human_to_traversible(
                map,
                robot.base,
                robot.height,
                robot.radius,
                env.valid_min,
                env.valid_max,
                env.num_point_threshold,
                shapess=all_human_shapes,
                sc=100.0,
                n_samples_per_face=env.n_samples_per_face,
                human_xy_center_2=pos_3[:2],
            )

            self.traversible = map.traversible
            self.individual_human_traversibles[name] = map._human_traversible
            self.human_traversible = self.compute_human_traversible()
            self.map = map
        self.human_ego_vertices = human_ego_vertices

    def compute_human_traversible(self) -> np.ndarray:
        new_human_traversible = np.ones_like(self.map._traversible * 1.0)
        for human_name in list(self.named_humans):
            if human_name in self.individual_human_traversibles:
                new_human_traversible = np.stack(
                    [
                        new_human_traversible,
                        self.individual_human_traversibles[human_name],
                    ],
                    axis=2,
                )
                new_human_traversible = np.all(new_human_traversible, axis=2)
        return new_human_traversible

    def remove_human(self, name):
        """
        Remove a human that has been loaded into the SBPD environment by name
        """
        # Delete the human mesh from memory
        self.r_obj.remove_human(name)

        # Remove the human from the list of loaded entities
        key: str = self.create_human_key(name)
        self.renderer_entitiy_ids.remove(key)

        # Update the traversible to be human free
        self.map.traversible = self.map._traversible
        self.traversible = self.map._traversible
        if name in self.individual_human_traversibles:
            self.individual_human_traversibles.pop(name)
        self.human_traversible = self.compute_human_traversible()

        # Remove from set
        if name in self.named_humans:
            self.named_humans.remove(name)

    def update_human(self, human: AgentState) -> None:
        """
        Removes the previously loaded human mesh,
        and loads a new one with the same gender, texture
        and body shape at the updated position and speed
        """
        # Remove the previous human instance
        if human.get_name() in self.named_humans:
            self.remove_human(human.get_name())

        # Load a new human with the updated speed and position
        # same human appearance
        self.load_human_into_scene(human)

    def update_bulk_humans(self, humans: List[AgentState]) -> None:
        """ updating many humans at once """
        # Remove the previous human
        for human in humans:
            if human.get_name() in self.named_humans:
                self.remove_human(human.get_name())
        self.load_bulk_humans_into_scene(humans)

    def to_actual_xyt(self, pqr: np.ndarray) -> np.ndarray:
        """Converts from node array to location array on the map."""
        out = pqr * 1.0
        # p = pqr[:,0:1]; q = pqr[:,1:2]; r = pqr[:,2:3];
        # out = np.concatenate((p + self.map.origin[0], q + self.map.origin[1], r), 1)
        return out

    def set_building_visibility(self, visibility: bool) -> None:
        self.r_obj.set_entity_visible(self.renderer_entitiy_ids, visibility)

    def set_human_visibility(self, visibility: bool) -> None:
        """
        Makes ALL humans visible or not, to remove a singular
        human use remove_human(ID)
        """
        human_entity_ids = list(
            filter(lambda x: "human" in x, self.renderer_entitiy_ids)
        )
        self.r_obj.set_entity_visible(human_entity_ids, visibility)

    def render_nodes(
        self,
        nodes: np.ndarray,
        modality,
        perturb: Optional[np.ndarray] = None,
        aux_delta_theta: Optional[float] = 0.0,
        human_visible: Optional[bool] = True,
    ) -> List[np.ndarray]:
        # List of nodes to render.
        self.set_building_visibility(True)
        self.set_human_visibility(human_visible)
        if perturb is None:
            perturb = np.zeros((len(nodes), 4))

        imgs = []
        r = 2
        elevation_z = r * np.tan(np.deg2rad(self.robot.camera_elevation_degree))

        for i in range(len(nodes)):
            xyt = self.to_actual_xyt(nodes[i][np.newaxis, :] * 1.0)[0, :]
            lookat_theta = 3.0 * np.pi / 2.0 - (
                xyt[2] + perturb[i, 2] + aux_delta_theta
            ) * (self.robot.delta_theta)
            nxy = np.array([xyt[0] + perturb[i, 0], xyt[1] + perturb[i, 1]]).reshape(
                1, -1
            )
            nxy = nxy * self.map.resolution
            nxy = nxy + self.map.origin
            camera_xyz = np.zeros((1, 3))
            camera_xyz[...] = [nxy[0, 0], nxy[0, 1], self.robot.sensor_height]
            camera_xyz = camera_xyz / 100.0
            lookat_xyz = np.array(
                [-r * np.sin(lookat_theta), -r * np.cos(lookat_theta), elevation_z]
            )
            lookat_xyz = lookat_xyz + camera_xyz[0, :]
            self.r_obj.position_camera(
                camera_xyz[0, :].tolist(), lookat_xyz.tolist(), [0.0, 0.0, 1.0]
            )
            img = self.r_obj.render(modality, take_screenshot=True, output_type=0)
            img = [x for x in img if x is not None]
            img = np.concatenate(img, axis=2).astype(np.float32)
            if perturb[i, 3] > 0:
                img = img[:, ::-1, :]
            imgs.append(img)

        self.set_building_visibility(False)
        return imgs
