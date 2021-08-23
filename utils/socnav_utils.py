import copy
from typing import Dict, Optional, Tuple

import numpy as np
from dotmap import DotMap
from socnav.socnav_renderer import SocNavRenderer
from agents.humans.human import HumanAppearance
from sbpd.sbpd import StanfordBuildingParserDataset

from utils.utils import color_text


def load_building(
    p: DotMap, force_rebuild: Optional[bool] = False
) -> Tuple[SocNavRenderer, float, np.ndarray]:
    if force_rebuild:
        print(
            "%sForce reloading building%s" % (color_text["yellow"], color_text["reset"])
        )
        # it *should* have been the case that the user did not load the meshes
        assert p.building_params.load_meshes == False
        p2 = copy.deepcopy(p)
        p2.building_params.load_meshes = True
        r = SocNavRenderer.get_renderer(p2)
        # obtain "resolution and traversible of building"
        dx_cm, traversible = r.get_config()
    else:
        try:
            # get the renderer from the camera p
            r = SocNavRenderer.get_renderer(p)
            # obtain "resolution and traversible of building"
            dx_cm, traversible = r.get_config()
        except FileNotFoundError:  # did not find traversible.pkl for this map
            print(
                "%sUnable to find traversible, reloading building%s"
                % (color_text["red"], color_text["reset"])
            )
            # it *should* have been the case that the user did not load the meshes
            assert p.building_params.load_meshes == False
            p2 = copy.deepcopy(p)
            p2.building_params.load_meshes = True
            r = SocNavRenderer.get_renderer(p2)
            # obtain "resolution and traversible of building"
            dx_cm, traversible = r.get_config()
    return r, dx_cm, traversible


def construct_environment(
    p: DotMap, test: str, episode: DotMap, verbose: Optional[bool] = True
) -> Tuple[Dict[str, int or float or np.ndarray], SocNavRenderer]:
    # update map to match the episode params
    p.building_params.building_name = episode.map_name
    if verbose:
        print(
            '\n\nStarting episode "%s%s%s" in building "%s%s%s"'
            % (
                color_text["yellow"],
                test,
                color_text["reset"],
                color_text["yellow"],
                p.building_params.building_name,
                color_text["reset"],
            )
        )
    r, dx_cm, traversible = load_building(p)
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m: float = dx_cm / 100.0
    if p.render_3D:
        # Get the surreal dataset for human generation
        surreal_data: StanfordBuildingParserDataset = r.d
        # Update the Human's appearance classes to contain the dataset
        HumanAppearance.dataset = surreal_data
        human_traversible = np.empty(traversible.shape)
        human_traversible.fill(1)  # initially all good
    room_center = (
        np.array([traversible.shape[1] * 0.5, traversible.shape[0] * 0.5, 0.0]) * dx_m
    )
    # Create default environment which is a dictionary
    # containing ["map_scale", "traversibles"]
    # which is a constant and list of traversibles respectively
    environment: Dict[str, float or np.ndarray] = {}
    environment["map_scale"] = float(dx_m)
    environment["map_name"] = p.building_params.building_name
    environment["room_center"] = room_center
    # obstacle traversible / human traversible
    if p.render_3D:
        environment["human_traversible"] = np.array(human_traversible)
    environment["map_traversible"] = 1.0 * np.array(traversible)
    return environment, r
