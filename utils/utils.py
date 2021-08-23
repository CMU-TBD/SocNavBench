import copy
import json
import os
import random
import shutil
import string
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotmap import DotMap
from matplotlib import figure, pyplot
from trajectory.trajectory import SystemConfig

color_text: Dict[str, str] = {
    "orange": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
    "blue": "\033[36m",
    "yellow": "\033[35m",
    "reset": "\033[00m",
}


def ensure_odd(integer: int) -> bool:
    if integer % 2 == 0:
        integer += 1
    return integer


def render_angle_frequency(p: DotMap) -> int:
    """Returns a render angle frequency
    that looks heuristically nice on plots."""
    return int(p.episode_horizon / 25)


def get_time_str() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def to_json_type(
    elem: Any, json_args: Optional[Dict[str, Any]] = {}
) -> str or int or float or list or dict:
    """ Converts an element to a json serializable type. """
    if isinstance(elem, (int, str, bool, float)):
        return elem  # nothing to do. Primitive already
    if isinstance(elem, (np.int64, np.int32)):
        return int(elem)
    elif isinstance(elem, (np.float64, np.float64)):
        return float(elem)
    elif isinstance(elem, np.ndarray):
        return elem.tolist()
    elif isinstance(elem, dict):
        # recursive for dictionaries within dictionaries
        return dict_to_json(elem, json_args)
    elif isinstance(elem, list):
        # recursive for lists within lists
        return list_to_json(elem, json_args)
    elif hasattr(elem, "to_json_type") and callable(getattr(elem, "to_json_type")):
        return elem.to_json_type(**json_args)
    elif type(elem) is type:  # elem is a class
        return str(elem)
    # try a catch-all by converting to str
    try:
        return str(elem)
    except Exception as e:
        print(
            "{}ERROR: could not serialize elem {} of type {}. Ex: {}{}".format(
                color_text["red"], elem, type(elem), e, color_text["reset"]
            )
        )
        raise e


def dict_to_json(
    param_dict: Dict[str, Any], json_args: Optional[Dict[str, Any]] = {}
) -> Dict[str, str or int or float]:
    """ Converts params_dict to a json serializable dict."""
    json_dict: Dict[str, str or int or float] = {}
    for key in param_dict.keys():
        # possibly recursive for dicts in dicts
        json_dict[key] = to_json_type(param_dict[key], json_args)
    return json_dict


def list_to_json(
    param_list: List[Any], json_args: Optional[Dict[str, Any]] = {}
) -> List[str or int or float or bool]:
    """ Converts params_list to a json serializable list."""
    json_list: List[str or int or float or bool] = [
        to_json_type(elem, json_args) for elem in param_list
    ]
    return json_list


def euclidean_dist2(p1: List[float], p2: List[float]) -> float:
    """Compute the 2D euclidean distance from p1 to p2.

    Args:
        p1 (list): A point in a 2D space (with at least 2 dimens).
        p2 (list): Another point in a 2D space (with at least 2 dimens).

    Returns:
        dist (float): the euclidean (straight-line) distance between the points.
    """
    diff_x: float = p1[0] - p2[0]
    diff_y: float = p1[1] - p2[1]
    return np.sqrt(diff_x ** 2 + diff_y ** 2)


def absmax(x: np.ndarray) -> float or int:
    # returns maximum based off magnitude, not sign
    return max(x.min(), x.max(), key=abs)


def touch(path: str) -> None:
    """Creates an empty file at a specific file location

    Args:
        path (str): The absolute path for the location of the new file
    """
    basedir: str = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, "a"):
        os.utime(path, None)


def natural_sort(l: List[float or int]) -> List[str or int]:
    """Sorts a list of items naturally.

    Args:
        l (list): the list of elements to sort. 

    Returns:
        A naturally sorted list with the same elements as l
    """
    import re

    def convert(text: str) -> int or str:
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str) -> List[int or str]:
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def generate_name(max_chars: int) -> str:
    """Creates a string of max_chars random characters.

    Args:
        max_chars (int): number of characters in random string (name).

    Returns:
        A string of length max_chars with random ascii characters
    """
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(max_chars)]
    )


def conn_recv(connection, buffr_amnt: int = 1024) -> Tuple[bytes, int]:
    """Makes sure all the data from a socket connection is correctly received

    Args:
        connection: The socket connection used as a communication channel.
        buffr_amnt (int, optional): Amount of bytes to transfer at a time. Defaults to 1024.

    Returns:
        data (bytes): The data received from the socket.
        response_len (int): The number of bytes that were transferred
    """
    chunks: List[bytes] = []
    response_len: int = 0
    while True:
        chunk = connection.recv(buffr_amnt)
        if chunk == b"":
            break
        chunks.append(chunk)
        response_len += len(chunk)
    data: bytes = b"".join(chunks)
    return data, response_len


def mkdir_if_missing(dirname: str) -> None:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delete_if_exists(dirname: str) -> None:
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def check_dotmap_equality(d1: DotMap, d2: DotMap) -> bool:
    """Check equality on nested map objects that all keys and values match."""
    assert len(set(d1.keys()).difference(set(d2.keys()))) == 0
    equality: List[bool] = [True] * len(d1.keys())
    for i, key in enumerate(d1.keys()):
        d1_attr = getattr(d1, key)
        d2_attr = getattr(d2, key)
        if type(d1_attr) is DotMap:
            equality[i] = check_dotmap_equality(d1_attr, d2_attr)
    return np.array(equality).all()


def configure_plotting() -> None:
    pyplot.plot.style.use("ggplot")


def subplot2(
    plt: pyplot.plot,
    Y_X: Tuple[int, int],
    sz_y_sz_x: Optional[Tuple[int, int]] = (10, 10),
    space_y_x: Optional[Tuple[int, int]] = (0.1, 0.1),
    T: Optional[bool] = False,
) -> Tuple[figure.Figure, pyplot.axes, List[pyplot.axes]]:
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams["figure.figsize"] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list


def termination_cause_to_color(cause: str) -> Optional[str]:
    cause_colour_mappings: Dict[str, str] = {
        "Success": "green",
        "Pedestrian Collision": "red",
        "Obstacle Collision": "orange",
        "Timeout": "blue",
    }
    if cause in cause_colour_mappings:
        return cause_colour_mappings[cause]
    return None


def iter_print(l: List or Dict) -> str:
    if isinstance(l[0], float):
        return ",".join(["{0: 0.2f}".format(i) for i in l])
    # return string
    return ",".join([str(i) for i in l])


""" BEGIN configs functions """


def generate_random_config(
    environment: Dict[str, int or float or np.ndarray],
    dt: Optional[float] = 0.1,
    max_vel: Optional[float] = 0.6,
) -> SystemConfig:
    pos_3: np.ndarray = generate_random_pos_in_environment(environment)
    return SystemConfig.from_pos3(pos_3, dt=dt, v=max_vel)


def generate_random_pos_3(
    center: np.ndarray, xdiff: Optional[float] = 3.0, ydiff: Optional[float] = 3.0
) -> np.ndarray:
    """
    Generates a random position near the center within an elliptical radius of xdiff and ydiff
    """
    offset_x = 2 * xdiff * random.random() - xdiff  # bound by (-xdiff, xdiff)
    offset_y = 2 * ydiff * random.random() - ydiff  # bound by (-ydiff, ydiff)
    offset_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
    return np.add(center, np.array([offset_x, offset_y, offset_theta]))


def within_traversible(
    new_pos: np.ndarray,
    traversible: np.ndarray,
    map_scale: float,
    stroked_radius: Optional[bool] = False,
) -> bool:
    """
    Returns whether or not the position is in a valid spot in the
    traversible
    """
    pos_x = int(new_pos[0] / map_scale)
    pos_y = int(new_pos[1] / map_scale)
    # Note: the traversible is mapped unintuitively, goes [y, x]
    try:
        if not traversible[pos_y][pos_x]:  # Looking for invalid spots
            return False
        return True
    except:
        return False


def within_traversible_with_radius(
    new_pos: np.ndarray,
    traversible: np.ndarray,
    map_scale: float,
    radius: Optional[int] = 1,
    stroked_radius: Optional[bool] = False,
) -> bool:
    """
    Returns whether or not the position is in a valid spot in the
    traversible the Radius input can determine how many surrounding
    spots must also be valid
    """
    # TODO: use np vectorizing instead of double for loops
    for i in range(2 * radius):
        for j in range(2 * radius):
            if stroked_radius:
                if not ((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                    continue
            pos_x = int(new_pos[0] / map_scale) - radius + i
            pos_y = int(new_pos[1] / map_scale) - radius + j
            # Note: the traversible is mapped unintuitively, goes [y, x]
            if not traversible[pos_y][pos_x]:  # Looking for invalid spots
                return False
    return True


def generate_random_pos_in_environment(
    environment: Dict[str, int or float or np.ndarray]
) -> np.ndarray:
    """
    Generate a random position (x : meters, y : meters, theta : radians)
    and near the 'center' with a nearby valid goal position.
    - Note that the obstacle_traversible and human_traversible are both
    checked to generate a valid pos_3.
    - Note that the "environment" holds the map scale and all the
    individual traversibles if they exists
    - Note that the map_scale primarily refers to the traversible's level
    of precision, it is best to use the dx_m provided in examples.py
    """
    map_scale = float(environment["map_scale"])
    # Combine the occupancy information from the static map and the human
    if "human_traversible" in environment:
        # in this case there exists a "human" traversible as well, and we
        # don't want to generate one human in the traversible of another
        global_traversible = np.empty(environment["map_traversible"].shape)
        global_traversible.fill(True)
        map_t = environment["map_traversible"]
        human_t = environment["human_traversible"]
        # append the map traversible
        global_traversible = np.stack([global_traversible, map_t], axis=2)
        global_traversible = np.all(global_traversible, axis=2)
        # stack the human traversible on top of the map one
        global_traversible = np.stack([global_traversible, human_t], axis=2)
        global_traversible = np.all(global_traversible, axis=2)
    else:
        global_traversible = environment["map_traversible"]

    # Generating new position as human's position
    pos_3 = np.array([0, 0, 0])  # start far out of the traversible

    # continuously generate random positions near the center until one is valid
    while not within_traversible(pos_3, global_traversible, map_scale):
        new_x = random.randint(0, global_traversible.shape[0])
        new_y = random.randint(0, global_traversible.shape[1])
        new_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
        pos_3 = np.array([new_x, new_y, new_theta])

    return pos_3


""" END configs functions """
