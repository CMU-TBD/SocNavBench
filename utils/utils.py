import os
import json
import copy
import numpy as np
import shutil
from dotmap import DotMap
from random import random
import string
import random
import time
import logging
from trajectory.trajectory import SystemConfig
from contextlib import contextmanager

color_orange = '\033[33m'
color_green = '\033[32m'
color_red = '\033[31m'
color_blue = '\033[36m'
color_yellow = '\033[35m'
color_reset = '\033[00m'


def ensure_odd(integer):
    if integer % 2 == 0:
        integer += 1
    return integer


def render_angle_frequency(p):
    """Returns a render angle frequency
    that looks heuristically nice on plots."""
    return int(p.episode_horizon / 25)


def log_dict_as_json(params, filename):
    """Save params (either a DotMap object or a python dictionary) to a file in json format"""
    with open(filename, 'w') as f:
        if isinstance(params, DotMap):
            params = params.toDict()
        param_dict_serializable = _to_json_serializable_dict(
            copy.deepcopy(params))
        json.dump(param_dict_serializable, f, indent=4, sort_keys=True)


def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


class Timer():
    def __init__(self, skip=0):
        self.calls = 0.
        self.start_time = 0.
        self.time_per_call = 0.
        self.time_ewma = 0.
        self.total_time = 0.
        self.last_log_time = 0.
        self.skip = skip

    def tic(self):
        self.start_time = time.time()

    def display(self, average=True, log_at=-1, log_str='', type='calls', mul=1,
                current_time=None):
        if current_time is None:
            current_time = time.time()
        if self.skip == 0:
            ewma = self.time_ewma * mul / \
                np.maximum(0.01, (1. - 0.99**self.calls))
            if type == 'calls' and log_at > 0 and np.mod(self.calls / mul, log_at) == 0:
                _ = []
                logging.info('%s: %f seconds / call, %d calls.',
                             log_str, ewma, self.calls / mul)
            elif type == 'time' and log_at > 0 and current_time - self.last_log_time >= log_at:
                _ = []
                logging.info('%s: %f seconds / call, %d calls.',
                             log_str, ewma, self.calls / mul)
                self.last_log_time = current_time
        # return self.time_per_call*mul
        return ewma

    def toc(self, average=True, log_at=-1, log_str='', type='calls', mul=1):
        if self.skip > 0:
            self.skip = self.skip - 1
        else:
            if self.start_time == 0:
                logging.error('Timer not started by calling tic().')
            t = time.time()
            diff = time.time() - self.start_time
            self.total_time += diff
            self.calls += 1.
            self.time_per_call = self.total_time / self.calls
            alpha = 0.99
            self.time_ewma = self.time_ewma * alpha + (1 - alpha) * diff
            self.display(average, log_at, log_str,
                         type, mul, current_time=time)

        if average:
            return self.time_per_call * mul
        else:
            return diff

    @contextmanager
    def record(self):
        self.tic()
        yield
        self.toc()


class Foo(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        str_ = ''
        for v in vars(self).keys():
            a = getattr(self, v)
            if True:  # isinstance(v, object):
                str__ = str(a)
                str__ = str__.replace('\n', '\n  ')
            else:
                str__ = str(a)
            str_ += '{:s}: {:s}'.format(v, str__)
            str_ += '\n'
        return str_


"""BEGIN SOCNAV UTILS"""


def load_building(p, force_rebuild=False):
    from socnav.socnav_renderer import SocNavRenderer
    if force_rebuild:
        print("%sForce reloading building%s" % (color_yellow, color_reset))
        # it *should* have been the case that the user did not load the meshes
        assert(p.building_params.load_meshes == False)
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
            print("%sUnable to find traversible, reloading building%s" %
                  (color_red, color_reset))
            # it *should* have been the case that the user did not load the meshes
            assert(p.building_params.load_meshes == False)
            p2 = copy.deepcopy(p)
            p2.building_params.load_meshes = True
            r = SocNavRenderer.get_renderer(p2)
            # obtain "resolution and traversible of building"
            dx_cm, traversible = r.get_config()
    return r, dx_cm, traversible


def construct_environment(p, test, episode, verbose=True):
    # update map to match the episode params
    p.building_params.building_name = episode.map_name
    if verbose:
        print("%s\n\nStarting episode \"%s\" in building \"%s\"%s\n\n" %
              (color_yellow, test, p.building_params.building_name, color_reset))
    r, dx_cm, traversible = load_building(p)
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm / 100.0
    if p.render_3D:
        # Get the surreal dataset for human generation
        surreal_data = r.d
        # Update the Human's appearance classes to contain the dataset
        from agents.humans.human_appearance import HumanAppearance
        HumanAppearance.dataset = surreal_data
        human_traversible = np.empty(traversible.shape)
        human_traversible.fill(1)  # initially all good
    room_center = np.array([traversible.shape[1] * 0.5,
                            traversible.shape[0] * 0.5,
                            0.0]) * dx_m
    # Create default environment which is a dictionary
    # containing ["map_scale", "traversibles"]
    # which is a constant and list of traversibles respectively
    environment = {}
    environment["map_scale"] = float(dx_m)
    environment["room_center"] = room_center
    # obstacle traversible / human traversible
    if p.render_3D:
        environment["human_traversible"] = np.array(human_traversible)
    environment["map_traversible"] = 1. * np.array(traversible)
    return environment, r


def _to_json_serializable_dict(param_dict: dict):
    """Converts params_dict to a json serializable dict.

    Args:
        param_dict (dict): the dictionary to be serialized
    """
    def _to_serializable_type(elem):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            return _to_json_serializable_dict(elem)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)
    for key in param_dict.keys():
        param_dict[key] = _to_serializable_type(param_dict[key])
    return param_dict


def euclidean_dist2(p1: list, p2: list):
    """Compute the 2D euclidean distance from p1 to p2.

    Args:
        p1 (list): A point in a 2D space (with at least 2 dimens).
        p2 (list): Another point in a 2D space (with at least 2 dimens).

    Returns:
        dist (float): the euclidean (straight-line) distance between the points.
    """
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]
    return np.sqrt(diff_x**2 + diff_y**2)


def absmax(x):
    # returns maximum based off magnitude, not sign
    return max(x.min(), x.max(), key=abs)


def touch(path: str):
    """Creates an empty file at a specific file location

    Args:
        path (str): The absolute path for the location of the new file
    """
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, 'a'):
        os.utime(path, None)


def natural_sort(l: list):
    """Sorts a list of items naturally.

    Args:
        l (list): the list of elements to sort. 

    Returns:
        A naturally sorted list with the same elements as l
    """
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def generate_name(max_chars: int):
    """Creates a string of max_chars random characters.

    Args:
        max_chars (int): number of characters in random string (name).

    Returns:
        A string of length max_chars with random ascii characters
    """
    return "".join([
        random.choice(string.ascii_letters + string.digits)
        for n in range(max_chars)
    ])


def conn_recv(connection, buffr_amnt: int = 1024):
    """Makes sure all the data from a socket connection is correctly received

    Args:
        connection: The socket connection used as a communication channel.
        buffr_amnt (int, optional): Amount of bytes to transfer at a time. Defaults to 1024.

    Returns:
        data (bytes): The data received from the socket.
        response_len (int): The number of bytes that were transferred
    """
    chunks = []
    response_len = 0
    while True:
        chunk = connection.recv(buffr_amnt)
        if chunk == b'':
            break
        chunks.append(chunk)
        response_len += len(chunk)
    data = b''.join(chunks)
    return data, response_len


def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delete_if_exists(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def check_dotmap_equality(d1, d2):
    """Check equality on nested map objects that all keys and values match."""
    assert(len(set(d1.keys()).difference(set(d2.keys()))) == 0)
    equality = [True] * len(d1.keys())
    for i, key in enumerate(d1.keys()):
        d1_attr = getattr(d1, key)
        d2_attr = getattr(d2, key)
        if type(d1_attr) is DotMap:
            equality[i] = check_dotmap_equality(d1_attr, d2_attr)
    return np.array(equality).all()


def configure_plotting():
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')


def subplot2(plt, Y_X, sz_y_sz_x=(10, 10), space_y_x=(0.1, 0.1), T=False):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list


def termination_cause_to_color(cause: str):
    if(cause == "Success"):
        return "green"
    if(cause == "Pedestrian Collision"):
        return "red"
    if(cause == "Obstacle Collision"):
        return "orange"
    if(cause == "Timeout"):
        return "blue"
    return None


def color_print(color: str):
    col_str = color_reset
    if color == "green":
        col_str = color_green
    elif color == "red":
        col_str = color_red
    elif color == "blue":
        col_str = color_blue
    elif color == "yellow":
        col_str = color_yellow
    elif color == "orange":
        col_str = color_orange
    return col_str


def iter_print(l):
    if(isinstance(l[0], float)):
        return ','.join(["{0: 0.2f}".format(i) for i in l])
    # return string
    return ','.join([str(i) for i in l])


""" BEGIN configs functions """


def generate_config_from_pos_3(pos_3, dt=0.1, v=0, w=0):
    pos_n11 = np.array([[[pos_3[0], pos_3[1]]]], dtype=np.float32)
    heading_n11 = np.array([[[pos_3[2]]]], dtype=np.float32)
    speed_nk1 = np.ones((1, 1, 1), dtype=np.float32) * v
    angular_speed_nk1 = np.ones((1, 1, 1), dtype=np.float32) * w
    return SystemConfig(dt, 1, 1,
                        position_nk2=pos_n11,
                        heading_nk1=heading_n11,
                        speed_nk1=speed_nk1,
                        angular_speed_nk1=angular_speed_nk1,
                        variable=False)


def generate_random_config(environment, dt=0.1,
                           max_vel=0.6):
    pos_3 = generate_random_pos_in_environment(environment)
    return generate_config_from_pos_3(pos_3, dt=dt, v=max_vel)

# For generating positional arguments in an environment


def generate_random_pos_3(center, xdiff=3, ydiff=3):
    """
    Generates a random position near the center within an elliptical radius of xdiff and ydiff
    """
    offset_x = 2 * xdiff * random.random() - xdiff  # bound by (-xdiff, xdiff)
    offset_y = 2 * ydiff * random.random() - ydiff  # bound by (-ydiff, ydiff)
    offset_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
    return np.add(center, np.array([offset_x, offset_y, offset_theta]))


def within_traversible(new_pos: np.array, traversible: np.array, map_scale: float,
                       stroked_radius: bool = False):
    """
    Returns whether or not the position is in a valid spot in the
    traversible
    """
    pos_x = int(new_pos[0] / map_scale)
    pos_y = int(new_pos[1] / map_scale)
    # Note: the traversible is mapped unintuitively, goes [y, x]
    try:
        if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
            return False
        return True
    except:
        return False


def within_traversible_with_radius(new_pos: np.array, traversible: np.array, map_scale: float, radius: int = 1,
                                   stroked_radius: bool = False):
    """
    Returns whether or not the position is in a valid spot in the
    traversible the Radius input can determine how many surrounding
    spots must also be valid
    """
    for i in range(2 * radius):
        for j in range(2 * radius):
            if(stroked_radius):
                if not((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                    continue
            pos_x = int(new_pos[0] / map_scale) - radius + i
            pos_y = int(new_pos[1] / map_scale) - radius + j
            # Note: the traversible is mapped unintuitively, goes [y, x]
            if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
                return False
    return True


def generate_random_pos_in_environment(environment: dict):
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
    if "human_traversible" in environment.keys():
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
