import json
import os
import socket
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from dotmap import DotMap
from utils.utils import color_text, euclidean_dist2, iter_print

lock = threading.Lock()  # for asynchronous data sending


def clip_vel(vel: float, bounds: Tuple[float, float]) -> float:
    vel = round(float(vel), 3)
    assert bounds[0] < bounds[1]
    if bounds[0] <= vel <= bounds[1]:
        return vel
    clipped = min(max(bounds[0], vel), bounds[1])
    print(
        "%svelocity %s out of bounds, clipped to %s%s"
        % (color_text["red"], vel, clipped, color_text["reset"])
    )
    return clipped


def clip_posn(
    sim_dt: float,
    old_pos3: List[float],
    new_pos3: List[float],
    v_bounds: Tuple[float, float],
    epsilon: Optional[float] = 0.01,
) -> List[float]:
    # margin of error for the velocity bounds
    assert sim_dt > 0
    dist_to_new = euclidean_dist2(old_pos3, new_pos3)
    req_vel = abs(dist_to_new / sim_dt)
    if req_vel <= v_bounds[1] + epsilon:
        return new_pos3
    # calculate theta of vector
    valid_theta = np.arctan2(new_pos3[1] - old_pos3[1], new_pos3[0] - old_pos3[0])
    # create new position scaled off the invalid one
    max_vel = sim_dt * v_bounds[1]
    valid_x = max_vel * np.cos(valid_theta) + old_pos3[0]
    valid_y = max_vel * np.sin(valid_theta) + old_pos3[1]
    reachable_pos3 = [valid_x, valid_y, valid_theta]
    print(
        "%sposn [%s] is unreachable, clipped to [%s] (%.3fm/s > %.3fm/s)%s"
        % (
            color_text["red"],
            iter_print(new_pos3),
            iter_print(reachable_pos3),
            req_vel,
            v_bounds[1],
            color_text["reset"],
        )
    )
    return reachable_pos3


"""MORE socket utils"""


def establish_joystick_receiver_connection(
    sock_id: str,
) -> Tuple[socket.socket, socket.socket, str]:
    """Connect to server (robot)"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(sock_id)
    except OSError:
        # clear sockets to be used
        if os.path.exists(sock_id):
            os.remove(sock_id)
            sock.bind(sock_id)
    # wait for a connection
    sock.listen(1)
    print("Waiting for Joystick connection...")
    connection, client = sock.accept()
    print(
        "%sRobot <-- Joystick (receiver) connection established%s"
        % (color_text["green"], color_text["reset"])
    )
    return sock, connection, client


def establish_joystick_sender_connection(sock_id: str) -> socket.socket:
    """Connect to client (joystick)"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(sock_id)
    except Exception as e:
        print(
            "%sUnable to connect to joystick%s. Reason: %s"
            % (color_text["red"], color_text["reset"], e)
        )
        print("Make sure you have a joystick instance running")
        exit(1)
    assert sock is not None
    print(
        "%sRobot --> Joystick (sender) connection established%s"
        % (color_text["green"], color_text["reset"])
    )
    return sock


def close_sockets(socks: List[socket.socket]) -> None:
    for sock in socks:
        sock.close()


def force_connect(robot_receiver_id: str) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # connect to the socket to break the accept() wait
    s.connect(robot_receiver_id)


def establish_handshake(
    p: DotMap, sender_id: str, receiver_id: str,
) -> Tuple[socket.socket, socket.socket]:
    # NOTE: this is from the robot's POV
    if p.episode_params.without_robot:
        # lite-mode episode does not include a robot or joystick
        return
    receiver_sock, _, _ = establish_joystick_receiver_connection(receiver_id)
    time.sleep(0.01)
    sender_sock = establish_joystick_sender_connection(sender_id)
    # send the preliminary episodes that the socnav is going to run
    json_dict = {}
    json_dict["episodes"] = list(p.episode_params.tests.keys())
    episodes = json.dumps(json_dict)
    # Create a TCP/IP socket
    send_episodes_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Connect the socket to the port where the server is listening
    send_episodes_socket.connect(sender_id)
    send_episodes_socket.sendall(bytes(episodes, "utf-8"))
    send_episodes_socket.close()
    return (receiver_sock, sender_sock)


""" END socket utils """
