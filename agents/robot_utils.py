import numpy as np
import json
import socket
import threading
from utils.utils import *

lock = threading.Lock()  # for asynchronous data sending


def clip_vel(vel, bounds):
    vel = round(float(vel), 3)
    assert(bounds[0] < bounds[1])
    if(bounds[0] <= vel <= bounds[1]):
        return vel
    clipped = min(max(bounds[0], vel), bounds[1])
    print("%svelocity %s out of bounds, clipped to %s%s" %
          (color_red, vel, clipped, color_reset))
    return clipped


def clip_posn(sim_dt: float, old_pos3: list, new_pos3: list, v_bounds: list, epsilon: float = 0.01):
    # margin of error for the velocity bounds
    assert(sim_dt > 0)
    dist_to_new = euclidean_dist2(old_pos3, new_pos3)
    if(abs(dist_to_new / sim_dt) <= v_bounds[1] + epsilon):
        return new_pos3
    # calculate theta of vector
    valid_theta = \
        np.arctan2(new_pos3[1] - old_pos3[1], new_pos3[0] - old_pos3[0])
    # create new position scaled off the invalid one
    max_vel = sim_dt * v_bounds[1]
    valid_x = max_vel * np.cos(valid_theta) + old_pos3[0]
    valid_y = max_vel * np.sin(valid_theta) + old_pos3[1]
    reachable_pos3 = [valid_x, valid_y, valid_theta]
    print("%sposition [%s] is unreachable with v bounds, clipped to [%s]%s" %
          (color_red, iter_print(new_pos3), iter_print(reachable_pos3), color_reset))
    return reachable_pos3


"""BEGIN socket utils"""

joystick_receiver_socket = None
joystick_sender_socket = None
host = '127.0.0.1'  # localhost
port_send = None  # to be added later from params (see establish_handshake())
port_recv = None  # to be added later from params (see establish_handshake())


def send_sim_state(robot):
    # send the (JSON serialized) world state per joystick's request
    if robot.joystick_requests_world == 0:
        world_state = robot.world_state.to_json(
            robot_on=not robot.get_end_acting()
        )
        send_to_joystick(world_state)
        # immediately note that the world has been sent:
        robot.joystick_requests_world = -1


def send_to_joystick(message: str):
    with lock:
        assert(isinstance(message, str))
        global joystick_sender_socket
        # Create a TCP/IP socket
        joystick_sender_socket = \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((host, port_send))
        try:
            joystick_sender_socket.connect(server_address)
        except ConnectionRefusedError:
            # abort and dont send data
            return
        # Send data
        joystick_sender_socket.sendall(bytes(message, "utf-8"))
        joystick_sender_socket.close()


def listen_once(robot):
    """Constantly connects to the robot listener socket and receives information from the
    joystick about the input commands as well as the world requests
    """
    global joystick_receiver_socket
    connection, _ = joystick_receiver_socket.accept()
    data_b, response_len = conn_recv(connection, buffr_amnt=128)
    # close connection to be reaccepted when the joystick sends data
    connection.close()
    if(data_b is not b'' and response_len > 0):
        data_str = data_b.decode("utf-8")  # bytes to str
        if(robot.get_end_acting()):
            robot.joystick_requests_world = 0
        else:
            manage_data(robot, data_str)


def is_keyword(robot, data_str: str):
    # non json important keyword
    if(data_str == "sense"):
        robot.joystick_requests_world = \
            len(robot.joystick_inputs) - (robot.num_executed)
        return True
    elif(data_str == "ready"):
        robot.joystick_ready = True
        return True
    elif("algo: " in data_str):
        robot.algo_name = data_str[len("algo: "):]
        return True
    elif(data_str == "abandon"):
        robot.power_off()
        return True
    return False


def manage_data(robot, data_str: str):
    if not is_keyword(robot, data_str):
        data = json.loads(data_str)
        joystick_input: list = data["j_input"]
        robot.num_cmds_per_batch = len(joystick_input)
        # add input commands to queue to keep track of
        for i in range(robot.num_cmds_per_batch):
            np_data = np.array(joystick_input[i], dtype=np.float32)
            # duplicate commands if *repeating* instead of blocking
            if robot.repeat_joystick:  # if need be, repeat n-1 times
                repeat_amnt = robot.calc_repeat_freq()
                for i in range(repeat_amnt):
                    # adds command to local list of individual commands
                    robot.joystick_inputs.append(np_data)
            else:
                # else no repeat, only account for the command once
                robot.joystick_inputs.append(np_data)


def establish_joystick_receiver_connection():
    """This is akin to a server connection (robot is server)"""
    global joystick_receiver_socket
    joystick_receiver_socket = \
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    joystick_receiver_socket.bind((host,
                                   port_recv))
    # wait for a connection
    joystick_receiver_socket.listen(1)
    print("Waiting for Joystick connection...")
    connection, client = joystick_receiver_socket.accept()
    print("%sRobot <-- Joystick (receiver) connection established%s" %
          (color_green, color_reset))
    return connection, client


def establish_joystick_sender_connection():
    """This is akin to a client connection (joystick is client)"""
    global joystick_sender_socket
    joystick_sender_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_STREAM)
    address = ((host, port_send))
    try:
        joystick_sender_socket.connect(address)
    except:
        print("%sUnable to connect to joystick%s" %
              (color_red, color_reset))
        print("Make sure you have a joystick instance running")
        exit(1)
    assert(joystick_sender_socket is not None)
    print("%sRobot --> Joystick (sender) connection established%s" %
          (color_green, color_reset))


def close_sockets():
    global joystick_sender_socket
    global joystick_receiver_socket
    joystick_sender_socket.close()
    joystick_receiver_socket.close()


def force_connect():
    global host
    global port_recv
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to the socket to break the accept() wait
    s.connect((host, port_recv))


def establish_handshake(p: DotMap):
    if(p.episode_params.without_robot):
        # lite-mode episode does not include a robot or joystick
        return
    import time
    # sockets for communication
    global port_recv
    global port_send
    global host
    # port for recieving commands from the joystick
    port_recv = p.robot_params.port
    # port for sending commands to the joystick (successor of port_recv)
    port_send = port_recv + 1
    establish_joystick_receiver_connection()
    time.sleep(0.01)
    establish_joystick_sender_connection()
    # send the preliminary episodes that the socnav is going to run
    json_dict = {}
    json_dict['episodes'] = list(p.episode_params.tests.keys())
    episodes = json.dumps(json_dict)
    # Create a TCP/IP socket
    send_episodes_socket = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)
    # Connect the socket to the port where the server is listening
    server_address = ((host, port_send))
    send_episodes_socket.connect(server_address)
    send_episodes_socket.sendall(bytes(episodes, "utf-8"))
    send_episodes_socket.close()


""" END socket utils """
