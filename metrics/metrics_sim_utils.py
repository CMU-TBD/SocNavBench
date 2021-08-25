from typing import Optional

import numpy as np
import pandas as pd
from simulators.simulator import Simulator

from metrics import cost_functions


# meta
def success(central_sim: Simulator) -> bool:
    terminate_cause: str = central_sim.robot.termination_cause
    if terminate_cause == "Pedestrian Collision":
        return False
    if terminate_cause == "Obstacle Collision":
        return False
    elif terminate_cause == "Timeout":
        return False
    elif terminate_cause == "Success":
        return True
    else:
        print(terminate_cause)
        raise ValueError(
            "Unexpected robot termination_cause must be one of Collided/Timeout/Success"
        )
    return False


def total_sim_time_taken(central_sim: Simulator) -> float:
    last_step_num = max(list(central_sim.sim_states.keys()))
    return last_step_num * central_sim.dt


def sim_time_budget(central_sim: Simulator) -> float:
    return central_sim.episode_params.max_time


def termination_cause(central_sim: Simulator) -> str:
    return central_sim.robot.termination_cause


def wall_wait_time(central_sim: Simulator) -> float:
    return central_sim.robot.get_block_t_total()


def map(central_sim: Simulator) -> str:
    return central_sim.episode_params.map_name


# motion
def robot_speed(central_sim: Simulator, percentile: Optional[bool] = False) -> float:
    # extract the bot traj and drop the heading
    robot_trajectory: np.ndarray = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    dt: float = central_sim.dt
    robot_displacement = np.diff(robot_trajectory, axis=0)
    robot_speed = np.sqrt(
        (robot_displacement[:, 0] / dt) ** 2 + (robot_displacement[:, 1] / dt) ** 2
    )

    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_speed


def robot_velocity(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> np.ndarray:
    # extract the bot traj and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    dt = central_sim.dt
    robot_displacement = np.diff(robot_trajectory, axis=0)
    robot_vel = robot_displacement / dt

    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_vel


def robot_acceleration(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> np.ndarray:
    # extract the bot traj and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    dt = central_sim.dt
    robot_displacement = np.diff(robot_trajectory, axis=0)
    robot_vel = robot_displacement / dt
    robot_acc = np.diff(robot_vel, axis=0) / dt

    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_acc


def robot_jerk(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> np.ndarray:
    # extract the bot traj and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    dt = central_sim.dt
    robot_vel = np.diff(robot_trajectory, axis=0) / dt
    robot_acc = np.diff(robot_vel, axis=0) / dt
    robot_jrk = np.diff(robot_acc, axis=0) / dt

    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_jrk


def robot_motion_energy(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> np.ndarray:
    # extract the bot traj and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    dt = central_sim.dt
    robot_displacement = np.diff(robot_trajectory, axis=0)
    robot_motion_energy = np.sum(
        (robot_displacement[:, 0] / dt) ** 2 + (robot_displacement[:, 1] / dt) ** 2
    )

    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_motion_energy


# path
def path_length(central_sim: Simulator, percentile: Optional[bool] = False) -> float:
    # extract the bot traj and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    # robot_goal = np.squeeze(central_sim.robot.goal_config.position_and_heading_nk3())[:-1]
    robot_path_ln = cost_functions.path_length(robot_trajectory)
    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_path_ln


def path_length_ratio(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> float:
    # extract the bot traj and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    robot_goal = np.squeeze(central_sim.robot.goal_config.position_and_heading_nk3())[
        :-1
    ]
    robot_path_ln_ratio = cost_functions.path_length_ratio(
        robot_trajectory, goal_config=robot_goal
    )
    if percentile:
        # TODO run for all peds
        df = central_sim.sim_df
        pass
    return robot_path_ln_ratio


def path_irregularity(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> float:

    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )
    # check if goal was reached
    if central_sim.robot.termination_cause == "Success":
        path_irr = cost_functions.path_irregularity(trajectory=robot_trajectory)
    else:
        goal = central_sim.robot.get_goal_config().position_and_heading_nk3()
        path_irr = cost_functions.path_irregularity(
            trajectory=robot_trajectory, goal_config=goal
        )

    if percentile:
        # TODO run for all
        df = central_sim.sim_df
        pass
    return path_irr


def goal_traversal_ratio(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> float:
    # extract the bot traj and the goal and drop the heading
    robot_trajectory = np.squeeze(
        central_sim.robot.get_trajectory().position_and_heading_nk3()
    )[:, :-1]
    robot_end = robot_trajectory[-1, :]
    robot_start = robot_trajectory[0, :]
    robot_goal = np.squeeze(central_sim.robot.goal_config.position_and_heading_nk3())[
        :-1
    ]
    # robot_path_ln = cost_functions.path_length(robot_trajectory)
    # extract bot dist to goal and bot start to goal
    start_goal_dist = np.linalg.norm(robot_start - robot_goal)
    end_goal_dist = np.linalg.norm(robot_end - robot_goal)

    goal_trav_ratio = end_goal_dist / start_goal_dist

    return goal_trav_ratio


# TODO incorporate radii
def time_to_collision(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> np.ndarray:
    sim_df = central_sim.sim_df
    robot_indcs = sim_df.agent_name == "robot_agent"
    ped_df = central_sim.sim_df[~robot_indcs]
    bot_df = central_sim.sim_df[robot_indcs]
    robot_trajectory = np.vstack([bot_df.x, bot_df.y, bot_df.theta]).T
    # ped_name_df = ped_df.set_index("agent_name")
    dt = central_sim.dt
    robot_displacement = np.diff(robot_trajectory, axis=0)
    robot_inst_vels = robot_displacement[:, :-1] / dt
    # robot_df = ped_df[ped_df.agent_name == "robot_agent"]

    # calculate velocities for pedestrians
    vel_df = ped_df.groupby(["agent_name"])[["x", "y"]].diff().fillna(0) / dt
    vel_df.columns = ["vx", "vy"]
    ped_df = pd.concat([ped_df, vel_df], axis=1)

    # for each time instance in which robot_trajectory exists
    ttc = np.zeros((len(robot_inst_vels)))
    for sim_step in range(len(robot_inst_vels)):
        robot_inst_vel = robot_inst_vels[sim_step - 1]
        sim_step += 1  # velocity is valid only after 2 steps
        # for each bot-ped pair
        # compute the robot-pedestrian relative velocity at each instant
        ped_inst = vel_df[ped_df.sim_step == sim_step]
        if len(ped_inst) == 0:
            ttc[sim_step - 1] = ttc[sim_step - 2]
            continue

        ped_inst_vels = np.array(
            ped_df[ped_df.sim_step == sim_step].loc[:, ("vx", "vy")]
        )
        botped_relative_vels = ped_inst_vels - robot_inst_vel

        # compute the robot-pedestrian joining unit vector
        ped_inst_posns = np.array(
            ped_df[ped_df.sim_step == sim_step].loc[:, ("x", "y")]
        )
        botped_vectors = robot_trajectory[sim_step, :2] - ped_inst_posns
        botped_distances = np.linalg.norm(botped_vectors, axis=1)
        # needs the extra axis to divide correctly
        botped_uvectors = (
            botped_vectors / np.linalg.norm(botped_vectors, axis=1)[:, None]
        )

        # take relative velocity component along the joining vector
        botped_component = np.sum(botped_relative_vels * botped_uvectors, axis=1)

        # see how long it would take to cover that distance w relative velocity
        # infs are fine here because it just means no collision
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc_all = (
                botped_distances - central_sim.robot.get_radius()
            ) / botped_component
        # ttc_all = botped_distances / botped_component
        # discard negative times since there is no collision
        ttc_pos = ttc_all[ttc_all > 0]
        if len(ttc_pos) == 0:  # no collisions
            ttc[sim_step - 1] = -1
        else:
            ttc[sim_step - 1] = np.min(ttc_pos)

    return ttc


def closest_pedestrian_distance(
    central_sim: Simulator, percentile: Optional[bool] = False
) -> np.ndarray:
    sim_df = central_sim.sim_df
    robot_indcs = sim_df.agent_name == "robot_agent"
    ped_df = central_sim.sim_df[~robot_indcs]
    bot_df = central_sim.sim_df[robot_indcs]
    robot_trajectory = np.vstack([bot_df.x, bot_df.y]).T
    # robot_trajectory = np.squeeze(central_sim.robot.get_trajectory().position_and_heading_nk3())[:, :-1]
    # dt = central_sim.dt

    cpd = np.zeros((len(robot_trajectory)))
    for sim_step in range(len(robot_trajectory)):
        ped_inst = ped_df[ped_df.sim_step == sim_step]
        if len(ped_inst) == 0:
            cpd[sim_step] = cpd[sim_step - 1]
            continue
        # compute the robot-pedestrian joining unit vector
        ped_inst_posns = np.vstack([ped_inst.x, ped_inst.y]).T
        botped_vectors = robot_trajectory[sim_step] - ped_inst_posns
        botped_distances = np.linalg.norm(botped_vectors, axis=1)
        cpd[sim_step] = np.min(botped_distances)

    return cpd
