import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from trajectory.trajectory import SystemConfig
from utils.utils import color_text


def test_spline_3rd_order(visualize=False) -> None:
    """
    Create a start and goal states, fit a spline from the two points
    assert tests ensure that the difference between the computed points
    and their manually computed points is very small (they are close)
    """
    np.random.seed(seed=1)
    n = 5
    dt = 0.01
    k = 100

    target_state = np.random.uniform(-np.pi, np.pi, 3)
    v0 = np.random.uniform(0.0, 0.5, 1)[0]  # Initial speed
    vf = 0.0

    # Initial SystemConfig is [0, 0, 0, v0, 0]
    start_speed_nk1 = np.ones((n, 1, 1), dtype=np.float32) * v0

    goal_posx_nk1 = np.ones((n, 1, 1), dtype=np.float32) * target_state[0]
    goal_posy_nk1 = np.ones((n, 1, 1), dtype=np.float32) * target_state[1]
    goal_pos_nk2 = np.concatenate([goal_posx_nk1, goal_posy_nk1], axis=2)
    goal_heading_nk1 = np.ones((n, 1, 1), dtype=np.float32) * target_state[2]
    goal_speed_nk1 = np.ones((n, 1, 1), dtype=np.float32) * vf

    start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1)
    goal_config = SystemConfig(
        dt=dt,
        n=n,
        k=1,
        position_nk2=goal_pos_nk2,
        speed_nk1=goal_speed_nk1,
        heading_nk1=goal_heading_nk1,
    )

    start_nk5 = start_config.position_heading_speed_and_angular_speed_nk5()
    start_n5 = start_nk5[:, 0]

    goal_nk5 = goal_config.position_heading_speed_and_angular_speed_nk5()
    goal_n5 = goal_nk5[:, 0]

    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    ts_nk = np.tile(np.linspace(0.0, dt * k, k)[None], [n, 1])
    spline_traj = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
    spline_traj.fit(start_config, goal_config, factors=None)
    spline_traj.eval_spline(ts_nk, calculate_speeds=True)

    pos_nk3 = spline_traj.position_and_heading_nk3()
    v_nk1 = spline_traj.speed_nk1()
    start_pos_diff = (pos_nk3 - start_n5[:, None, :3])[:, 0]
    goal_pos_diff = (pos_nk3 - goal_n5[:, None, :3])[:, -1]
    assert np.allclose(start_pos_diff, np.zeros((n, 3)), atol=1e-6)
    assert np.allclose(goal_pos_diff, np.zeros((n, 3)), atol=1e-6)

    start_vel_diff = (v_nk1 - start_n5[:, None, 3:4])[:, 0]
    goal_vel_diff = (v_nk1 - goal_n5[:, None, 3:4])[:, -1]
    assert np.allclose(start_vel_diff, np.zeros((n, 1)), atol=1e-6)
    assert np.allclose(goal_vel_diff, np.zeros((n, 1)), atol=1e-6)

    if visualize:
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        fig, ax = plt.subplots(1, 1, squeeze=False)

        spline_traj.render(ax, freq=4)
        # plt.show()
        fig.savefig("./tests/spline/test_spline.png", bbox_inches="tight", pad_inches=0)


def test_piecewise_spline(visualize=False) -> None:
    # Testing with splines
    np.random.seed(seed=1)
    n = 5  # Batch size (unused now)
    dt = 0.01  # delta-t: time intervals
    k = 100  # Number of time-steps (should be 1/dt)

    # States represents each individual tangent point that the spline will pass through
    # states[0] = initial state, and states[len(states) - 1] = terminal state
    states = [
        (8, 12, np.pi / 2.0, 0.5),  # Start State (x, y, theta, vel)
        (15, 14.5, np.pi / 2.0, 0.8),  # Middle State (x, y, theta, vel)
        (10, 15, -np.pi / 2.0, 1),  # Middle State (x, y, theta, vel)
        (18, 16.5, np.pi / 2.0, 0.2),  # Goal State (x, y, theta, vel)
    ]

    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    ts_nk = np.tile(np.linspace(0.0, dt * k, k)[None], [n, 1])
    splines = []
    prev_config = None
    next_config = None
    # Generate all two-point splines from the states
    for s in states:
        spline_traj = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
        # Keep track of old trajectory
        if next_config is not None:
            # Rewrite the previous state config with the 'old' next one
            prev_config = next_config
        # Generate position
        s_posx_nk1 = np.ones((n, 1, 1), dtype=np.float32) * s[0]  # X position matrix
        s_posy_nk1 = np.ones((n, 1, 1), dtype=np.float32) * s[1]  # Y position matrix
        # combined matrix of (X,Y)
        s_pos_nk2 = np.concatenate([s_posx_nk1, s_posy_nk1], axis=2)
        # Generate speed and heading
        heading_nk1 = np.ones((n, 1, 1), dtype=np.float32) * s[2]  # Theta angle matrix
        speed_nk1 = np.ones((n, 1, 1), dtype=np.float32) * s[3]  # Speed matrix
        next_config = SystemConfig(
            dt=dt,
            n=n,
            k=1,
            position_nk2=s_pos_nk2,
            speed_nk1=speed_nk1,
            heading_nk1=heading_nk1,
        )
        # Append to the trajectory if a new trajectory can be constructed
        # Note that any spline needs a 'previous' and 'next' state
        if prev_config is not None:
            spline_traj.fit(prev_config, next_config, factors=None)
            spline_traj.eval_spline(ts_nk, calculate_speeds=True)
            splines.append(spline_traj)

    # Loop through all calculated splines to combine them together into a singular one
    final_spline = None
    for i, s in enumerate(splines):
        if final_spline is not None:
            final_spline.append_along_time_axis(s)
        if i == 0:
            # For first spline
            final_spline = s
    if visualize:
        fig = plt.figure()
        fig, ax = plt.subplots(4, 1, figsize=(5, 15), squeeze=False)
        final_spline.render_multi(
            ax, freq=4, plot_heading=True, plot_velocity=True, label_start_and_end=True
        )
        # trajectory.render(ax, freq=1, plot_heading=True, plot_velocity=True, label_start_and_end=True)
        fig.savefig(
            "./tests/spline/test_piecewise_spline.png",
            bbox_inches="tight",
            pad_inches=0,
        )


def test_spline_rescaling() -> None:
    # Set the random seed
    np.random.seed(seed=1)

    # Spline trajectory params
    n = 2
    dt = 0.1
    k = 10
    final_times_n1 = np.array([[2.0], [1.0]])

    # Goal states and initial speeds
    goal_posx_n11 = np.array([[[0.4]], [[1.0]]])
    goal_posy_n11 = np.array([[[0.0]], [[1.0]]])
    goal_heading_n11 = np.array([[[0.0]], [[np.pi / 2]]])
    start_speed_nk1 = np.ones((2, 1, 1), dtype=np.float32) * 0.5

    # Define the maximum speed, angular speed and maximum horizon
    max_speed = 0.6
    max_angular_speed = 1.1

    # Define start and goal configurations
    start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1)
    goal_config = SystemConfig(
        dt=dt,
        n=n,
        k=1,
        position_nk2=np.concatenate([goal_posx_n11, goal_posy_n11], axis=2),
        heading_nk1=goal_heading_n11,
    )

    # Fit the splines
    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    spline_trajs = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
    spline_trajs.fit(start_config, goal_config, final_times_n1, factors=None)

    # Evaluate the splines
    ts_nk = np.stack(
        [
            np.linspace(0.0, final_times_n1[0, 0], 100),
            np.linspace(0.0, final_times_n1[1, 0], 100),
        ],
        axis=0,
    )
    spline_trajs.eval_spline(ts_nk, calculate_speeds=True)

    # Compute the required horizon
    required_horizon_n1 = spline_trajs.compute_dynamically_feasible_horizon(
        max_speed, max_angular_speed
    )
    assert required_horizon_n1[0, 0] < final_times_n1[0, 0]
    assert required_horizon_n1[1, 0] > final_times_n1[1, 0]

    # Compute the maximum speed and angular speed
    max_speed_n1 = np.amax(spline_trajs.speed_nk1(), axis=1)
    max_angular_speed_n1 = np.amax(np.abs(spline_trajs.angular_speed_nk1()), axis=1)
    assert max_speed_n1[0, 0] < max_speed
    assert max_angular_speed_n1[0, 0] < max_angular_speed
    assert max_speed_n1[1, 0] > max_speed
    assert max_angular_speed_n1[1, 0] > max_angular_speed

    # Rescale horizon so that the trajectories are dynamically feasible
    spline_trajs.rescale_spline_horizon_to_dynamically_feasible_horizon(
        max_speed, max_angular_speed
    )
    assert np.allclose(spline_trajs.final_times_n1, required_horizon_n1, atol=1e-2)

    # Compute the maximum speed and angular speed
    max_speed_n1 = np.amax(spline_trajs.speed_nk1(), axis=1)
    max_angular_speed_n1 = np.amax(np.abs(spline_trajs.angular_speed_nk1()), axis=1)
    assert max_speed_n1[0, 0] <= max_speed
    assert max_angular_speed_n1[0, 0] <= max_angular_speed
    assert max_speed_n1[1, 0] <= max_speed
    assert max_angular_speed_n1[1, 0] <= max_angular_speed

    # Find the spline trajectories that are valid
    valid_idxs_n = spline_trajs.find_trajectories_within_a_horizon(horizon_s=2.0)
    assert valid_idxs_n.shape == (1,)
    assert valid_idxs_n[0] == 0


def main_test() -> None:
    test_spline_3rd_order(visualize=False)
    test_spline_rescaling()
    test_piecewise_spline(visualize=False)
    print("%sSpline tests passed!%s" % (color_text["green"], color_text["reset"]))


if __name__ == "__main__":
    main_test()
