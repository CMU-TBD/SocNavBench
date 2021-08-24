import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
from systems.dubins_v1 import DubinsV1
from systems.dubins_v2 import DubinsV2
from systems.dubins_v3 import DubinsV3
from utils.utils import color_text


def create_system_dynamics_params() -> DotMap:
    p = DotMap()

    p.v_bounds = [0.0, 0.6]
    p.w_bounds = [-1.1, 1.1]

    p.simulation_params = DotMap(
        simulation_mode="ideal",
        noise_params=DotMap(
            is_noisy=False,
            noise_type="uniform",
            noise_lb=[-0.02, -0.02, 0.0],
            noise_ub=[0.02, 0.02, 0.0],
            noise_mean=[0.0, 0.0, 0.0],
            noise_std=[0.02, 0.02, 0.0],
        ),
    )
    return p


def test_dubins_v1(visualize=False) -> None:
    np.random.seed(seed=1)
    dt = 0.1
    n, k = 5, 20
    x_dim, u_dim = 3, 2

    # Test that All Dimensions Work
    db = DubinsV1(dt, create_system_dynamics_params())

    state_nk3 = np.array(np.zeros((n, k, x_dim), dtype=np.float32))
    ctrl_nk2 = np.array(np.random.randn(n, k, u_dim), dtype=np.float32)

    trajectory = db.assemble_trajectory(state_nk3, ctrl_nk2)
    state_tp1_nk3 = db.simulate(state_nk3, ctrl_nk2)
    assert state_tp1_nk3.shape == (n, k, x_dim)
    jac_x_nk33 = db.jac_x(trajectory)
    assert jac_x_nk33.shape == (n, k, x_dim, x_dim)

    jac_u_nk32 = db.jac_u(trajectory)
    assert jac_u_nk32.shape == (n, k, x_dim, u_dim)

    A, B, c = db.affine_factors(trajectory)

    # Test that computation is occurring correctly
    n, k = 2, 3
    ctrl = 1
    state_n13 = np.array(np.zeros((n, 1, x_dim)), dtype=np.float32)
    ctrl_nk2 = np.array(np.ones((n, k, u_dim)) * ctrl, dtype=np.float32)
    trajectory = db.simulate_T(state_n13, ctrl_nk2, T=k)
    state_nk3 = trajectory.position_and_heading_nk3()

    x1, x2, x3, x4 = (
        state_nk3[0, 0],
        state_nk3[0, 1],
        state_nk3[0, 2],
        state_nk3[0, 3],
    )
    assert (x1 == np.zeros(3)).all()
    assert np.allclose(x2, [0.1, 0.0, 0.1])
    assert np.allclose(x3, [0.1 + 0.1 * np.cos(0.1), 0.1 * np.sin(0.1), 0.2])
    assert np.allclose(x4, [0.2975, 0.0298, 0.3], atol=1e-4)

    trajectory = db.assemble_trajectory(state_nk3[:, :-1], ctrl_nk2)
    A, B, c = db.affine_factors(trajectory)
    A0, A1, A2 = A[0, 0], A[0, 1], A[0, 2]
    A0_c = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])
    A1_c = np.array(
        [[1.0, 0.0, -0.1 * np.sin(0.1)], [0.0, 1.0, 0.1 * np.cos(0.1)], [0.0, 0.0, 1.0]]
    )
    A2_c = np.array(
        [[1.0, 0.0, -0.1 * np.sin(0.2)], [0.0, 1.0, 0.1 * np.cos(0.2)], [0.0, 0.0, 1.0]]
    )
    assert np.allclose(A0, A0_c)
    assert np.allclose(A1, A1_c)
    assert np.allclose(A2, A2_c)

    B0, B1, B2 = B[0, 0], B[0, 1], B[0, 2]
    B0_c = np.array([[0.1, 0.0], [0.0, 0.0], [0.0, 0.1]])
    B1_c = np.array([[0.1 * np.cos(0.1), 0.0], [0.1 * np.sin(0.1), 0.0], [0.0, 0.1]])
    B2_c = np.array([[0.1 * np.cos(0.2), 0.0], [0.1 * np.sin(0.2), 0.0], [0.0, 0.1]])
    assert np.allclose(B0, B0_c)
    assert np.allclose(B1, B1_c)
    assert np.allclose(B2, B2_c)

    if visualize:
        # Visualize One Trajectory for Debugging
        k = 50
        state_113 = np.array(np.zeros((1, 1, x_dim)), dtype=np.float32)
        v_1k, w_1k = np.ones((k, 1)) * 0.2, np.linspace(1.1, 0.9, k)[:, None]
        ctrl_1k2 = np.array(
            np.concatenate([v_1k, w_1k], axis=1)[None], dtype=np.float32
        )
        trajectory = db.simulate_T(state_113, ctrl_1k2, T=k)
        state_1k3, _ = db.parse_trajectory(trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs, ys, ts = state_1k3[0, :, 0], state_1k3[0, :, 1], state_1k3[0, :, 2]
        ax.plot(xs, ys, "r--")
        ax.quiver(xs, ys, np.cos(ts), np.sin(ts))
        # plt.show()
        fig.savefig(
            "./tests/dynamics/test_dynamics1.png", bbox_inches="tight", pad_inches=0
        )


def rand_array(from_range, size, decimals=3) -> np.int64:
    """
    Return a size='size' array with values in between 'from_range'
    with 'decimals' decimal places
    """
    np.random.seed(seed=2)
    return np.random.randint(
        from_range[0] * (10 ** decimals), from_range[1] * (10 ** decimals), size=size
    ) / (10.0 ** decimals)


def test_custom_dubins_v1(plot=False) -> None:
    # Visualize One Trajectory for Debugging
    dt = 0.1
    n, k = 5, 50
    x_dim, u_dim = 3, 2
    # Generate Dubin's Model
    db = DubinsV1(dt, create_system_dynamics_params())

    # Generate Start State (x0, y0, t0)
    state_113 = np.array([[[0, 0, 0]]], dtype=np.float32)
    # Generate linear velocity controls
    v0 = 0.8
    t0 = 0.5
    v_1k = np.ones((k, 1)) * v0
    # v_1k = rand_array(from_range=(0, 10), size=(k,1))
    # Generate angular velocity controls
    # Randomly generate directions to control
    # w_1k = rand_array(from_range=(-10, 10), size=(k,1))
    w_1k = np.linspace(1, 10, k)[:, None]
    # w_1k = np.ones((k, 1)) * t0 # np.linspace(0.9, 1.1, k)[:, None]
    ctrl_1k2 = np.array(np.concatenate([v_1k, w_1k], axis=1)[None], dtype=np.float32)

    trajectory = db.simulate_T(state_113, ctrl_1k2, T=k)
    state_1k3, _ = db.parse_trajectory(trajectory)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs, ys, ts = state_1k3[0, :, 0], state_1k3[0, :, 1], state_1k3[0, :, 2]
        ax.plot(xs, ys, "r--")
        ax.quiver(xs, ys, np.cos(ts), np.sin(ts))
        # plt.show()
        fig.savefig(
            "./tests/dynamics/custom_dubins.png", bbox_inches="tight", pad_inches=0
        )


def test_dubins_v2(visualize=False) -> None:
    np.random.seed(seed=1)
    dt = 0.1
    x_dim, u_dim = 3, 2
    n, k = 17, 12
    ctrl = 1

    # Test that computation is occurring correctly
    db = DubinsV2(dt, create_system_dynamics_params())
    state_n13 = np.array(np.zeros((n, 1, x_dim)), dtype=np.float32)
    ctrl_nk2 = np.array(np.ones((n, k, u_dim)) * ctrl, dtype=np.float32)
    trajectory = db.simulate_T(state_n13, ctrl_nk2, T=k)
    state_nk3 = trajectory.position_and_heading_nk3()

    x1, x2, x3, x4 = (
        state_nk3[0, 0],
        state_nk3[0, 1],
        state_nk3[0, 2],
        state_nk3[0, 3],
    )
    assert (x1 == np.zeros(3)).all()
    assert np.allclose(x2, [0.06, 0.0, 0.1])
    assert np.allclose(x3, [0.06 + 0.06 * np.cos(0.1), 0.06 * np.sin(0.1), 0.2])
    assert np.allclose(x4, [0.17850246, 0.01791017, 0.3], atol=1e-4)

    trajectory = db.assemble_trajectory(state_nk3[:, :-1], ctrl_nk2)
    A, B, c = db.affine_factors(trajectory)
    A0, A1, A2 = A[0, 0], A[0, 1], A[0, 2]
    A0_c = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.06], [0.0, 0.0, 1.0]])
    A1_c = np.array(
        [
            [1.0, 0.0, -0.06 * np.sin(0.1)],
            [0.0, 1.0, 0.06 * np.cos(0.1)],
            [0.0, 0.0, 1.0],
        ]
    )
    A2_c = np.array(
        [
            [1.0, 0.0, -0.06 * np.sin(0.2)],
            [0.0, 1.0, 0.06 * np.cos(0.2)],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(A0, A0_c)
    assert np.allclose(A1, A1_c)
    assert np.allclose(A2, A2_c)

    B0, B1, B2 = B[0, 0], B[0, 1], B[0, 2]
    B0_c = np.array([[0.1, 0.0], [0.0, 0.0], [0.0, 0.1]])
    B1_c = np.array([[0.1 * np.cos(0.1), 0.0], [0.1 * np.sin(0.1), 0.0], [0.0, 0.1]])
    B2_c = np.array([[0.1 * np.cos(0.2), 0.0], [0.1 * np.sin(0.2), 0.0], [0.0, 0.1]])
    assert np.allclose(B0, B0_c)
    assert np.allclose(B1, B1_c)
    assert np.allclose(B2, B2_c)

    if visualize:
        # Visualize One Trajectory for Debugging
        k = 50
        state_113 = np.array(np.zeros((1, 1, x_dim)), dtype=np.float32)
        v_1k, w_1k = np.ones((k, 1)) * 0.2, np.linspace(1.1, 0.9, k)[:, None]
        ctrl_1k2 = np.array(
            np.concatenate([v_1k, w_1k], axis=1)[None], dtype=np.float32
        )
        trajectory = db.simulate_T(state_113, ctrl_1k2, T=k)
        state_1k3, _ = db.parse_trajectory(trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs, ys, ts = state_1k3[0, :, 0], state_1k3[0, :, 1], state_1k3[0, :, 2]
        ax.plot(xs, ys, "r--")
        ax.quiver(xs, ys, np.cos(ts), np.sin(ts))
        # plt.show()
        fig.savefig(
            "./tests/dynamics/test_dynamics2.png", bbox_inches="tight", pad_inches=0
        )


def test_dubins_v3() -> None:
    np.random.seed(seed=1)
    dt = 0.1
    x_dim, u_dim = 5, 2
    n, k = 17, 12
    ctrl = 1

    # Test that computation is occurring correctly
    db = DubinsV3(dt, create_system_dynamics_params())
    state_n15 = np.array(np.zeros((n, 1, x_dim)), dtype=np.float32)
    ctrl_nk2 = np.array(np.ones((n, k, u_dim)) * ctrl, dtype=np.float32)
    trajectory = db.simulate_T(state_n15, ctrl_nk2, T=k)
    state_nk5 = trajectory.position_heading_speed_and_angular_speed_nk5()

    x2, x3, x4 = state_nk5[0, 1], state_nk5[0, 2], state_nk5[0, 3]

    assert np.allclose(x2, [0.0, 0.0, 0.0, 0.1, 0.1])
    assert np.allclose(x3, [0.01, 0.0, 0.01, 0.2, 0.2])
    assert np.allclose(
        x4,
        [np.cos(0.01) * 0.1 * 0.2 + 0.01, np.sin(0.01) * 0.1 * 0.2, 0.03, 0.3, 0.3],
        atol=1e-4,
    )

    trajectory = db.assemble_trajectory(state_nk5[:, :-1], ctrl_nk2)
    A, B, c = db.affine_factors(trajectory)
    A0, A1, A2 = A[0, 0], A[0, 1], A[0, 2]

    A0_c = np.eye(5)
    A0_c[0, 3] += 0.1
    A0_c[2, 4] += dt
    A1_c = np.eye(5)
    A1_c[0, 3] += 0.1
    A1_c[1, 2] += 0.01
    A1_c[2, 4] += dt
    A2_c = np.eye(5)
    A2_c[2, 4] += dt
    A2_c[0, 2] += -0.2 * np.sin(0.01) * dt
    A2_c[1, 2] += 0.2 * np.cos(0.01) * dt
    A2_c[0, 3] += dt * np.cos(0.01)
    A2_c[1, 3] += dt * np.sin(0.01)
    assert np.allclose(A0, A0_c)
    assert np.allclose(A1, A1_c)
    assert np.allclose(A2, A2_c)

    B0, B1, B2 = B[0, 0], B[0, 1], B[0, 2]
    B0_c = np.zeros((x_dim, u_dim))
    B0_c[3, 0] += 0.1
    B0_c[4, 1] += 0.1
    B1_c = 1.0 * B0_c
    B2_c = 1.0 * B0_c
    assert np.allclose(B0, B0_c)
    assert np.allclose(B1, B1_c)
    assert np.allclose(B2, B2_c)


def main_test() -> None:
    test_dubins_v1(visualize=False)
    test_custom_dubins_v1()
    test_dubins_v2(visualize=False)
    test_dubins_v3()
    print("%sDynamics tests passed!%s" % (color_text["green"], color_text["reset"]))


if __name__ == "__main__":
    main_test()
