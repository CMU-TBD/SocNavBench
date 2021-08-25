import matplotlib.pyplot as plt
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from dotmap import DotMap
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import DubinsV1
from utils.utils import color_text


def create_params() -> DotMap:
    p = DotMap()
    p.seed = 1
    p.n = 5
    p.k = 20
    p.map_bounds = [[0.0, 0.0], [4.0, 4.0]]
    p.dx, p.dt = 0.05, 0.1

    p.quad_coeffs = [1.0, 1.0, 1.0, 1e-10, 1e-10]
    p.linear_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    p.system_dynamics_params = DotMap(
        system=DubinsV1, dt=0.1, v_bounds=[0.0, 0.6], w_bounds=[-1.1, 1.1]
    )
    p.system_dynamics_params.simulation_params = DotMap(
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


def test_lqr0(visualize=False) -> None:
    p = create_params()
    np.random.seed(seed=p.seed)
    n, k = 1, p.k
    dt = p.dt

    db = DubinsV1(dt, params=p.system_dynamics_params)
    x_dim, u_dim = db._x_dim, db._u_dim

    goal_x, goal_y = 4.0, 0.0
    goal = np.array([goal_x, goal_y, 0.0], dtype=np.float32)
    x_ref_nk3 = np.array(np.tile(goal, (n, k, 1)))
    u_ref_nk2 = np.array(np.zeros((n, k, u_dim), dtype=np.float32))
    trajectory_ref = db.assemble_trajectory(x_ref_nk3, u_ref_nk2)

    cost_fn = QuadraticRegulatorRef(trajectory_ref, db, p)

    x_nk3 = np.array(np.zeros((n, k, x_dim), dtype=np.float32))
    u_nk2 = np.array(np.zeros((n, k, u_dim), dtype=np.float32))
    # Initiate a blank trajectory (all 0's)
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)

    lqr_solver = LQRSolver(T=k - 1, dynamics=db, cost=cost_fn)
    cost = lqr_solver.evaluate_trajectory_cost(trajectory)
    expected_cost = 0.5 * goal_x ** 2 * k + 0.5 * goal_y ** 2 * k
    assert (cost == expected_cost).all()

    start_config = db.init_egocentric_robot_config(dt=dt, n=n)
    lqr_res = lqr_solver.lqr(start_config, trajectory, verbose=False)
    trajectory_opt = lqr_res["trajectory_opt"]
    J_opt = lqr_res["J_hist"][-1]
    assert abs(J_opt[0] - 8.0) <= 1e-4
    assert np.allclose(trajectory_opt.position_nk2()[:, 1:, 0], 4.0)

    if visualize:
        pos_ref = trajectory_ref.position_nk2()[0]
        pos_opt = trajectory_opt.position_nk2()[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pos_ref[:, 0], pos_ref[:, 1])
        ax.plot(pos_opt[:, 0], pos_opt[:, 1], "b--", label="opt")
        ax.legend()
        # plt.show()
        fig.savefig("./tests/lqr/test_lqr0.png", bbox_inches="tight", pad_inches=0)


def test_lqr1(visualize=False) -> None:
    p = create_params()
    np.random.seed(seed=p.seed)
    n, k = p.n, 50
    dt = p.dt

    db = DubinsV1(dt, params=p.system_dynamics_params)
    x_dim, u_dim = db._x_dim, db._u_dim

    x_n13 = np.array(np.zeros((n, 1, x_dim)), dtype=np.float32)
    v_1k = np.ones((k - 1, 1)) * 0.1
    w_1k = np.linspace(0.5, 0.3, k - 1)[:, None]

    u_1k2 = np.array(np.concatenate([v_1k, w_1k], axis=1)[None], dtype=np.float32)
    u_nk2 = np.zeros((n, k - 1, 2), dtype=np.float32) + u_1k2
    trajectory_ref = db.simulate_T(x_n13, u_nk2, T=k)

    cost_fn = QuadraticRegulatorRef(trajectory_ref, db, p)

    x_nk3 = np.array(np.zeros((n, k, x_dim), dtype=np.float32))
    u_nk2 = np.array(np.zeros((n, k, u_dim), dtype=np.float32))
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)

    lqr_solver = LQRSolver(T=k - 1, dynamics=db, cost=cost_fn)

    start_config = db.init_egocentric_robot_config(dt=dt, n=n)
    lqr_res = lqr_solver.lqr(start_config, trajectory, verbose=False)
    trajectory_opt = lqr_res["trajectory_opt"]
    assert np.abs(lqr_res["J_hist"][1][0] - 0.022867) < 1e-4
    assert np.abs(lqr_res["J_hist"][0][0] - 38.17334) < 1e-4

    if visualize:
        pos_ref = trajectory_ref.position_nk2()[0]
        pos_opt = trajectory_opt.position_nk2()[0]
        heading_ref = trajectory_ref.heading_nk1()[0]
        heading_opt = trajectory_opt.heading_nk1()[0]
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], "r-", label="ref")
        ax.quiver(
            pos_ref[:, 0], pos_ref[:, 1], np.cos(heading_ref), np.sin(heading_ref)
        )
        ax.plot(pos_opt[:, 0], pos_opt[:, 1], "b-", label="opt")
        ax.quiver(
            pos_opt[:, 0], pos_opt[:, 1], np.cos(heading_opt), np.sin(heading_opt)
        )
        ax.legend()

        # plt.show()
        fig.savefig("./tests/lqr/test_lqr1.png", bbox_inches="tight", pad_inches=0)


def test_lqr2(visualize=False) -> None:
    p = create_params()
    np.random.seed(seed=p.seed)
    n, k = 2, 50
    dt = p.dt

    db = DubinsV1(dt, params=p.system_dynamics_params)
    x_dim, u_dim = db._x_dim, db._u_dim

    x_n13 = np.array(np.zeros((n, 1, x_dim)), dtype=np.float32)
    v_1k, w_1k = np.ones((k - 1, 1)) * 0.1, np.linspace(0.5, 0.3, k - 1)[:, None]

    u_1k2 = np.array(np.concatenate([v_1k, w_1k], axis=1)[None], dtype=np.float32)
    u_nk2 = np.zeros((n, k - 1, 2), dtype=np.float32) + u_1k2
    trajectory_ref = db.simulate_T(x_n13, u_nk2, T=k)

    x_nk3, u_nk2 = db.parse_trajectory(trajectory_ref)

    # stack two different reference trajectories together
    # to verify that batched LQR works across the batch dim
    goal_x, goal_y = 4.0, 0.0
    goal = np.array([goal_x, goal_y, 0.0], dtype=np.float32)
    x_ref_nk3 = np.array(np.tile(goal, (1, k, 1)))
    u_ref_nk2 = np.array(np.zeros((1, k, u_dim), dtype=np.float32))
    x_nk3 = np.concatenate([x_ref_nk3, x_nk3[0:1]], axis=0)
    u_nk2 = np.concatenate([u_ref_nk2, u_nk2[0:1]], axis=0)
    trajectory_ref = db.assemble_trajectory(x_nk3, u_nk2)

    cost_fn = QuadraticRegulatorRef(trajectory_ref, db, p)

    x_nk3 = np.array(np.zeros((n, k, x_dim), dtype=np.float32))
    u_nk2 = np.array(np.zeros((n, k, u_dim), dtype=np.float32))
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)

    lqr_solver = LQRSolver(T=k - 1, dynamics=db, cost=cost_fn)

    start_config = db.init_egocentric_robot_config(dt=dt, n=n)
    lqr_res = lqr_solver.lqr(start_config, trajectory, verbose=False)
    trajectory_opt = lqr_res["trajectory_opt"]
    assert np.abs(lqr_res["J_hist"][1][0] - 8.0) < 1e-4
    assert np.abs(lqr_res["J_hist"][0][0] - 400.0) < 1e-4

    if visualize:
        pos_ref = trajectory_ref.position_nk2()[0]
        pos_opt = trajectory_opt.position_nk2()[0]
        heading_ref = trajectory_ref.heading_nk1()[0]
        heading_opt = trajectory_opt.heading_nk1()[0]
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], "r-", label="ref")
        ax.quiver(
            pos_ref[:, 0], pos_ref[:, 1], np.cos(heading_ref), np.sin(heading_ref)
        )
        ax.plot(pos_opt[:, 0], pos_opt[:, 1], "b-", label="opt")
        ax.quiver(
            pos_opt[:, 0], pos_opt[:, 1], np.cos(heading_opt), np.sin(heading_opt)
        )
        ax.set_title("Goal [4.0, 0.0]")
        ax.legend()

        pos_ref = trajectory_ref.position_nk2()[1]
        pos_opt = trajectory_opt.position_nk2()[1]
        heading_ref = trajectory_ref.heading_nk1()[1]
        heading_opt = trajectory_opt.heading_nk1()[1]
        ax = fig.add_subplot(122)
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], "r-", label="ref")
        ax.quiver(
            pos_ref[:, 0], pos_ref[:, 1], np.cos(heading_ref), np.sin(heading_ref)
        )
        ax.plot(pos_opt[:, 0], pos_opt[:, 1], "b-", label="opt")
        ax.quiver(
            pos_opt[:, 0], pos_opt[:, 1], np.cos(heading_opt), np.sin(heading_opt)
        )
        ax.set_title("Nonlinear Traj")
        ax.legend()

        # plt.show()
        fig.savefig("./tests/lqr/test_lqr2.png", bbox_inches="tight", pad_inches=0)


def main_test() -> None:
    # robot should move to goal in 1 step and stay there
    test_lqr0(visualize=False)
    test_lqr1(visualize=False)  # robot should track a trajectory
    test_lqr2(visualize=False)  # LQR should track 2 trajectories in a batch
    print("%sLqr tests passed!%s" % (color_text["green"], color_text["reset"]))


if __name__ == "__main__":
    main_test()
