import matplotlib.pyplot as plt
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from dotmap import DotMap
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import DubinsV1
from trajectory.trajectory import SystemConfig, Trajectory
from utils import utils
from utils.angle_utils import angle_normalize, rotate_pos_nk2
from utils.utils import color_text


def create_params() -> DotMap:
    p = DotMap()
    p.seed = 1
    p.n = 1
    p.k = 100
    p.dt = 0.05

    p.quad_coeffs = [1.0, 1.0, 1.0, 1e-10, 1e-10]
    p.linear_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    p.system_dynamics_params = DotMap(
        system=DubinsV1, dt=0.05, v_bounds=[0.0, 0.6], w_bounds=[-1.1, 1.1]
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


def test_rotate() -> None:
    pos_2 = np.array([1.0, 0], dtype=np.float32)
    theta_1 = np.array([np.pi / 2.0], dtype=np.float32)

    pos_112 = np.array(pos_2[None, None])
    theta_111 = np.array(theta_1[None, None])

    new_pos_112 = rotate_pos_nk2(pos_112, theta_111)
    new_pos_2 = new_pos_112[0, 0]
    assert np.abs(new_pos_2[0]) < 1e-5
    assert np.abs(1.0 - new_pos_2[1]) < 1e-5


def test_coordinate_transform() -> None:
    n, k = 1, 30
    dt = 0.1
    p = create_params()
    dubins_car = DubinsV1(dt=dt, params=p.system_dynamics_params)
    ref_config = dubins_car.init_egocentric_robot_config(dt=dt, n=n)

    pos_nk2 = np.ones((n, k, 2), dtype=np.float32) * np.random.rand()
    traj_global = Trajectory(dt=dt, n=n, k=k, position_nk2=pos_nk2)
    traj_egocentric = Trajectory(dt=dt, n=n, k=k)
    traj_global_new = Trajectory(dt=dt, n=n, k=k)

    dubins_car.to_egocentric_coordinates(ref_config, traj_global, traj_egocentric)

    # Test 0 transform
    assert (pos_nk2 == traj_egocentric.position_nk2()).all()

    ref_config_pos_112 = np.array([[[5.0, 5.0]]], dtype=np.float32)
    ref_config_pos_n12 = np.repeat(ref_config_pos_112, repeats=n, axis=0)
    ref_config = SystemConfig(dt=dt, n=n, k=1, position_nk2=ref_config_pos_n12)
    traj_egocentric = dubins_car.to_egocentric_coordinates(
        ref_config, traj_global, traj_egocentric
    )
    # Test translation
    assert (pos_nk2 - 5.0 == traj_egocentric.position_nk2()).all()

    ref_config_heading_111 = np.array([[[3.0 * np.pi / 4.0]]], dtype=np.float32)
    ref_config_heading_nk1 = np.repeat(ref_config_heading_111, repeats=n, axis=0)
    ref_config = SystemConfig(
        dt=dt,
        n=n,
        k=1,
        position_nk2=ref_config_pos_n12,
        heading_nk1=ref_config_heading_nk1,
    )

    traj_egocentric = dubins_car.to_egocentric_coordinates(
        ref_config, traj_global, traj_egocentric
    )
    traj_global_new = dubins_car.to_world_coordinates(
        ref_config, traj_egocentric, traj_global_new
    )
    assert np.allclose(traj_global.position_nk2(), traj_global_new.position_nk2(), atol=1E-5)


def visualize_coordinate_transform() -> None:
    """Visual sanity check that coordinate transforms
    are working. """
    _, _, axs = utils.subplot2(plt, (2, 2), (8, 8), (0.4, 0.4))
    axs = axs[::-1]

    n, k = 1, 30
    dt = 0.1
    dubins_car = DubinsV1(dt=dt)

    traj_egocentric = Trajectory(dt=dt, n=n, k=k)
    traj_world = Trajectory(dt=dt, n=n, k=k)

    # Form a trajectory in global frame
    # convert to egocentric and back
    start_pos_global_n12 = np.array([[[1.0, 1.0]]], dtype=np.float32)
    start_heading_global_n11 = np.array([[[np.pi / 2.0]]], dtype=np.float32)
    start_config_global = SystemConfig(
        dt=dt,
        n=n,
        k=1,
        position_nk2=start_pos_global_n12,
        heading_nk1=start_heading_global_n11,
    )

    start_n13 = np.concat([start_pos_global_n12, start_heading_global_n11], axis=2)
    u_n12 = np.array([[[0.01, 0.1]]], dtype=np.float32)
    u_nk2 = np.array(np.broadcast_to(u_n12, (n, k, 2)), dtype=np.float32)
    trajectory_world = dubins_car.simulate_T(start_n13, u_nk2, T=k - 1)
    trajectory_world.render([axs[0]], batch_idx=0, freq=4, name="World")

    # Convert to egocentric
    dubins_car.to_egocentric_coordinates(
        start_config_global, trajectory_world, traj_egocentric
    )
    traj_egocentric.render([axs[1]], batch_idx=0, freq=4, name="Egocentric")

    dubins_car.to_world_coordinates(start_config_global, traj_egocentric, traj_world)
    traj_world.render([axs[2]], batch_idx=0, freq=4, name="World #2")
    plt.savefig("./tmp/coordinate_transform.png", bbox_inches="tight")


def test_lqr_feedback_coordinate_transform() -> None:

    p = create_params()
    n, k = p.n, p.k
    dubins = p.system_dynamics_params.system(p.dt, params=p.system_dynamics_params)

    # # Robot starts from (0, 0, 0)
    # # and does a small spiral
    start_pos_n13 = np.array(np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))
    speed_nk1 = np.ones((n, k - 1, 1), dtype=np.float32) * 2.0
    angular_speed_nk1 = np.linspace(1.5, 1.3, k - 1, dtype=np.float32)[None, :, None]
    u_nk2 = np.array(np.concatenate([speed_nk1, angular_speed_nk1], axis=2))
    traj_ref_egocentric = dubins.simulate_T(start_pos_n13, u_nk2, k)

    cost_fn = QuadraticRegulatorRef(traj_ref_egocentric, dubins, p)

    lqr_solver = LQRSolver(T=k - 1, dynamics=dubins, cost=cost_fn)

    start_config0 = SystemConfig(
        dt=p.dt,
        n=p.n,
        k=1,
        position_nk2=start_pos_n13[:, :, :2],
        heading_nk1=start_pos_n13[:, :, 2:3],
    )
    lqr_res = lqr_solver.lqr(start_config0, traj_ref_egocentric, verbose=False)
    traj_lqr_egocentric = lqr_res["trajectory_opt"]
    K_array_egocentric = lqr_res["K_opt_nkfd"]
    k_array = lqr_res["k_opt_nkf1"]

    # The origin of the egocentric frame in world coordinates
    start_pos_n13 = np.array(np.array([[[1.0, 1.0, np.pi / 2.0]]], dtype=np.float32))
    ref_config = SystemConfig(
        dt=p.dt,
        n=p.n,
        k=1,
        position_nk2=start_pos_n13[:, :, :2],
        heading_nk1=start_pos_n13[:, :, 2:3],
    )

    # Convert to world coordinates
    traj_ref_world = dubins.to_world_coordinates(
        ref_config, traj_ref_egocentric, mode="new"
    )
    traj_lqr_world = dubins.to_world_coordinates(
        ref_config, traj_lqr_egocentric, mode="new"
    )
    K_array_world = dubins.convert_K_to_world_coordinates(
        ref_config, K_array_egocentric, mode="new"
    )

    # Apply K_array_world to the system from ref_config
    traj_test_world = lqr_solver.apply_control(
        ref_config, traj_ref_world, k_array, K_array_world
    )

    # Check that the LQR Trajectory computed using K_array_egocentric
    # then transformed to world (traj_lqr_world) matches the
    # LQR Trajectory computed directly using K_array_world
    assert (
        np.linalg.norm(
            traj_lqr_world.position_nk2() - traj_test_world.position_nk2(), axis=2
        )
        < 1e-4
    ).all()
    assert (
        np.linalg.norm(
            angle_normalize(
                traj_lqr_world.heading_nk1() - traj_test_world.heading_nk1()
            ),
            axis=2,
        )
        < 1e-4
    ).all()
    assert (
        np.linalg.norm(traj_lqr_world.speed_nk1() - traj_test_world.speed_nk1(), axis=2)
        < 1e-4
    ).all()
    assert (
        np.linalg.norm(
            traj_lqr_world.angular_speed_nk1() - traj_test_world.angular_speed_nk1(),
            axis=2,
        )
        < 1e-4
    ).all()


def main_test() -> None:
    plt.style.use("ggplot")
    test_rotate()
    test_coordinate_transform()
    test_lqr_feedback_coordinate_transform()
    print(
        "%sCoordinate transform tests passed!%s"
        % (color_text["green"], color_text["reset"])
    )


if __name__ == "__main__":
    main_test()
