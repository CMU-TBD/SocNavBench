from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
from trajectory.spline.spline import Spline
from trajectory.trajectory import SystemConfig


class Spline3rdOrder(Spline):
    def __init__(self, dt: float, n: int, k: int, params: DotMap):
        super(Spline3rdOrder, self).__init__(dt=dt, n=n, k=k)
        self.params: DotMap = params

    """ A class representing a 3rd order spline for a mobile ground robot
    (in a 2d cartesian plane). The 3rd order spline allows for constraints
    on the start config, [x0, y0, theta0, v0], and goal config,
    [xg, yg, thetag,vg]. Angular speeds w0 and wg are not constrainable.
    """

    def fit(
        self,
        start_config: SystemConfig,
        goal_config: SystemConfig,
        final_times_n1: Optional[np.ndarray] = None,
        factors: Optional[np.ndarray] = None,
    ) -> None:
        """Fit a 3rd order spline between start config and goal config.
        Factors_n2 represent 2 degrees of freedom in fitting the spline.
        If factors_n2=None it is set heuristically below.
        If final_time_n1=None, a final time of 1 is used.
        The spline is of the form:
            p(t) = a3(t/tf)^3+b3(t/tf)^2+c3(t/tf)+d3
            x(p) = a1p^3+b1p^2+c1p+d1
            y(p) = a2p^2+b2p^2+c2p+d2
        """

        # Compute the factors
        if factors is None:  # Compute them heuristically
            factor1_n1 = start_config.speed_nk1()[:, :, 0] + np.linalg.norm(
                goal_config.position_nk2() - start_config.position_nk2(), axis=2
            )
            factor2_n1 = factor1_n1
            factors_n2 = np.concatenate([factor1_n1, factor2_n1], axis=1)
        else:
            factors_n2 = factors

        # Compute the final times
        if final_times_n1 is None:
            final_times_n1 = np.ones((self.n, 1))

        # Fit spline
        # with tf.name_scope('fit_spline'):
        f1_n1, f2_n1 = factors_n2[:, 0:1], factors_n2[:, 1:]

        start_pos_n12 = start_config.position_nk2()
        goal_pos_n12 = goal_config.position_nk2()

        # Multiple solutions if start and goal are the same x,y coordinates
        assert np.all(
            np.linalg.norm(goal_pos_n12 - start_pos_n12, axis=2) >= self.params.epsilon
        )

        x0_n1, y0_n1 = start_pos_n12[:, :, 0], start_pos_n12[:, :, 1]
        t0_n1 = start_config.heading_nk1()[:, :, 0]
        v0_n1 = start_config.speed_nk1()[:, :, 0]

        xg_n1, yg_n1 = goal_pos_n12[:, :, 0], goal_pos_n12[:, :, 1]
        tg_n1 = goal_config.heading_nk1()[:, :, 0]
        vg_n1 = goal_config.speed_nk1()[:, :, 0]

        d1_n1 = x0_n1
        c1_n1 = f1_n1 * np.cos(t0_n1)
        a1_n1 = f2_n1 * np.cos(tg_n1) - 2 * xg_n1 + c1_n1 + 2 * d1_n1
        b1_n1 = 3 * xg_n1 - f2_n1 * np.cos(tg_n1) - 2 * c1_n1 - 3 * d1_n1

        d2_n1 = y0_n1
        c2_n1 = f1_n1 * np.sin(t0_n1)
        a2_n1 = f2_n1 * np.sin(tg_n1) - 2 * yg_n1 + c2_n1 + 2 * d2_n1
        b2_n1 = 3 * yg_n1 - f2_n1 * np.sin(tg_n1) - 2 * c2_n1 - 3 * d2_n1

        c3_n1 = (final_times_n1 * v0_n1) / f1_n1
        a3_n1 = (final_times_n1 * vg_n1 / f2_n1) + c3_n1 - 2.0
        b3_n1 = 1.0 - c3_n1 - a3_n1

        self.x_coeffs_n14 = np.stack([a1_n1, b1_n1, c1_n1, d1_n1], axis=2)
        self.y_coeffs_n14 = np.stack([a2_n1, b2_n1, c2_n1, d2_n1], axis=2)
        self.p_coeffs_n14 = np.stack([a3_n1, b3_n1, c3_n1, 0.0 * c3_n1], axis=2)
        self.final_times_n1 = final_times_n1

        # Update the batch size as the same spline object
        # can be used with multiple start/ goal configurations
        self.n = start_config.n

    def _eval_spline(
        self, ts_nk: np.ndarray, calculate_speeds: Optional[bool] = True
    ) -> None:
        """ Evaluates the spline on points in ts_nk
        Assumes ts is normalized to be in [0, 1.]
        """
        x_coeffs_n14 = self.x_coeffs_n14
        y_coeffs_n14 = self.y_coeffs_n14
        p_coeffs_n14 = self.p_coeffs_n14

        # with tf.name_scope('eval_spline'):
        ts_n4k = np.stack(
            [np.power(ts_nk, 3), np.power(ts_nk, 2), ts_nk, np.ones_like(ts_nk)], axis=1
        )
        ps_nk = np.squeeze(np.matmul(p_coeffs_n14, ts_n4k), axis=1)

        ps_n4k = np.stack(
            [np.power(ps_nk, 3), np.power(ps_nk, 2), ps_nk, np.ones_like(ps_nk)], axis=1
        )
        ps_dot_n4k = np.stack(
            [
                3.0 * np.power(ps_nk, 2),
                2.0 * ps_nk,
                np.ones_like(ps_nk),
                np.zeros_like(ps_nk),
            ],
            axis=1,
        )

        xs_nk = np.squeeze(np.matmul(x_coeffs_n14, ps_n4k), axis=1)
        ys_nk = np.squeeze(np.matmul(y_coeffs_n14, ps_n4k), axis=1)

        xs_dot_nk = np.squeeze(np.matmul(x_coeffs_n14, ps_dot_n4k), axis=1)
        ys_dot_nk = np.squeeze(np.matmul(y_coeffs_n14, ps_dot_n4k), axis=1)

        self._position_nk2 = np.stack([xs_nk, ys_nk], axis=2)
        self._heading_nk1 = np.arctan2(ys_dot_nk, xs_dot_nk)[:, :, None]

        if calculate_speeds:
            ts_dot_n4k = np.stack(
                [
                    3.0 * np.power(ts_nk, 2),
                    2.0 * ts_nk,
                    np.ones_like(ts_nk),
                    np.zeros_like(ts_nk),
                ],
                axis=1,
            )
            ps_ddot_n4k = np.stack(
                [
                    6.0 * ps_nk,
                    2.0 * np.ones_like(ps_nk),
                    np.zeros_like(ps_nk),
                    np.zeros_like(ps_nk),
                ],
                axis=1,
            )

            ps_dot_nk = np.squeeze(np.matmul(p_coeffs_n14, ts_dot_n4k), axis=1)

            xs_ddot_nk = np.squeeze(np.matmul(x_coeffs_n14, ps_ddot_n4k), axis=1)
            ys_ddot_nk = np.squeeze(np.matmul(y_coeffs_n14, ps_ddot_n4k), axis=1)

            speed_ps_nk = np.sqrt(xs_dot_nk ** 2 + ys_dot_nk ** 2)
            speed_nk = speed_ps_nk * ps_dot_nk

            numerator_nk = xs_dot_nk * ys_ddot_nk - ys_dot_nk * xs_ddot_nk
            angular_speed_nk = numerator_nk / (speed_ps_nk ** 2) * ps_dot_nk

            self._speed_nk1 = speed_nk[:, :, None]
            self._angular_speed_nk1 = angular_speed_nk[:, :, None]

            self._acceleration_nk1 = np.zeros_like(self._speed_nk1)
            self._angular_acceleration_nk1 = np.zeros_like(self._speed_nk1)

    def check_dynamic_feasibility(
        self, speed_max_system: float, angular_speed_max_system: float, horizon_s: float
    ) -> np.int32:
        """Checks whether the current computed spline can be executed in time <= horizon_s (specified in seconds)
        while respecting max speed and angular speed constraints. Returns the batch indices of all valid splines."""

        # Compute the minimum horizon required to execute the spline while ensuring dynamic feasibility
        required_horizon_n1 = self.compute_dynamically_feasible_horizon(
            speed_max_system, angular_speed_max_system
        )

        # Compute the valid splines
        valid_idxs_n = np.where(required_horizon_n1 <= horizon_s)[:, 0]
        return valid_idxs_n.astype(np.int32)

    def compute_dynamically_feasible_horizon(
        self, speed_max_system: float, angular_speed_max_system: float
    ) -> float:
        """
        Compute the horizon (in seconds) such that the computed spline respect the speed and angular
        speed at all times.
        Speed assumed to be in [0, speed_max_system]
        Angular speed assumed to be in [-angular_speed_max_system, angular_speed_max_system]
        """
        # Compute the horizon required to make sure that we satisfy the speed constraints at all times
        max_speed_n1 = np.amax(self.speed_nk1(), axis=1)
        required_horizon_speed_n1 = (
            self.final_times_n1 * max_speed_n1 / speed_max_system
        )

        # Compute the horizon required to make sure that we satisfy the angular speed constraints at all times
        max_angular_speed_n1 = np.amax(np.abs(self.angular_speed_nk1()), axis=1)
        required_horizon_angular_speed_n1 = (
            self.final_times_n1 * max_angular_speed_n1 / angular_speed_max_system
        )

        # Compute the horizon required to make sure that we satisfy all control constraints at all times
        return np.maximum(required_horizon_speed_n1, required_horizon_angular_speed_n1)

    def rescale_spline_horizon_to_dynamically_feasible_horizon(
        self,
        speed_max_system: float,
        angular_speed_max_system: float,
        minimum_horizon: Optional[float] = 0.0,
    ) -> None:
        """
        Rescale the spline horizon to a new horizon without recomputing the spline coefficients.
        """
        # Compute the minimum horizon required to execute the spline while ensuring dynamic feasibility

        required_horizon_n1 = self.compute_dynamically_feasible_horizon(
            speed_max_system, angular_speed_max_system
        )

        # Enforce a minimum horizon
        required_horizon_n1 = np.maximum(required_horizon_n1, minimum_horizon)

        # Reset the final times
        self.final_times_n1 = required_horizon_n1

        # Valid horizon for each trajectory in the batch
        # in discrete time steps
        self.valid_horizons_n1 = np.ceil(self.final_times_n1 / self.dt)

        # Reevaluate the spline to be consistent with the new horizon
        self.eval_spline(self.ts_nk)

    def find_trajectories_within_a_horizon(self, horizon_s: float) -> np.int32:
        """
        Find the indices of splines whose final time is within the horizon [0, horizon_s].
        """
        valid_idxs_n = np.where(self.final_times_n1 <= horizon_s)[0]
        return valid_idxs_n.astype(np.int32)

    @staticmethod
    def ensure_goals_valid(
        start_x: float,
        start_y: float,
        goal_x_nk1: np.ndarray,
        goal_y_nk1: np.ndarray,
        goal_theta_nk1: np.ndarray,
        epsilon: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Perturbs goal_x and goal_y by epsilon if needed ensuring that a unique spline exists.
        Assumes that all goal angles are within [-pi/2., pi/2]."""
        assert (goal_theta_nk1 >= -np.pi / 2.0).all() and (
            goal_theta_nk1 <= np.pi / 2.0
        ).all()
        norms = np.linalg.norm(
            np.concatenate([goal_x_nk1 - start_x, goal_y_nk1 - start_y], axis=2), axis=2
        )
        invalid_idxs = norms == 0.0
        goal_x_nk1[invalid_idxs] += epsilon
        goal_y_nk1[invalid_idxs] += (
            np.sign(np.sin(goal_theta_nk1[invalid_idxs])) * epsilon
        )
        return goal_x_nk1, goal_y_nk1, goal_theta_nk1

    def render(
        self,
        axs: List[plt.axes],
        batch_idx: Optional[int] = 0,
        freq: Optional[int] = 4,
        plot_heading: Optional[bool] = False,
        plot_velocity: Optional[bool] = False,
        label_start_and_end: Optional[bool] = True,
    ):
        super().render(
            axs,
            batch_idx,
            freq,
            plot_heading=plot_heading,
            plot_velocity=plot_velocity,
            label_start_and_end=label_start_and_end,
            name="Spline",
        )

