from trajectory.trajectory import Trajectory
from systems.dubins_car import DubinsCar
import numpy as np


class Dubins5D(DubinsCar):
    """ A discrete time dubins car with state
    [x, y, theta, v, w] and actions [a, alpha]
    (linear and angular acceleration) The dynamics are:

    x(t+1) = x(t) + v(t)*cos(theta_t)*delta_t
    y(t+1) = y(t) + v(t)*sin(theta_t)*delta_t
    theta(t+1) = theta(t) + w(t)*delta_t
    v(t+1) = saturate_linear_velocity(a(t)*dt + v(t))
    w(t+1) = saturate_angular_velocity(alpha(t)*dt + w(t)). """

    def __init__(self, dt):
        super().__init__(dt, x_dim=5, u_dim=2)
        self._angle_dims = 2

    def _simulate_ideal(self, x_nkd, u_nkf, t=None):
        # with tf.name_scope('simulate'):
        theta_nk1 = x_nkd[:, :, 2:3]
        v_nk1 = x_nkd[:, :, 3:4]
        w_nk1 = x_nkd[:, :, 4:5]
        x_new_nkd = np.concatenate([x_nkd[:, :, :3],
                                    self._saturate_linear_velocity(
                                        v_nk1 + self._dt * u_nkf[:, :, 0:1]),
                                    self._saturate_angular_velocity(w_nk1 + self._dt * u_nkf[:, :, 1:2])],
                                   axis=2)
        delta_x_nkd = np.concatenate([v_nk1 * np.cos(theta_nk1),
                                      v_nk1 * np.sin(theta_nk1),
                                      w_nk1,
                                      np.zeros_like(u_nkf)], axis=2)
        return x_new_nkd + self._dt * delta_x_nkd

    def jac_x(self, trajectory):
        x_nk5, u_nk2 = self.parse_trajectory(trajectory)
        # with tf.name_scope('jac_x'):
        # Rightmost Column
        theta_nk1 = x_nk5[:, :, 2:3]
        v_nk1 = x_nk5[:, :, 3:4]
        w_nk1 = x_nk5[:, :, 4:5]

        diag_nk5 = np.concatenate([np.ones_like(x_nk5[:, :, :3]),
                                   self._saturate_linear_velocity_prime(
            u_nk2[:, :, 0:1] * self._dt + v_nk1),
            self._saturate_angular_velocity_prime(u_nk2[:, :, 1:2] * self._dt + w_nk1)], axis=2)

        column2_nk5 = np.concatenate([-v_nk1 * np.sin(theta_nk1),
                                      v_nk1 * np.cos(theta_nk1),
                                      np.zeros_like(x_nk5[:, :, :3])], axis=2)
        column3_nk5 = np.concatenate([np.cos(theta_nk1),
                                      np.sin(theta_nk1),
                                      np.zeros_like(x_nk5[:, :, :3])],
                                     axis=2)
        column4_nk5 = np.concatenate([np.zeros_like(x_nk5[:, :, :2]),
                                      np.ones_like(v_nk1),
                                      np.zeros_like(u_nk2)], axis=2)

        update_nk55 = np.stack([np.zeros_like(x_nk5),
                                np.zeros_like(x_nk5),
                                column2_nk5,
                                column3_nk5,
                                column4_nk5], axis=3)
        # simple multidimensional tensorflow diag implementation for np
        res = []
        for i in range(diag_nk5.shape[0]):
            res_1 = []
            for j in range(diag_nk5.shape[1]):
                res_1.append(np.diag(diag_nk5[i][j]))
            res.append(res_1)
        res = np.array(res)
        return res + self._dt * update_nk55

    def jac_u(self, trajectory):
        x_nk5, u_nk2 = self.parse_trajectory(trajectory)
        # with tf.name_scope('jac_u'):
        v_nk1 = x_nk5[:, :, 3:4]
        w_nk1 = x_nk5[:, :, 4:5]

        column0_nk5 = np.concatenate([np.zeros_like(x_nk5[:, :, :3]),
                                      self._saturate_linear_velocity_prime(
            u_nk2[:, :, 0:1] * self._dt + v_nk1),
            np.zeros_like(v_nk1)], axis=2)

        column1_nk5 = np.concatenate([np.zeros_like(x_nk5[:, :, :4]),
                                      self._saturate_angular_velocity_prime(u_nk2[:, :, 1:2] * self._dt + w_nk1)],
                                     axis=2)
        B_nk52 = np.stack([column0_nk5, column1_nk5], axis=3)
        return B_nk52 * self._dt

    def parse_trajectory(self, trajectory):
        """ A utility function for parsing a trajectory object.
        Returns x_nkd, u_nkf which are states and actions for the
        system """
        u_nk2 = np.concatenate([trajectory.acceleration_nk1(),
                                trajectory.angular_acceleration_nk1()], axis=2)
        return trajectory.position_heading_speed_and_angular_speed_nk5(), u_nk2

    def assemble_trajectory(self, x_nkd, u_nkf, pad_mode=None):
        """ A utility function for assembling a trajectory object
        from x_nkd, u_nkf, a list of states and actions for the system.
        Here d=5=state dimension and u=2=action dimension. """
        n = x_nkd.shape[0]
        k = x_nkd.shape[1]
        u_nkf = self._pad_control_vector(u_nkf, k, pad_mode=pad_mode)
        position_nk2, heading_nk1 = x_nkd[:, :, :2], x_nkd[:, :, 2:3]
        speed_nk1, angular_speed_nk1 = x_nkd[:, :, 3:4], x_nkd[:, :, 4:]
        acceleration_nk1, angular_acceleration_nk1 = u_nkf[:,
                                                           :, 0:1], u_nkf[:, :, 1:2]
        return Trajectory(dt=self._dt, n=n, k=k, position_nk2=position_nk2,
                          heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                          angular_speed_nk1=angular_speed_nk1, acceleration_nk1=acceleration_nk1,
                          angular_acceleration_nk1=angular_acceleration_nk1)
