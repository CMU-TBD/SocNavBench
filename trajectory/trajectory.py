import numpy as np
import matplotlib.pyplot as plt
import copy


class Trajectory(object):
    """
    The base class for the trajectory of a ground vehicle.
    n is the batch size and k is the # of time steps in the trajectory.
    """

    def __init__(self, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
                 angular_speed_nk1=None, angular_acceleration_nk1=None,
                 dtype=np.float32, variable=True, direct_init=False,
                 valid_horizons_n1=None,
                 track_trajectory_acceleration=True,
                 check_dimens=True):

        # Check dimensions now to make your life easier later
        if position_nk2 is not None and check_dimens:
            assert(n == position_nk2.shape[0])
            try:
                assert(k == position_nk2.shape[1])
            except:
                try:
                    # tuple with implied 1 as dimen 1
                    assert(position_nk2.shape[0] == position_nk2.size)
                except:
                    print("ERROR, dimens mismatch:", k, position_nk2.shape[1])
                    exit(1)

        # Discretization step
        self.dt = dt

        # Number of timesteps
        self.k = k
        if valid_horizons_n1 is None:
            self.valid_horizons_n1 = np.ones((n, 1), dtype=np.float32) * k
        else:
            self.valid_horizons_n1 = np.array(valid_horizons_n1)

        # Batch Size
        self.n = n

        # If not tracking trajectory acceleration
        # then set them to be arrays of size
        # (1, 1, 0) to save memory
        if not track_trajectory_acceleration:
            angular_acceleration_nk1 = np.array([[[]]], dtype=np.float32)
            acceleration_nk1 = np.array([[[]]], dtype=np.float32)

        self.vars = []
        # When these are already all tensorflow object use direct-init
        if direct_init:
            self._position_nk2 = position_nk2
            self._speed_nk1 = speed_nk1
            self._acceleration_nk1 = acceleration_nk1
            self._heading_nk1 = heading_nk1
            self._angular_speed_nk1 = angular_speed_nk1
            self._angular_acceleration_nk1 = angular_acceleration_nk1
        else:
            # Translational trajectories
            self._position_nk2 = np.zeros([n, k, 2], dtype=dtype) if position_nk2 is None \
                else np.array(position_nk2, dtype=dtype)
            self._speed_nk1 = np.zeros([n, k, 1], dtype=dtype) if speed_nk1 is None \
                else np.array(speed_nk1, dtype=dtype)
            self._acceleration_nk1 = np.zeros([n, k, 1], dtype=dtype) if acceleration_nk1 is None \
                else np.array(acceleration_nk1, dtype=dtype)

            # Rotational trajectories
            self._heading_nk1 = np.zeros([n, k, 1], dtype=dtype) if heading_nk1 is None \
                else np.array(heading_nk1, dtype=dtype)
            self._angular_speed_nk1 = np.zeros([n, k, 1], dtype=dtype) if angular_speed_nk1 is None \
                else np.array(angular_speed_nk1, dtype=dtype)
            self._angular_acceleration_nk1 = np.zeros([n, k, 1], dtype=dtype) if angular_acceleration_nk1 is None \
                else np.array(angular_acceleration_nk1, dtype=dtype)

    def memory_usage_bytes(self):
        """
        A function which gives the memory usage of this trajectory object
        in bytes.
        """
        var_names = [self._position_nk2, self.valid_horizons_n1, self._speed_nk1,
                     self._acceleration_nk1, self._heading_nk1, self._angular_speed_nk1,
                     self._angular_acceleration_nk1]
        return np.sum([var_name.numpy().nbytes for var_name in var_names])

    @classmethod
    def init_from_numpy_repr(cls, dt, n, k, position_nk2, speed_nk1,
                             acceleration_nk1, heading_nk1, angular_speed_nk1,
                             angular_acceleration_nk1, valid_horizons_n1,
                             track_trajectory_acceleration=True):
        """Utility function to initialize a trajectory object from its numpy
        representation. Useful for loading pickled trajectories"""
        return cls(dt=dt, n=n, k=k, position_nk2=position_nk2,
                   speed_nk1=speed_nk1, acceleration_nk1=acceleration_nk1,
                   heading_nk1=heading_nk1,
                   angular_speed_nk1=angular_speed_nk1,
                   angular_acceleration_nk1=angular_acceleration_nk1,
                   valid_horizons_n1=valid_horizons_n1,
                   variable=False,
                   track_trajectory_acceleration=track_trajectory_acceleration)

    def update_valid_mask_nk(self):
        """Update this trajectories valid mask. The valid mask is a mask of 1's
        and 0's at the trajectories sampling interval where 1's represent
        trajectory data within the valid horizon and 0's otherwise."""
        all_valid_nk = np.broadcast_to(
            np.arange(self.k, dtype=np.float32) + 1, (self.n, self.k))
        self.valid_mask_nk = (
            all_valid_nk <= self.valid_horizons_n1).astype(np.float32)

    def assign_from_trajectory_batch_idx(self, trajectory, batch_idx):
        """Assigns a trajectory object's instance variables from the trajectory stored
        at batch index batch_idx in trajectory."""
        self.assign_trajectory_from_tensors(
            position_nk2=trajectory.position_nk2()[batch_idx:batch_idx + 1],
            speed_nk1=trajectory.speed_nk1()[batch_idx:batch_idx + 1],
            acceleration_nk1=trajectory.acceleration_nk1()[
                batch_idx:batch_idx + 1],
            heading_nk1=trajectory.heading_nk1()[batch_idx:batch_idx + 1],
            angular_speed_nk1=trajectory.angular_speed_nk1()[
                batch_idx:batch_idx + 1],
            angular_acceleration_nk1=trajectory.angular_acceleration_nk1()[
                batch_idx:batch_idx + 1],
            valid_horizons_n1=trajectory.valid_horizons_n1[batch_idx:batch_idx + 1]
        )

    def assign_trajectory_from_tensors(self, position_nk2, speed_nk1, acceleration_nk1,
                                       heading_nk1, angular_speed_nk1, angular_acceleration_nk1,
                                       valid_horizons_n1):
        self._position_nk2 = position_nk2
        self._speed_nk1 = speed_nk1
        self._acceleration_nk1 = acceleration_nk1
        self._heading_nk1 = heading_nk1
        self._angular_speed_nk1 = angular_speed_nk1
        self._angular_acceleration_nk1 = angular_acceleration_nk1
        self.valid_horizons_n1 = valid_horizons_n1

    def gather_across_batch_dim(self, idxs):
        """Given a tensor of indexes to gather in the batch dimension,
        update this trajectories instance variables and shape."""
        self.n = idxs.size

        def gather(arr, idxs):  # used for when arr is multidim and dont want slicing
            return arr[idxs]
        self._position_nk2 = gather(self._position_nk2, idxs)
        self._speed_nk1 = gather(self._speed_nk1, idxs)
        self._acceleration_nk1 = gather(self._acceleration_nk1, idxs)
        self._heading_nk1 = gather(self._heading_nk1, idxs)
        self._angular_speed_nk1 = gather(self._angular_speed_nk1, idxs)
        self._angular_acceleration_nk1 = gather(
            self._angular_acceleration_nk1, idxs)
        self.valid_horizons_n1 = gather(self.valid_horizons_n1, idxs)
        return self

    def to_numpy_repr(self):
        """Utility function to return a representation of the trajectory using
        numpy arrays. Useful for pickling trajectories."""
        numpy_dict = {'dt': self.dt, 'n': self.n, 'k': self.k,
                      'position_nk2': self.position_nk2(),
                      'speed_nk1': self.speed_nk1(),
                      'acceleration_nk1': self.acceleration_nk1(),
                      'heading_nk1': self.heading_nk1(),
                      'angular_speed_nk1': self.angular_speed_nk1(),
                      'angular_acceleration_nk1': self.angular_acceleration_nk1(),
                      'valid_horizons_n1': self.valid_horizons_n1}
        return numpy_dict

    @classmethod
    def concat_across_batch_dim(cls, trajs):
        """Concatenates a list of trajectory objects
        across the batch dimension, returning a new
        trajectory object."""
        if len(trajs) == 0:
            return None

        position_nk2 = np.concatenate([traj.position_nk2()
                                       for traj in trajs], axis=0)
        speed_nk1 = np.concatenate([traj.speed_nk1()
                                    for traj in trajs], axis=0)
        acceleration_nk1 = np.concatenate(
            [traj.acceleration_nk1() for traj in trajs], axis=0)
        heading_nk1 = np.concatenate(
            [traj.heading_nk1() for traj in trajs], axis=0)
        angular_speed_nk1 = np.concatenate(
            [traj.angular_speed_nk1() for traj in trajs], axis=0)
        angular_acceleration_nk1 = np.concatenate(
            [traj.angular_acceleration_nk1() for traj in trajs], axis=0)
        valid_horizons_n1 = np.concatenate(
            [traj.valid_horizons_n1 for traj in trajs], axis=0)

        dt = trajs[0].dt
        k = trajs[0].k
        n = position_nk2.shape[0]
        return cls(dt=dt, n=n, k=k, position_nk2=position_nk2,
                   speed_nk1=speed_nk1, acceleration_nk1=acceleration_nk1,
                   heading_nk1=heading_nk1, angular_speed_nk1=angular_speed_nk1,
                   angular_acceleration_nk1=angular_acceleration_nk1,
                   valid_horizons_n1=valid_horizons_n1)

    @classmethod
    def gather_across_batch_dim_and_create(cls, traj, idxs):
        """Given a tensor of indexes to gather in the batch dimension,
        gather traj's instance variables across the batch dimension
        creating a new trajectory object."""
        dt = traj.dt
        n = idxs.size
        k = traj.k

        def gather(arr, idxs):  # used for when arr is multidim and dont want slicing
            return arr[idxs]

        position_nk2 = gather(traj.position_nk2(), idxs)
        speed_nk1 = gather(traj.speed_nk1(), idxs)
        acceleration_nk1 = np.zeros_like(traj.acceleration_nk1()) if traj.acceleration_nk1().size == 0 else gather(
            traj.acceleration_nk1(), idxs)
        heading_nk1 = gather(traj.heading_nk1(), idxs)
        angular_speed_nk1 = gather(traj.angular_speed_nk1(), idxs)
        angular_acceleration_nk1 = np.zeros_like(traj.angular_acceleration_nk1()) if traj.angular_acceleration_nk1(
        ).size == 0 else gather(traj.angular_acceleration_nk1(), idxs)
        valid_horizons_n1 = gather(traj.valid_horizons_n1, idxs)
        return cls(dt=dt, n=n, k=k, position_nk2=position_nk2,
                   speed_nk1=speed_nk1, acceleration_nk1=acceleration_nk1,
                   heading_nk1=heading_nk1, angular_speed_nk1=angular_speed_nk1,
                   angular_acceleration_nk1=angular_acceleration_nk1,
                   valid_horizons_n1=valid_horizons_n1)

    @ property
    def trainable_variables(self):
        return self.vars

    @ property
    def shape(self):
        return '({:d}, {:d})'.format(self.n, self.k)

    def position_nk2(self):
        return self._position_nk2

    def speed_nk1(self):
        return self._speed_nk1

    def acceleration_nk1(self):
        return self._acceleration_nk1

    def heading_nk1(self):
        return self._heading_nk1

    def angular_speed_nk1(self):
        return self._angular_speed_nk1

    def to_3D_numpy(self):
        pos_2 = self.position_nk2()[0][0]
        heading = self.heading_nk1()[0][0]
        return np.append(pos_2, heading)

    def angular_acceleration_nk1(self):
        return self._angular_acceleration_nk1

    def position_and_heading_nk3(self):
        p_nk2 = self.position_nk2()
        h_nk1 = self.heading_nk1()
        if(len(p_nk2) == 0 and len(h_nk1) == 0):
            return np.array([])
        return np.concatenate([p_nk2, h_nk1], axis=2)

    def speed_and_angular_speed_nk2(self):
        s_nk1 = self.speed_nk1()
        a_nk1 = self.angular_speed_nk1()
        if(len(s_nk1) == 0 and len(a_nk1) == 0):
            return np.array([])
        return np.concatenate([s_nk1, a_nk1], axis=2)

    def position_heading_speed_and_angular_speed_nk5(self):
        ph_nk3 = self.position_and_heading_nk3()
        sa_nk2 = self.speed_and_angular_speed_nk2()
        if(len(ph_nk3) == 0 and len(sa_nk2) == 0):
            return np.array([])
        return np.concatenate([ph_nk3, sa_nk2], axis=2)

    def append_along_time_axis(self, trajectory, track_trajectory_acceleration=True):
        """ Utility function to concatenate trajectory
        over time. Useful for assembling an entire
        trajectory from multiple sub-trajectories. """
        self._position_nk2 = np.concatenate([self.position_nk2(),
                                             trajectory.position_nk2()],
                                            axis=1)
        self._speed_nk1 = np.concatenate([self.speed_nk1(), trajectory.speed_nk1()],
                                         axis=1)
        if(track_trajectory_acceleration):
            self._acceleration_nk1 = np.concatenate([self.acceleration_nk1(),
                                                     trajectory.acceleration_nk1()],
                                                    axis=1)
            self._angular_acceleration_nk1 = np.concatenate([self.angular_acceleration_nk1(),
                                                             trajectory.angular_acceleration_nk1()],
                                                            axis=1)
        self._heading_nk1 = np.concatenate([self.heading_nk1(),
                                            trajectory.heading_nk1()], axis=1)
        self._angular_speed_nk1 = np.concatenate([self.angular_speed_nk1(),
                                                  trajectory.angular_speed_nk1()],
                                                 axis=1)
        self.k = self.k + trajectory.k
        self.valid_horizons_n1 = self.valid_horizons_n1 + trajectory.valid_horizons_n1

    def clip_along_time_axis(self, horizon):
        """ Utility function for clipping a trajectory along
        the time axis. Useful for clipping a trajectory within
        a specified horizon."""
        if self.k <= horizon:
            return

        self._position_nk2 = self._position_nk2[:, :horizon]
        self._speed_nk1 = self._speed_nk1[:, :horizon]
        self._acceleration_nk1 = self._acceleration_nk1[:, :horizon]
        self._heading_nk1 = self._heading_nk1[:, :horizon]
        self._angular_speed_nk1 = self._angular_speed_nk1[:, :horizon]
        self._angular_acceleration_nk1 = self._angular_acceleration_nk1[:, :horizon]
        self.k = horizon
        self.valid_horizons_n1 = np.clip(self.valid_horizons_n1, 0, horizon)

    def take_along_time_axis(self, horizon):
        """ Utility function for taking all elements in a trajectory past
        the time axis."""
        if self.k <= horizon:
            return

        self._position_nk2 = self._position_nk2[:, horizon:]
        self._speed_nk1 = self._speed_nk1[:, horizon:]
        self._acceleration_nk1 = self._acceleration_nk1[:, horizon:]
        self._heading_nk1 = self._heading_nk1[:, horizon:]
        self._angular_speed_nk1 = self._angular_speed_nk1[:, horizon:]
        self._angular_acceleration_nk1 = self._angular_acceleration_nk1[:, horizon:]
        self.k = self.k - horizon
        self.valid_horizons_n1 = np.clip(self.valid_horizons_n1, horizon, -1)

    @classmethod
    def concat_along_time_axis(cls, trajectories):
        """ Concatenates a list of trajectory objects
        along the time axis. Useful for assembling an entire
        trajectory from multiple sub-trajectories. """

        # Check all subtrajectories have the same batch size and dt
        assert([x.n for x in trajectories] == [1] * len(trajectories))
        assert([x.dt for x in trajectories] == [
               trajectories[0].dt] * len(trajectories))

        n = trajectories[0].n
        dt = trajectories[0].dt
        k = sum([x.k for x in trajectories])

        position_nk2 = np.concatenate([x.position_nk2()
                                       for x in trajectories], axis=1)
        speed_nk1 = np.concatenate([x.speed_nk1()
                                    for x in trajectories], axis=1)
        acceleration_nk1 = np.concatenate(
            [x.acceleration_nk1() for x in trajectories], axis=1)
        heading_nk1 = np.concatenate([x.heading_nk1()
                                      for x in trajectories], axis=1)
        angular_speed_nk1 = np.concatenate(
            [x.angular_speed_nk1() for x in trajectories], axis=1)
        angular_acceleration_nk1 = np.concatenate(
            [x.angular_acceleration_nk1() for x in trajectories], axis=1)
        valid_horizons_n1 = np.reduce_sum(
            [x.valid_horizons_n1 for x in trajectories], axis=0)
        return cls(dt=dt, n=n, k=k,
                   position_nk2=position_nk2,
                   speed_nk1=speed_nk1,
                   acceleration_nk1=acceleration_nk1,
                   heading_nk1=heading_nk1,
                   angular_speed_nk1=angular_speed_nk1,
                   angular_acceleration_nk1=angular_acceleration_nk1,
                   valid_horizons_n1=valid_horizons_n1,
                   direct_init=True)

    @classmethod
    def copy(cls, traj, check_dimens=True):
        return cls(dt=traj.dt, n=traj.n, k=traj.k,
                   position_nk2=traj.position_nk2() * 1.,
                   speed_nk1=traj.speed_nk1() * 1.,
                   acceleration_nk1=traj.acceleration_nk1() * 1.,
                   heading_nk1=traj.heading_nk1() * 1.,
                   angular_speed_nk1=traj.angular_speed_nk1() * 1.,
                   angular_acceleration_nk1=traj.angular_acceleration_nk1() * 1.,
                   valid_horizons_n1=traj.valid_horizons_n1 * 1.,
                   variable=False, direct_init=True, check_dimens=check_dimens)

    @classmethod
    def new_traj_clip_along_time_axis(cls, trajectory, horizon,
                                      repeat_second_to_last_speed=False):
        """
        Utility function for clipping a trajectory along
        the time axis. Useful for clipping a trajectory within
        a specified horizon. Creates a new object as dimensions
        are being changed and assign will not work.
        """
        if trajectory.k <= horizon:
            return trajectory

        speed_nk1 = trajectory.speed_nk1()[:, :horizon]
        angular_speed_nk1 = trajectory.angular_speed_nk1()[:, :horizon]

        if repeat_second_to_last_speed:
            speed_nk1 = np.concatenate(
                [speed_nk1[:, :-1], speed_nk1[:, -2:-1]], axis=1)
            angular_speed_nk1 = np.concatenate([angular_speed_nk1[:, :-1],
                                                angular_speed_nk1[:, -2:-1]], axis=1)

        return cls(dt=trajectory.dt, n=trajectory.n, k=horizon,
                   position_nk2=trajectory.position_nk2()[:, :horizon],
                   speed_nk1=speed_nk1,
                   acceleration_nk1=trajectory.acceleration_nk1()[:, :horizon],
                   heading_nk1=trajectory.heading_nk1()[:, :horizon],
                   angular_speed_nk1=angular_speed_nk1,
                   angular_acceleration_nk1=trajectory.angular_acceleration_nk1()[:, :horizon])

    def __getitem__(self, index):
        """Allow for indexing along the batch dimension similar
        to a regular tensor. Returns a new object corresponding
        to the batch index, index"""
        if index >= self.n:
            raise IndexError

        pos_nk2 = self.position_nk2()[index: index + 1]
        speed_nk1 = self.speed_nk1()[index: index + 1]
        acceleration_nk1 = self.acceleration_nk1()[index: index + 1]
        heading_nk1 = self.heading_nk1()[index: index + 1]
        angular_speed_nk1 = self.angular_speed_nk1()[index: index + 1]
        angular_acceleration_nk1 = self.angular_acceleration_nk1()[
            index: index + 1]
        valid_horizons_n1 = self.valid_horizons_n1[index: index + 1]
        return self.__class__(dt=self.dt, n=1, k=self.k,
                              position_nk2=pos_nk2,
                              speed_nk1=speed_nk1,
                              acceleration_nk1=acceleration_nk1,
                              heading_nk1=heading_nk1,
                              angular_speed_nk1=angular_speed_nk1,
                              angular_acceleration_nk1=angular_acceleration_nk1,
                              valid_horizons_n1=valid_horizons_n1, direct_init=True)

    def render(self, axs, batch_idx: int = 0, freq=4, plot_quiver=True, plot_heading=False,
               plot_velocity: bool = False, color: str = "red", alpha: float = 0,
               label_start_and_end: bool = False, name: str = '', linewidth: float = 4,
               clip: int = 0):
        ax = axs
        # use clip to only render the *last* "clip" numpoints of the trajectory
        xs = self._position_nk2[batch_idx, -1:-1 * clip:-1, 0]
        ys = self._position_nk2[batch_idx, -1:-1 * clip:-1, 1]
        thetas = self._heading_nk1[batch_idx]

        if plot_quiver:
            ax.quiver(xs[::freq], ys[::freq],
                      np.cos(thetas[::freq]),
                      np.sin(thetas[::freq]))
        title_str = '{:s} Trajectory'.format(name)
        ax.plot(xs, ys, '-', color=color, alpha=alpha, linewidth=linewidth)
        if label_start_and_end:
            start_5 = self.position_heading_speed_and_angular_speed_nk5()[
                batch_idx, 0]
            end_5 = self.position_heading_speed_and_angular_speed_nk5()[
                batch_idx, -1]
            title_str += ('\nStart: [{:.3e}, {:.3e}, {:.3f}, {:.3f}, {:.3f}]\n'.format(*start_5) +
                          'End: [{:.3e}, {:.3e}, {:.3f}, {:.3f}, {:.3f}]'.format(*end_5))
        ax.set_title(title_str)

        if plot_heading:
            print('\033[33m', "Plotting Heading", '\033[0m')
            ax = axs[1]  # [0]
            ax.plot(np.r_[:self.k] * self.dt,
                    self._heading_nk1[batch_idx, :, 0], 'r-')
            ax.set_title('Theta')

        if plot_velocity:
            print('\033[33m', "Plotting Velocity", '\033[0m')
            time = np.r_[:self.k] * self.dt

            ax = axs[2]  # [0]
            ax.plot(time, self._speed_nk1[batch_idx, :, 0], 'r-')
            ax.set_title('Linear Velocity')

            ax = axs[3]  # [0]
            ax.plot(
                time, self._angular_speed_nk1[batch_idx, :, 0], 'r-')
            ax.set_title('Angular Velocity')


class SystemConfig(Trajectory):
    """
    A class representing a system configuration using a trajectory of
    time duration = 1 step.
    """

    def __init__(self, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
                 angular_speed_nk1=None, angular_acceleration_nk1=None,
                 dtype=np.float32, variable=True, direct_init=False,
                 valid_horizons_n1=None,
                 track_trajectory_acceleration=True, check_dimens=True):
        assert(k == 1)
        # Don't pass on valid_horizons_n1 as a SystemConfig has no horizon
        super(SystemConfig, self).__init__(dt, n, k, position_nk2, speed_nk1, acceleration_nk1,
                                           heading_nk1, angular_speed_nk1,
                                           angular_acceleration_nk1, dtype=np.float32,
                                           variable=variable, direct_init=direct_init,
                                           track_trajectory_acceleration=track_trajectory_acceleration,
                                           check_dimens=check_dimens)

    def assign_from_broadcasted_batch(self, config, n):
        """ Assigns a SystemConfig's variables by broadcasting a given config to
        batch size n """
        k = config.k
        self.assign_config_from_tensors(position_nk2=np.broadcast_to(config.position_nk2(), (n, k, 2)),
                                        speed_nk1=np.broadcast_to(
                                            config.speed_nk1(), (n, k, 1)),
                                        acceleration_nk1=np.broadcast_to(
                                            config.acceleration_nk1(), (n, k, 1)),
                                        heading_nk1=np.broadcast_to(
                                            config.heading_nk1(), (n, k, 1)),
                                        angular_speed_nk1=np.broadcast_to(
                                            config.angular_speed_nk1(), (n, k, 1)),
                                        angular_acceleration_nk1=np.broadcast_to(config.angular_acceleration_nk1(), (n, k, 1)))

    @classmethod
    def init_config_from_trajectory_time_index(cls, trajectory, t):
        """ A utility method to initialize a config object
        from a particular timestep of a given trajectory object"""
        position_nk2 = trajectory.position_nk2()
        speed_nk1 = trajectory.speed_nk1()
        acceleration_nk1 = trajectory.acceleration_nk1()
        heading_nk1 = trajectory.heading_nk1()
        angular_speed_nk1 = trajectory.angular_speed_nk1()
        angular_acceleration_nk1 = trajectory.angular_acceleration_nk1()

        if t == -1:
            return cls(dt=trajectory.dt, n=trajectory.n, k=1,
                       position_nk2=position_nk2[:, t:],
                       speed_nk1=speed_nk1[:, t:],
                       acceleration_nk1=acceleration_nk1[:, t:],
                       heading_nk1=heading_nk1[:, t:],
                       angular_speed_nk1=angular_speed_nk1[:, t:],
                       angular_acceleration_nk1=angular_acceleration_nk1[:, t:])

        return cls(dt=trajectory.dt, n=trajectory.n, k=1,
                   position_nk2=position_nk2[:, t:t + 1],
                   speed_nk1=speed_nk1[:, t:t + 1],
                   acceleration_nk1=acceleration_nk1[:, t:t + 1],
                   heading_nk1=heading_nk1[:, t:t + 1],
                   angular_speed_nk1=angular_speed_nk1[:, t:t + 1],
                   angular_acceleration_nk1=angular_acceleration_nk1[:, t:t + 1])

    def assign_from_config_batch_idx(self, config, batch_idx):
        super(SystemConfig, self).assign_from_trajectory_batch_idx(
            config, batch_idx)

    def assign_config_from_tensors(self, position_nk2, speed_nk1, acceleration_nk1,
                                   heading_nk1, angular_speed_nk1, angular_acceleration_nk1):
        super().assign_trajectory_from_tensors(position_nk2, speed_nk1,
                                               acceleration_nk1, heading_nk1,
                                               angular_speed_nk1, angular_acceleration_nk1)

    def render(self, ax, batch_idx=0, plot_quiver=False, **kwargs):
        pos_n13 = self.position_and_heading_nk3()
        pos_3 = pos_n13[batch_idx, 0]
        ax.plot(pos_3[0], pos_3[1], **kwargs)
        if plot_quiver:
            ax.quiver([pos_3[0]], [pos_3[1]],
                      np.cos([pos_3[2]]), np.sin([pos_3[2]]))

    def render_with_boundary(self, ax, batch_idx, boundary_params, **kwargs):
        self.render(ax, batch_idx, **kwargs)
        if boundary_params['norm'] == 2:
            center = self.position_nk2()[batch_idx, 0]
            radius = boundary_params['cutoff']
            c = plt.Circle(center, radius, color=boundary_params['color'])
            ax.add_artist(c)
        else:
            # assert(False)
            center = self.position_nk2()[batch_idx, 0]
            radius = boundary_params['cutoff']
            c = plt.Circle(center, radius, color=boundary_params['color'])
            ax.add_artist(c)
