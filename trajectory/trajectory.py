from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Trajectory(object):
    """
    The base class for the trajectory of a ground vehicle.
    n is the batch size and k is the # of time steps in the trajectory.
    """

    def __init__(
        self,
        dt: float,
        n: int,
        k: int,
        position_nk2: Optional[np.ndarray] = None,
        speed_nk1: Optional[np.ndarray] = None,
        acceleration_nk1: Optional[np.ndarray] = None,
        heading_nk1: Optional[np.ndarray] = None,
        angular_speed_nk1: Optional[np.ndarray] = None,
        angular_acceleration_nk1: Optional[np.ndarray] = None,
        dtype: Optional[type] = np.float32,
        direct_init: Optional[bool] = False,
        valid_horizons_n1: Optional[np.ndarray] = None,
        track_trajectory_acceleration: Optional[bool] = True,
        check_dimens: Optional[bool] = True,
    ):

        # Check dimensions now to make your life easier later
        if position_nk2 is not None and check_dimens:
            assert n == position_nk2.shape[0]
            try:
                assert k == position_nk2.shape[1]
            except:
                try:
                    # tuple with implied 1 as dimen 1
                    assert position_nk2.shape[0] == position_nk2.size
                except:
                    print("ERROR, dimens mismatch:", k, position_nk2.shape[1])
                    exit(1)

        # Discretization step
        self.dt: float = dt

        # Number of timesteps
        self.k: int = k
        if valid_horizons_n1 is None:
            self.valid_horizons_n1: np.ndarray = np.ones((n, 1), dtype=np.float32) * k
        else:
            self.valid_horizons_n1: np.ndarray = np.array(valid_horizons_n1)

        # Batch Size
        self.n: int = n

        # If not tracking trajectory acceleration
        # then set them to be arrays of size
        # (1, 1, 0) to save memory
        if not track_trajectory_acceleration:
            angular_acceleration_nk1: np.ndarray = np.array([[[]]], dtype=np.float32)
            acceleration_nk1: np.ndarray = np.array([[[]]], dtype=np.float32)

        # When these are already all numpy objects use direct-init
        if direct_init:
            self._position_nk2: np.ndarray = position_nk2
            self._speed_nk1: np.ndarray = speed_nk1
            self._acceleration_nk1: np.ndarray = acceleration_nk1
            self._heading_nk1: np.ndarray = heading_nk1
            self._angular_speed_nk1: np.ndarray = angular_speed_nk1
            self._angular_acceleration_nk1: np.ndarray = angular_acceleration_nk1
        else:
            # Translational trajectories
            self._position_nk2 = (
                np.zeros([n, k, 2], dtype=dtype)
                if position_nk2 is None
                else np.array(position_nk2, dtype=dtype)
            )
            self._speed_nk1 = (
                np.zeros([n, k, 1], dtype=dtype)
                if speed_nk1 is None
                else np.array(speed_nk1, dtype=dtype)
            )
            self._acceleration_nk1 = (
                np.zeros([n, k, 1], dtype=dtype)
                if acceleration_nk1 is None
                else np.array(acceleration_nk1, dtype=dtype)
            )

            # Rotational trajectories
            self._heading_nk1 = (
                np.zeros([n, k, 1], dtype=dtype)
                if heading_nk1 is None
                else np.array(heading_nk1, dtype=dtype)
            )
            self._angular_speed_nk1 = (
                np.zeros([n, k, 1], dtype=dtype)
                if angular_speed_nk1 is None
                else np.array(angular_speed_nk1, dtype=dtype)
            )
            self._angular_acceleration_nk1 = (
                np.zeros([n, k, 1], dtype=dtype)
                if angular_acceleration_nk1 is None
                else np.array(angular_acceleration_nk1, dtype=dtype)
            )

    def memory_usage_bytes(self) -> int:
        """
        A function which gives the memory usage of this trajectory object
        in bytes.
        """
        var_names = [
            self._position_nk2,
            self.valid_horizons_n1,
            self._speed_nk1,
            self._acceleration_nk1,
            self._heading_nk1,
            self._angular_speed_nk1,
            self._angular_acceleration_nk1,
        ]
        return np.sum([var_name.numpy().nbytes for var_name in var_names])

    @classmethod
    def init_from_numpy_repr(
        cls,
        dt: float,
        n: int,
        k: int,
        position_nk2: np.ndarray,
        speed_nk1: np.ndarray,
        acceleration_nk1: np.ndarray,
        heading_nk1: np.ndarray,
        angular_speed_nk1: np.ndarray,
        angular_acceleration_nk1: np.ndarray,
        valid_horizons_n1: np.ndarray,
        track_trajectory_acceleration: Optional[bool] = True,
    ):
        """Utility function to initialize a trajectory object from its numpy
        representation. Useful for loading pickled trajectories"""
        return cls(
            dt=dt,
            n=n,
            k=k,
            position_nk2=position_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            valid_horizons_n1=valid_horizons_n1,
            direct_init=True,
            track_trajectory_acceleration=track_trajectory_acceleration,
        )

    @classmethod
    def from_pos3_array(cls, pos3_nk3: np.ndarray, dt: Optional[float] = 0.05):
        """ Construct a Trajectory from an (n x 3) pos3 array of (x, y, theta) """
        if not isinstance(pos3_nk3, np.ndarray):
            pos3_nk3 = np.array(pos3_nk3)
        n, k, dims = pos3_nk3.shape
        assert dims == 3
        if k == 0:
            return Trajectory.empty(dt)
        position_nk2 = pos3_nk3[:, :, :2].reshape(n, k, 2)
        heading_nk1 = pos3_nk3[:, :, 2].reshape(n, k, 1)
        speed_nk1 = (
            np.sqrt(np.sum(np.diff(position_nk2, axis=1) ** 2, axis=2)) / dt
        ).reshape(n, k - 1, 1)
        acceleration_nk1 = (
            np.sqrt(np.sum(np.diff(speed_nk1, axis=1) ** 2, axis=2)) / dt
        ).reshape(n, k - 2, 1)
        angular_speed_nk1 = (
            np.sqrt(np.sum(np.diff(heading_nk1, axis=1) ** 2, axis=2)) / dt
        ).reshape(n, k - 1, 1)
        angular_acceleration_nk1 = (
            np.sqrt(np.sum(np.diff(angular_speed_nk1, axis=1) ** 2, axis=2)) / dt
        ).reshape(n, k - 2, 1)
        return cls(
            dt=dt,
            n=n,
            k=k,  # only for a single agent's trajectory
            position_nk2=position_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            dtype=np.float32,
            direct_init=True,
            valid_horizons_n1=None,
            track_trajectory_acceleration=True,
            check_dimens=True,
        )

    @classmethod
    def empty(cls, dt: Optional[float] = 0.05):
        return cls(
            dt=dt,
            n=0,
            k=1,  # only for a single agent's trajectory
            position_nk2=np.array([[]]),
            speed_nk1=np.array([]),
            acceleration_nk1=np.array([]),
            heading_nk1=np.array([]),
            angular_speed_nk1=np.array([]),
            angular_acceleration_nk1=np.array([]),
            dtype=np.float32,
            direct_init=True,
            valid_horizons_n1=None,
            track_trajectory_acceleration=True,
            check_dimens=True,
        )

    def update_valid_mask_nk(self) -> None:
        """Update this trajectories valid mask. The valid mask is a mask of 1's
        and 0's at the trajectories sampling interval where 1's represent
        trajectory data within the valid horizon and 0's otherwise."""
        all_valid_nk = np.broadcast_to(
            np.arange(self.k, dtype=np.float32) + 1, (self.n, self.k)
        )
        self.valid_mask_nk = (all_valid_nk <= self.valid_horizons_n1).astype(np.float32)

    def assign_from_trajectory_batch_idx(self, trajectory, batch_idx: int) -> None:
        """Assigns a trajectory object's instance variables from the trajectory stored
        at batch index batch_idx in trajectory."""
        self.assign_trajectory_from_tensors(
            position_nk2=trajectory.position_nk2()[batch_idx : batch_idx + 1],
            speed_nk1=trajectory.speed_nk1()[batch_idx : batch_idx + 1],
            acceleration_nk1=trajectory.acceleration_nk1()[batch_idx : batch_idx + 1],
            heading_nk1=trajectory.heading_nk1()[batch_idx : batch_idx + 1],
            angular_speed_nk1=trajectory.angular_speed_nk1()[batch_idx : batch_idx + 1],
            angular_acceleration_nk1=trajectory.angular_acceleration_nk1()[
                batch_idx : batch_idx + 1
            ],
            valid_horizons_n1=trajectory.valid_horizons_n1[batch_idx : batch_idx + 1],
        )

    def assign_trajectory_from_tensors(
        self,
        position_nk2: np.ndarray,
        speed_nk1: np.ndarray,
        acceleration_nk1: np.ndarray,
        heading_nk1: np.ndarray,
        angular_speed_nk1: np.ndarray,
        angular_acceleration_nk1: np.ndarray,
        valid_horizons_n1: np.ndarray,
    ):
        self._position_nk2 = position_nk2
        self._speed_nk1 = speed_nk1
        self._acceleration_nk1 = acceleration_nk1
        self._heading_nk1 = heading_nk1
        self._angular_speed_nk1 = angular_speed_nk1
        self._angular_acceleration_nk1 = angular_acceleration_nk1
        self.valid_horizons_n1 = valid_horizons_n1

    def gather_across_batch_dim(self, idxs: List[int] or np.ndarray):
        """Given a list of indexes to gather in the batch dimension,
        update this trajectories instance variables and shape."""
        self.n = idxs.size

        self._position_nk2 = self._position_nk2[idxs]
        self._speed_nk1 = self._speed_nk1[idxs]
        self._acceleration_nk1 = self._acceleration_nk1[idxs]
        self._heading_nk1 = self._heading_nk1[idxs]
        self._angular_speed_nk1 = self._angular_speed_nk1[idxs]
        self._angular_acceleration_nk1 = self._angular_acceleration_nk1[idxs]
        self.valid_horizons_n1 = self.valid_horizons_n1[idxs]
        return self

    def to_numpy_repr(self) -> Dict[str, np.ndarray and int and float]:
        """Utility function to return a representation of the trajectory using
        numpy arrays. Useful for pickling trajectories."""
        numpy_dict: Dict[str, np.ndarray and int and float] = {
            "dt": self.dt,
            "n": self.n,
            "k": self.k,
            "position_nk2": self.position_nk2(),
            "speed_nk1": self.speed_nk1(),
            "acceleration_nk1": self.acceleration_nk1(),
            "heading_nk1": self.heading_nk1(),
            "angular_speed_nk1": self.angular_speed_nk1(),
            "angular_acceleration_nk1": self.angular_acceleration_nk1(),
            "valid_horizons_n1": self.valid_horizons_n1,
        }
        return numpy_dict

    @classmethod
    def concat_across_batch_dim(cls, trajs: List):
        """Concatenates a list of trajectory objects
        across the batch dimension, returning a new
        trajectory object."""
        if len(trajs) == 0:
            return None

        position_nk2 = np.concatenate([traj.position_nk2() for traj in trajs], axis=0)
        speed_nk1 = np.concatenate([traj.speed_nk1() for traj in trajs], axis=0)
        acceleration_nk1 = np.concatenate(
            [traj.acceleration_nk1() for traj in trajs], axis=0
        )
        heading_nk1 = np.concatenate([traj.heading_nk1() for traj in trajs], axis=0)
        angular_speed_nk1 = np.concatenate(
            [traj.angular_speed_nk1() for traj in trajs], axis=0
        )
        angular_acceleration_nk1 = np.concatenate(
            [traj.angular_acceleration_nk1() for traj in trajs], axis=0
        )
        valid_horizons_n1 = np.concatenate(
            [traj.valid_horizons_n1 for traj in trajs], axis=0
        )
        return cls(
            dt=trajs[0].dt,  # should all have the same dt
            n=position_nk2.shape[0],  # concat'ed shape
            k=trajs[0].k,  # should all have the same k
            position_nk2=position_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            valid_horizons_n1=valid_horizons_n1,
        )

    @classmethod
    def gather_across_batch_dim_and_create(cls, traj, idxs: List[int] or np.ndarray):
        """Given a list of indexes to gather in the batch dimension,
        gather traj's instance variables across the batch dimension
        creating a new trajectory object."""

        position_nk2 = traj.position_nk2()[idxs]
        speed_nk1 = traj.speed_nk1()[idxs]
        acceleration_nk1 = (
            np.zeros_like(traj.acceleration_nk1())
            if traj.acceleration_nk1().size == 0
            else traj.acceleration_nk1()[idxs]
        )
        heading_nk1 = traj.heading_nk1()[idxs]
        angular_speed_nk1 = traj.angular_speed_nk1()[idxs]
        angular_acceleration_nk1 = (
            np.zeros_like(traj.angular_acceleration_nk1())
            if traj.angular_acceleration_nk1().size == 0
            else traj.angular_acceleration_nk1()[idxs]
        )
        valid_horizons_n1 = traj.valid_horizons_n1[idxs]
        return cls(
            dt=traj.dt,
            n=idxs.size,
            k=traj.k,
            position_nk2=position_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            valid_horizons_n1=valid_horizons_n1,
        )

    @property
    def shape(self) -> str:
        return "({:d}, {:d})".format(self.n, self.k)

    def position_nk2(self) -> np.ndarray:
        return self._position_nk2

    def speed_nk1(self) -> np.ndarray:
        return self._speed_nk1

    def acceleration_nk1(self) -> np.ndarray:
        return self._acceleration_nk1

    def heading_nk1(self) -> np.ndarray:
        return self._heading_nk1

    def angular_speed_nk1(self) -> np.ndarray:
        return self._angular_speed_nk1

    def angular_acceleration_nk1(self) -> np.ndarray:
        return self._angular_acceleration_nk1

    def position_and_heading_nk3(self, squeeze: Optional[bool] = False) -> np.ndarray:
        p_nk2 = self.position_nk2()
        h_nk1 = self.heading_nk1()
        if len(p_nk2) == 0 and len(h_nk1) == 0:
            return np.array([])
        pos2_th1: np.ndarray = np.concatenate([p_nk2, h_nk1], axis=2)
        if squeeze:  # get only the inner-most element(s) if requested
            pos2_th1 = np.squeeze(pos2_th1)
        return pos2_th1

    def speed_and_angular_speed_nk2(self) -> np.ndarray:
        s_nk1 = self.speed_nk1()
        a_nk1 = self.angular_speed_nk1()
        if len(s_nk1) == 0 and len(a_nk1) == 0:
            return np.array([])
        return np.concatenate([s_nk1, a_nk1], axis=2)

    def position_heading_speed_and_angular_speed_nk5(self) -> np.ndarray:
        ph_nk3 = self.position_and_heading_nk3()
        sa_nk2 = self.speed_and_angular_speed_nk2()
        if len(ph_nk3) == 0 and len(sa_nk2) == 0:
            return np.array([])
        return np.concatenate([ph_nk3, sa_nk2], axis=2)

    def append_along_time_axis(
        self, trajectory, track_trajectory_acceleration: Optional[bool] = True
    ) -> None:
        """ Utility function to concatenate trajectory
        over time. Useful for assembling an entire
        trajectory from multiple sub-trajectories. """
        self._position_nk2 = np.concatenate(
            [self.position_nk2(), trajectory.position_nk2()], axis=1
        )
        self._speed_nk1 = np.concatenate(
            [self.speed_nk1(), trajectory.speed_nk1()], axis=1
        )
        if track_trajectory_acceleration:
            self._acceleration_nk1 = np.concatenate(
                [self.acceleration_nk1(), trajectory.acceleration_nk1()], axis=1
            )
            self._angular_acceleration_nk1 = np.concatenate(
                [
                    self.angular_acceleration_nk1(),
                    trajectory.angular_acceleration_nk1(),
                ],
                axis=1,
            )
        self._heading_nk1 = np.concatenate(
            [self.heading_nk1(), trajectory.heading_nk1()], axis=1
        )
        self._angular_speed_nk1 = np.concatenate(
            [self.angular_speed_nk1(), trajectory.angular_speed_nk1()], axis=1
        )
        self.k = self.k + trajectory.k
        self.valid_horizons_n1 = self.valid_horizons_n1 + trajectory.valid_horizons_n1

    def clip_along_time_axis(self, horizon: int) -> None:
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

    def take_along_time_axis(self, horizon: int) -> None:
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
    def concat_along_time_axis(cls, trajectories: List):
        """ Concatenates a list of trajectory objects
        along the time axis. Useful for assembling an entire
        trajectory from multiple sub-trajectories. """

        # Check all subtrajectories have the same batch size and dt
        assert [x.n for x in trajectories] == [1] * len(trajectories)
        assert [x.dt for x in trajectories] == [trajectories[0].dt] * len(trajectories)

        n = trajectories[0].n
        dt = trajectories[0].dt
        k = sum([x.k for x in trajectories])

        position_nk2 = np.concatenate([x.position_nk2() for x in trajectories], axis=1)
        speed_nk1 = np.concatenate([x.speed_nk1() for x in trajectories], axis=1)
        acceleration_nk1 = np.concatenate(
            [x.acceleration_nk1() for x in trajectories], axis=1
        )
        heading_nk1 = np.concatenate([x.heading_nk1() for x in trajectories], axis=1)
        angular_speed_nk1 = np.concatenate(
            [x.angular_speed_nk1() for x in trajectories], axis=1
        )
        angular_acceleration_nk1 = np.concatenate(
            [x.angular_acceleration_nk1() for x in trajectories], axis=1
        )
        valid_horizons_n1 = np.reduce_sum(
            [x.valid_horizons_n1 for x in trajectories], axis=0
        )
        return cls(
            dt=dt,
            n=n,
            k=k,
            position_nk2=position_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            valid_horizons_n1=valid_horizons_n1,
            direct_init=True,
        )

    @classmethod
    def copy(cls, traj, check_dimens: Optional[bool] = True):
        return cls(
            dt=traj.dt,
            n=traj.n,
            k=traj.k,
            position_nk2=np.copy(traj.position_nk2()),
            speed_nk1=np.copy(traj.speed_nk1()),
            acceleration_nk1=np.copy(traj.acceleration_nk1()),
            heading_nk1=np.copy(traj.heading_nk1()),
            angular_speed_nk1=np.copy(traj.angular_speed_nk1()),
            angular_acceleration_nk1=np.copy(traj.angular_acceleration_nk1()),
            valid_horizons_n1=np.copy(traj.valid_horizons_n1),
            direct_init=True,
            check_dimens=check_dimens,
        )

    @classmethod
    def new_traj_clip_along_time_axis(
        cls,
        trajectory,
        horizon: int,
        repeat_second_to_last_speed: Optional[bool] = False,
    ):
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
            speed_nk1 = np.concatenate([speed_nk1[:, :-1], speed_nk1[:, -2:-1]], axis=1)
            angular_speed_nk1 = np.concatenate(
                [angular_speed_nk1[:, :-1], angular_speed_nk1[:, -2:-1]], axis=1
            )

        return cls(
            dt=trajectory.dt,
            n=trajectory.n,
            k=horizon,
            position_nk2=trajectory.position_nk2()[:, :horizon],
            speed_nk1=speed_nk1,
            acceleration_nk1=trajectory.acceleration_nk1()[:, :horizon],
            heading_nk1=trajectory.heading_nk1()[:, :horizon],
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=trajectory.angular_acceleration_nk1()[:, :horizon],
        )

    def __getitem__(self, index: int):
        """Allow for indexing along the batch dimension similar
        to a regular tensor. Returns a new object corresponding
        to the batch index, index"""
        if index >= self.n:
            raise IndexError

        pos_nk2 = self.position_nk2()[index : index + 1]
        speed_nk1 = self.speed_nk1()[index : index + 1]
        acceleration_nk1 = self.acceleration_nk1()[index : index + 1]
        heading_nk1 = self.heading_nk1()[index : index + 1]
        angular_speed_nk1 = self.angular_speed_nk1()[index : index + 1]
        angular_acceleration_nk1 = self.angular_acceleration_nk1()[index : index + 1]
        valid_horizons_n1 = self.valid_horizons_n1[index : index + 1]
        return self.__class__(
            dt=self.dt,
            n=1,  # single point
            k=self.k,
            position_nk2=pos_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            valid_horizons_n1=valid_horizons_n1,
            direct_init=True,
        )

    def render(
        self,
        axs: plt.axes,
        batch_idx: int = 0,
        freq: Optional[int] = 4,
        plot_quiver: Optional[bool] = True,
        plot_heading: Optional[bool] = False,
        plot_velocity: Optional[bool] = False,
        clip: Optional[int] = 0,
        mpl_kwargs: Dict[str, Any] = {},
    ) -> None:
        # use clip to only render the *last* "clip" numpoints of the trajectory
        xs = self._position_nk2[batch_idx, -1 : -1 * clip : -1, 0]
        ys = self._position_nk2[batch_idx, -1 : -1 * clip : -1, 1]
        thetas = self._heading_nk1[batch_idx]

        if plot_quiver:
            axs.quiver(
                xs[::freq], ys[::freq], np.cos(thetas[::freq]), np.sin(thetas[::freq])
            )
        axs.plot(xs, ys, **mpl_kwargs)

        if plot_heading:
            raise NotImplementedError
            axs[1].plot(
                np.r_[: self.k] * self.dt, self._heading_nk1[batch_idx, :, 0], "r-"
            )
            axs[1].set_title("Theta")

        if plot_velocity:
            raise NotImplementedError
            time = np.r_[: self.k] * self.dt

            axs[2].plot(time, self._speed_nk1[batch_idx, :, 0], "r-")
            axs[2].set_title("Linear Velocity")

            axs[3].plot(time, self._angular_speed_nk1[batch_idx, :, 0], "r-")
            axs[3].set_title("Angular Velocity")


class SystemConfig(Trajectory):
    """
    A class representing a system configuration using a trajectory of
    time duration = 1 step.
    """

    def __init__(
        self,
        dt: Optional[float] = 0.05,
        n: Optional[int] = 1,
        k: Optional[int] = 1,
        position_nk2: Optional[np.ndarray] = None,
        speed_nk1: Optional[np.ndarray] = None,
        acceleration_nk1: Optional[np.ndarray] = None,
        heading_nk1: Optional[np.ndarray] = None,
        angular_speed_nk1: Optional[np.ndarray] = None,
        angular_acceleration_nk1: Optional[np.ndarray] = None,
        dtype: Optional[type] = np.float32,
        direct_init: Optional[bool] = False,
        valid_horizons_n1: Optional[np.ndarray] = None,
        track_trajectory_acceleration: Optional[bool] = True,
        check_dimens: Optional[bool] = True,
    ):
        assert k == 1
        # Don't pass on valid_horizons_n1 as a SystemConfig has no horizon
        super().__init__(
            dt=dt,
            n=n,
            k=k,
            position_nk2=position_nk2,
            speed_nk1=speed_nk1,
            acceleration_nk1=acceleration_nk1,
            heading_nk1=heading_nk1,
            angular_speed_nk1=angular_speed_nk1,
            angular_acceleration_nk1=angular_acceleration_nk1,
            dtype=dtype,
            direct_init=direct_init,
            valid_horizons_n1=valid_horizons_n1,
            track_trajectory_acceleration=track_trajectory_acceleration,
            check_dimens=check_dimens,
        )

    def assign_from_broadcasted_batch(self, config, n: int):
        """ Assigns a SystemConfig's variables by broadcasting a given config to
        batch size n """
        k: int = config.k
        self.assign_config_from_tensors(
            position_nk2=np.broadcast_to(config.position_nk2(), (n, k, 2)),
            speed_nk1=np.broadcast_to(config.speed_nk1(), (n, k, 1)),
            acceleration_nk1=np.broadcast_to(config.acceleration_nk1(), (n, k, 1)),
            heading_nk1=np.broadcast_to(config.heading_nk1(), (n, k, 1)),
            angular_speed_nk1=np.broadcast_to(config.angular_speed_nk1(), (n, k, 1)),
            angular_acceleration_nk1=np.broadcast_to(
                config.angular_acceleration_nk1(), (n, k, 1)
            ),
        )

    @classmethod
    def from_pos3(
        cls,
        pos3: np.ndarray or List[float] or Tuple[float],
        dt: Optional[float] = 0.05,
        v: Optional[float] = 0,
        w: Optional[float] = 0,
    ):
        """ Construct a SystemConfig from a list/tuple/np of (x, y, theta) 
        with optional velocity and angular velocity """
        if isinstance(pos3, np.ndarray):
            assert pos3.shape == (3,)
        else:
            assert len(pos3) == 3
        return cls(
            dt=dt,
            position_nk2=np.array([[pos3[:2]]], dtype=np.float32),
            speed_nk1=np.array([[[v]]], dtype=np.float32),
            acceleration_nk1=np.array([[[0]]], dtype=np.float32),
            heading_nk1=np.array([[[pos3[2]]]], dtype=np.float32),
            angular_speed_nk1=np.array([[[w]]], dtype=np.float32),
            angular_acceleration_nk1=np.array([[[0]]], dtype=np.float32),
            dtype=np.float32,
            direct_init=True,
            valid_horizons_n1=None,
            track_trajectory_acceleration=True,
            check_dimens=True,
        )

    @classmethod
    def init_config_from_trajectory_time_index(cls, trajectory: Trajectory, idx: int):
        """ A utility method to initialize a config object
        from a particular timestep of a given trajectory object"""
        position_nk2 = trajectory.position_nk2()
        speed_nk1 = trajectory.speed_nk1()
        acceleration_nk1 = trajectory.acceleration_nk1()
        heading_nk1 = trajectory.heading_nk1()
        angular_speed_nk1 = trajectory.angular_speed_nk1()
        angular_acceleration_nk1 = trajectory.angular_acceleration_nk1()

        if idx == -1:
            return cls(
                dt=trajectory.dt,
                n=trajectory.n,
                k=1,
                position_nk2=position_nk2[:, idx:],
                speed_nk1=speed_nk1[:, idx:],
                acceleration_nk1=acceleration_nk1[:, idx:],
                heading_nk1=heading_nk1[:, idx:],
                angular_speed_nk1=angular_speed_nk1[:, idx:],
                angular_acceleration_nk1=angular_acceleration_nk1[:, idx:],
            )

        return cls(
            dt=trajectory.dt,
            n=trajectory.n,
            k=1,
            position_nk2=position_nk2[:, idx : idx + 1],
            speed_nk1=speed_nk1[:, idx : idx + 1],
            acceleration_nk1=acceleration_nk1[:, idx : idx + 1],
            heading_nk1=heading_nk1[:, idx : idx + 1],
            angular_speed_nk1=angular_speed_nk1[:, idx : idx + 1],
            angular_acceleration_nk1=angular_acceleration_nk1[:, idx : idx + 1],
        )

    def assign_from_config_batch_idx(self, config, batch_idx: int) -> None:
        super(SystemConfig, self).assign_from_trajectory_batch_idx(config, batch_idx)

    def assign_config_from_tensors(
        self,
        position_nk2: np.ndarray,
        speed_nk1: np.ndarray,
        acceleration_nk1: np.ndarray,
        heading_nk1: np.ndarray,
        angular_speed_nk1: np.ndarray,
        angular_acceleration_nk1: np.ndarray,
    ) -> None:
        super().assign_trajectory_from_tensors(
            position_nk2,
            speed_nk1,
            acceleration_nk1,
            heading_nk1,
            angular_speed_nk1,
            angular_acceleration_nk1,
            valid_horizons_n1=None,
        )
