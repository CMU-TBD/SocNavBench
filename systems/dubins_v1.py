from systems.dubins_3d import Dubins3D
import numpy as np


class DubinsV1(Dubins3D):
    """ A discrete time 3 dimensional dubins car with identity saturation
    functions (i.e. no saturation) on linear or angular velocity.
    """
    name = 'dubins_v1'

    def __init__(self, dt, params=None):
        if(params is not None):
            super().__init__(dt, params.simulation_params)
        else:
            super().__init__(dt)

    def _saturate_linear_velocity(self, vtilde_nk):
        """ Identity saturation function for linear velocity"""
        return vtilde_nk

    def _saturate_angular_velocity(self, wtilde_nk):
        """ Identity saturation function for angular velocity"""
        return wtilde_nk

    def _saturate_linear_velocity_prime(self, vtilde_nk):
        """ Time derivative of identity function"""
        return np.ones_like(vtilde_nk)

    def _saturate_angular_velocity_prime(self, wtilde_nk):
        """ Time derivative of identity function"""
        return np.ones_like(wtilde_nk)
