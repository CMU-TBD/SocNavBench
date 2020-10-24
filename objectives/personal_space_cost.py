# import numpy as np
from objectives.objective_function import Objective
from simulators.sim_state import SimState, get_all_agents
from metrics.cost_functions import *


class PersonalSpaceCost(Objective):
    """
    Compute the cost of being in non ego gen_agents' path.
    """

    def __init__(self, params):
        self.p = params
        self.tag = 'personal_space_cost_per_nonego_agent'

    def evaluate_objective(self, trajectory, sim_state_hist: SimState):
        # get ego agent trajectory
        ego_traj = trajectory.position_and_heading_nk3()

        # get the last sim_state if it exists
        if len(sim_state_hist) > 0:
            sim_state = sim_state_hist[max(sim_state_hist.keys())]
        else:
            return 0

        # loop through each trajectory point
        n, k, _ = ego_traj.shape
        personal_space_cost = np.zeros((1, k))
        for i in range(k):
            ego_pos3 = ego_traj[0, i]  # (x,y,th)_self latest timestep

            # iterate through every non ego agent
            agents = sim_state.get_all_agents() if isinstance(sim_state, SimState) else get_all_agents(sim_state)

            for agent_name, agent_vals in agents.items():
                agent_pos3 = agent_vals.get_pos3()  # (x,y,th)
                theta = agent_pos3[2]
                # gaussian centered around the non ego agent
                # TODO actually account for velocity here
                personal_space_cost[0, i] += asym_gauss_from_vel(x=ego_pos3[0], y=ego_pos3[1],
                                           velx=np.cos(theta), vely=np.sin(theta),
                                           xc=agent_pos3[0], yc=agent_pos3[1])

        return self.p.psc_scale * personal_space_cost
