from agents.humans.human_appearance import HumanAppearance
from agents.humans.human_configs import HumanConfigs
from utils.utils import *
from agents.agent import Agent
import numpy as np


class Human(Agent):
    def __init__(self, name, appearance, start_configs):
        self.name = name
        self.appearance = appearance
        super().__init__(start_configs.get_start_config(),
                         start_configs.get_goal_config(), name)

    # Getters for the Human class
    # NOTE: most of the dynamics/configs implementation is in Agent.py

    def get_appearance(self):
        return self.appearance

    @staticmethod
    def generate_human(appearance, configs, name=None, max_chars=20, verbose=False):
        """
        Sample a new random human from all required features
        """
        human_name = None
        if(name is None):
            human_name = generate_name(max_chars)
        else:
            human_name = name
        if(verbose):
            # In order to print more readable arrays
            np.set_printoptions(precision=2)
            pos_2 = (configs.get_start_config().position_nk2())[0][0]
            goal_2 = (configs.get_goal_config().position_nk2())[0][0]
            print(" Human", human_name, "at", pos_2, "with goal", goal_2)
        return Human(human_name, appearance, configs)

    @staticmethod
    def generate_human_with_appearance(appearance,
                                       environment):
        """
        Sample a new human with a known appearance at a random 
        config with a random goal config.
        """
        configs = HumanConfigs.generate_random_human_config(environment)
        return Human.generate_human(appearance, configs)

    @staticmethod
    def generate_human_with_configs(configs, generate_appearance=False, name=None, verbose=False):
        """
        Sample a new random from known configs and a randomized
        appearance, if any of the configs are None they will be generated
        """
        if(generate_appearance):
            appearance = HumanAppearance.generate_random_human_appearance(
                HumanAppearance)
        else:
            appearance = None
        return Human.generate_human(appearance, configs, verbose=verbose, name=name)

    @staticmethod
    def generate_random_human_from_environment(environment,
                                               generate_appearance=False):
        """
        Sample a new human without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        appearance = None
        if generate_appearance:
            appearance = HumanAppearance.generate_random_human_appearance(
                HumanAppearance)
        configs = HumanConfigs.generate_random_human_config(environment)
        return Human.generate_human(appearance, configs)
