from agents.humans.human_appearance import HumanAppearance
from agents.humans.human_configs import HumanConfigs
from utils.utils import generate_name, generate_config_from_pos_3
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
        human_name = name if name is not None else generate_name(max_chars)
        if verbose:
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
        if generate_appearance:
            appearance = \
                HumanAppearance.generate_rand_human_appearance(HumanAppearance)
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
            appearance = \
                HumanAppearance.generate_rand_human_appearance(HumanAppearance)
        configs = HumanConfigs.generate_random_human_config(environment)
        return Human.generate_human(appearance, configs)

    @staticmethod
    def generate(simulator, p, starts, goals, environment, r):
        """
        Generate and add num_humans number of randomly generated humans to the simulator
        """
        num_gen_humans = min(len(starts), len(goals))
        print("Generating auto humans:", num_gen_humans)
        from agents.humans.human_configs import HumanConfigs
        for i in range(num_gen_humans):
            start_config = generate_config_from_pos_3(starts[i])
            goal_config = generate_config_from_pos_3(goals[i])
            start_goal_configs = HumanConfigs(start_config, goal_config)
            human_i_name = "auto_%04d" % i
            # Generates a random human from the environment
            new_human_i = Human.generate_human_with_configs(
                start_goal_configs,
                generate_appearance=p.render_3D,
                name=human_i_name
            )
            # update renderer and get human traversible if it exists
            if p.render_3D:
                r.add_human(new_human_i)
                environment["human_traversible"] = \
                    np.array(r.get_human_traversible())

            # Input human fields into simulator
            simulator.add_agent(new_human_i)
