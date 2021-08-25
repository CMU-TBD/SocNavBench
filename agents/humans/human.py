from random import randint
from typing import Any, Dict, List, Optional, Tuple

from numpy.random import RandomState
from agents.agent import Agent
from dotmap import DotMap
from trajectory.trajectory import SystemConfig
from utils.utils import generate_name, generate_random_config


class HumanAppearance:

    # Static variable shared amongst all human appearances
    # This dataset holds all the SURREAL human textures and meshes
    # TODO: fix circular dep: from sbpd.sbpd import StanfordBuildingParserDataset
    # dataset: Optional[StanfordBuildingParserDataset] = None
    dataset = None

    def __init__(
        self, gender: str, texture: List[str], shape: int, mesh_rng: RandomState
    ):
        self.gender: str = gender
        self.shape: int = shape
        self.texture: List[str] = texture
        self.mesh_rng: RandomState = mesh_rng

    # Getters for the HumanAppearance class
    def get_shape(self) -> int:
        return self.shape

    def get_gender(self) -> str:
        return self.gender

    def get_texture(self) -> List[str]:
        return self.texture

    def get_mesh_rng(self) -> RandomState:
        return self.mesh_rng

    @classmethod
    def generate_rand_human_appearance(cls):
        """
        Sample a new human from known identity features, but unknown 
        positional/speed arguments (and mesh rng)
        """
        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng: int = RandomState(randint(1, 1000))

        gender, texture, shape = HumanAppearance.random_human_identity_from_dataset()

        return cls(gender, texture, shape, mesh_rng)

    @staticmethod
    def random_human_identity_from_dataset() -> Tuple[str, List[str], int]:
        """
        Sample a new human identity, but don't load it into
        memory
        """
        # Set the identity seed. this is used to sample the indentity that generates
        # the human gender, texture, and body shape
        identity_rng: RandomState = RandomState(randint(1, 1000))
        # Collecting Humanav dataset
        if HumanAppearance.dataset is None:
            print("\033[31m", "ERROR: can't find Surreal Dataset", "\033[0m")
            exit(1)  # Failure condition
        # Using the SBPD dataset to generate a random gender, texture, and body shape
        (
            human_gender,
            human_texture,
            body_shape,
        ) = HumanAppearance.dataset.get_random_human_gender_texture_and_body_shape(
            identity_rng
        )
        return human_gender, human_texture, body_shape


class Human(Agent):
    def __init__(
        self,
        name: str,
        appearance: HumanAppearance,
        start_config: SystemConfig,
        goal_config: SystemConfig,
    ):
        self.appearance: HumanAppearance = appearance
        super().__init__(start_config, goal_config, name)

    # Getters for the Human class
    # NOTE: most of the dynamics/configs implementation is in Agent.py

    def get_appearance(self) -> HumanAppearance:
        return self.appearance

    @classmethod
    def generate_human(
        cls,
        start_config: Optional[SystemConfig] = None,
        goal_config: Optional[SystemConfig] = None,
        environment: Optional[Dict[str, Any]] = None,
        appearance: Optional[HumanAppearance] = None,
        generate_appearance: Optional[bool] = False,
        name: Optional[str] = None,
        max_chars: Optional[int] = 20,
        verbose: Optional[bool] = False,
    ) -> Agent:  # technically a Human
        """
        Sample a new random human from all required features
        """
        human_name: str = name if name else generate_name(max_chars)
        # generate the appearance if requested
        if appearance is None and generate_appearance:
            appearance = HumanAppearance.generate_rand_human_appearance()

        # generate the configs if environment provided
        if environment is not None:
            if start_config is None:
                start_config = generate_random_config(environment)
            if goal_config is None:
                goal_config = generate_random_config(environment)

        if verbose:
            assert start_config is not None
            assert goal_config is not None
            pos_2 = list(start_config.position_nk2()[0][0])
            goal_2 = list(goal_config.position_nk2()[0][0])
            print(
                "Generated human {} starting at {} with goal {}".format(
                    human_name, pos_2, goal_2
                )
            )
        return cls(human_name, appearance, start_config, goal_config)

    @staticmethod
    def generate_humans(
        p: DotMap, starts: List[List[float]], goals: List[List[float]]
    ) -> List[Agent]:
        """
        Generate and add num_humans number of randomly generated humans to the simulator
        """
        num_gen_humans: int = min(len(starts), len(goals))
        print("Generating {} autonomous human agents".format(num_gen_humans))

        generated_humans: List[Agent] = []
        for i in range(num_gen_humans):
            start_config = SystemConfig.from_pos3(starts[i])
            goal_config = SystemConfig.from_pos3(goals[i])
            human_i_name = "auto_%04d" % i
            # Generates a new human from the configs
            new_human_i = Human.generate_human(
                start_config=start_config,
                goal_config=goal_config,
                generate_appearance=p.render_3D,
                name=human_i_name,
            )
            # Input human fields into simulator
            generated_humans.append(new_human_i)
        return generated_humans
