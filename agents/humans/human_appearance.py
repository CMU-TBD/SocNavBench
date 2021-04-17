from random import randint
import numpy as np


class HumanAppearance():
    # Static variable shared amongst all human appearances
    # This dataset holds all the SURREAL human textures and meshes
    dataset = None

    def __init__(self, gender, texture, shape, mesh_rng):
        self.gender = gender
        self.shape = shape
        self.texture = texture
        self.mesh_rng = mesh_rng

    # Getters for the HumanAppearance class
    def get_shape(self):
        return self.shape

    def get_gender(self):
        return self.gender

    def get_texture(self):
        return self.texture

    def get_mesh_rng(self):
        return self.mesh_rng

    def generate_human_appearance(self, gender, texture, shape, mesh_rng):
        """
        Sample a new random human from all required features
        """
        return HumanAppearance(gender, texture, shape, mesh_rng)

    def create_random_human_identity_from_dataset(self):
        """
        Sample a new human identity, but don't load it into
        memory
        """
        # Set the identity seed. this is used to sample the indentity that generates
        # the human gender, texture, and body shape
        identity_rng = np.random.RandomState(randint(1, 1000))
        # Collecting Humanav dataset
        dataset = HumanAppearance.dataset
        if(dataset is None):
            print('\033[31m', "ERROR: can't find Surreal Dataset", '\033[0m')
            exit(1)  # Failure condition
        # Using the SBPD dataset to generate a random gender, texture, and body shape
        human_gender, human_texture, body_shape = \
            dataset.get_random_human_gender_texture_and_body_shape(
                identity_rng)
        return human_gender, human_texture, body_shape

    def generate_rand_human_appearance(self):
        """
        Sample a new human from known identity features, but unknown 
        positional/speed arguments (and mesh rng)
        """
        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng = np.random.RandomState(randint(1, 1000))

        gender, texture, shape = \
            self.create_random_human_identity_from_dataset(self)

        return self.generate_human_appearance(self, gender, texture, shape, mesh_rng)
