# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Wrapper for selecting the navigation environment that we want to train and
test on.
"""
import os
import glob
import pickle
import numpy as np

from mp_env.mp_env import Building
from mp_env.map_utils import Foo
from params.central_params import get_sbpd_data_dir
from utils.utils import *
from mp_env.render import swiftshader_renderer as renderer


def get_dataset(dataset_name, imset, data_dir, surreal_params=None):
    if dataset_name == 'sbpd':

        dataset = StanfordBuildingParserDataset(imset,
                                                data_dir=data_dir,
                                                surreal_params=surreal_params)
    else:
        assert(False and 'Not one of sbpd')
    return dataset


class Loader():
    def load_building(self, name, data_dir=None):
        if data_dir is None:
            data_dir = get_sbpd_data_dir()  # when not using surreal
        out = {}
        out['name'] = name
        out['data_dir'] = data_dir
        return out

    def load_building_meshes(self, building, materials_scale=1.0):
        dir_name = os.path.join(building['data_dir'], 'mesh', building['name'])
        obj_files = glob.glob1(dir_name, '*.obj')
        assert len(obj_files) > 0 and "could not find .obj file"
        mesh_file_name = obj_files[0]
        mesh_file_name_full = os.path.join(dir_name, mesh_file_name)
        #logging.error('Loading building from obj file: %s', mesh_file_name_full)
        print("%sLoading building mesh from obj:" % (color_text["blue"]),
              mesh_file_name_full, "%s" % (color_text["reset"]))
        shape = renderer.Shape(mesh_file_name_full, load_materials=True,
                               name_prefix=building['name'] + '_', materials_scale=materials_scale)
        return [shape]

    def load_data(self, name, robot, flip=False):
        env = Foo(padding=10, resolution=5, num_point_threshold=2,
                  valid_min=-10, valid_max=200, n_samples_per_face=200)
        building = Building(self, name, robot, env, flip=flip)
        return building

    def load_random_human(self, speed, gender, human_materials, body_shape, rng):
        """
        Load a human mesh of random pose, shape, gender
        and compute the corresponding center of this human
        mesh. Assumes the human mesh data is stored in this fasion:
            surreal_dir/pose_dir/body_shape_dir/gender/human_mesh_{:d}.obj
        """

        # Find the closest velocity bin
        velocity_dirs = [x for x in os.listdir(self.surreal_params.data_dir) if os.path.isdir(
            os.path.join(self.surreal_params.data_dir, x))]
        velocities_float = [float(velocity_str.split('velocity_')[1].split('_m_s')[
                                  0]) for velocity_str in velocity_dirs]
        idx = np.argmin(np.abs(np.array(velocities_float) - speed))
        velocity_dir = os.path.join(
            self.surreal_params.data_dir, velocity_dirs[idx])

        # Sample A Random Pose
        poses = [x for x in os.listdir(velocity_dir) if not x.startswith('.')]
        poses.sort()
        pose = rng.choice(poses)
        pose_dir = os.path.join(velocity_dir, pose)

        # Choose the Body Shape Directory
        body_shape_dir = os.path.join(
            pose_dir, 'body_shape_{:d}'.format(body_shape))
        assert os.path.isdir(body_shape_dir)

        # Sample Gender
        gender_dir = os.path.join(body_shape_dir, gender)
        assert os.path.isdir(gender_dir)

        # Sample a frame number
        frames = os.listdir(gender_dir)
        frames = list(filter(lambda x: 'obj' in x, frames))
        frame_numbers = [int(x.strip('.obj').split('_')[-1]) for x in frames]
        frame_numbers.sort()
        frame = rng.choice(frame_numbers)

        human_mesh_info = {'mesh_dir': gender_dir,
                           'frame': frame, 'gender': gender}
        shapess, center_pos_3 = self.load_human_mesh(
            human_materials=human_materials, **human_mesh_info)
        return shapess, center_pos_3, human_mesh_info

    def load_human_mesh(self, human_materials, mesh_dir, frame, gender):
        """
        Loads the human mesh named human_mesh_{:d}.obj
        in mesh_dir
        """
        # Load the Human Mesh
        mesh_file = os.path.join(mesh_dir, 'human_mesh_{:d}.obj'.format(frame))
        human = renderer.HumanShape(
            mesh_file, human_materials, name_prefix='human')

        # Load data which tells us the approximate (x, y, theta) configuration
        # of the human. This is computed as the centerpoint between the humans two
        # feet assuming they are pointing halfway between the direction of each foot
        centering_file = os.path.join(
            mesh_dir, 'human_centering_info_{:d}.pkl'.format(frame))
        with open(centering_file, 'rb') as f:
            centering_data = pickle.load(f)
        human_pos_3 = centering_data['human_pos_3']

        return [human], human_pos_3


class StanfordBuildingParserDataset(Loader):
    def __init__(self, imset, data_dir=None, surreal_params=None):
        self.imset = imset
        self.data_dir = data_dir

        self.surreal_params = surreal_params

    def get_data_dir(self):
        return self.data_dir

    def get_benchmark_sets(self):
        return self._get_benchmark_sets()

    def get_split(self):
        return self._get_split(self.imset)

    def get_imset(self):
        return self._get_split(self.imset)

    def _get_benchmark_sets(self):
        sets = ['train1', 'train2', 'val', 'test']
        return sets

    def _get_split(self, split_name):
        train = ['area1', 'area5a', 'area5b', 'area6']
        train1 = ['area1']
        train2 = ['area1', 'area5a']
        train2x2 = ['area1+area5a', 'area5b+area6']
        train1x4 = ['area1+area5a+area5b+area6']
        val = ['area3']
        test = ['area4']

        sets = {}
        sets['train'] = train
        sets['train1'] = train1
        sets['train2'] = train2
        sets['train2x2'] = train2x2
        sets['train1x4'] = train1x4
        sets['val'] = val
        sets['test'] = test
        sets['all'] = sorted(list(set(train + val + test)))
        return sets[split_name]

    def get_random_human_gender_texture_and_body_shape(self, rng, load_materials=True):
        # Sample a random gender for the human
        genders = ['male', 'female']
        gender = rng.choice(genders)

        # Sample a random set of materials for the human (i.e. skin/hair color, clothing, etc.)
        human_materials = renderer.HumanShape.get_random_materials(self.surreal_params.texture_dir,
                                                                   self.surreal_params.mode,
                                                                   gender, rng, load_materials=load_materials)

        # Sample a random body shape
        if self.surreal_params.mode == 'train':
            body_shape = rng.choice(self.surreal_params.body_shapes_train)
        elif self.surreal_params.mode == 'test':
            body_shape = rng.choice(self.surreal_params.body_shapes_test)
        else:
            raise NotImplementedError
        return gender, human_materials, body_shape
