import os
import shutil
import sys

sys.path.insert(0, ".")

female_train_filename = 'txt/female_train_textures.txt'
male_train_filename = 'txt/male_train_textures.txt'
female_test_filename = 'txt/female_test_textures.txt'
male_test_filename = 'txt/male_test_textures.txt'

base_outdir = 'human_textures'

import config
params = config.load_file('config', 'SYNTH_DATA')
smpl_data_dir = params['smpl_data_folder']

def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def copy_human_textures(mode, gender, filename):
    outdir = os.path.join(base_outdir, mode, gender)
    mkdir_if_missing(outdir)

    with open(filename, 'r') as f:
        textures = f.read().split('\n')

    # Append base directory to each texture
    # and Remove any empty string characters
    textures = [os.path.join(smpl_data_dir, 'textures', gender, x) for x in textures if x is not '']

    # Copy the relevant files over
    [shutil.copy(f, outdir) for f in textures]

def collect_human_textures():
    copy_human_textures('train', 'female', female_train_filename)
    copy_human_textures('train', 'male', male_train_filename)
    copy_human_textures('test', 'female', female_test_filename)
    copy_human_textures('test', 'male', male_test_filename)


if __name__ == '__main__':
    collect_human_textures()
