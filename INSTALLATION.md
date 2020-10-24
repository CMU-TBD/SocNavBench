# Installation

## Download and Configure Data

### Download SMPL data & Render human meshes
Follow the instructions in [`surreal/README.md`](surreal/README.md) to correctly install the human meshes.

### Download enrivonment data
To download our custom curated maps (edited from the SD3DIS models) follow this link to our [official online drive](https://drive.google.com/drive/folders/1LAySlmE9dwrTghnL3Y5gE62K5cDJkPm1?usp=sharing) and download the `stanford_builder_parser_dataset` and place it in the `sd3dis` directory as is. 

However, if you'd like to download and configure all the original maps, follow the instructions in [`sd3dis/README.md`](sd3dis/README.md) to correctly install the building/area meshes. 

Note: `SocNavBench` is independent of the actual indoor office environment and human meshes used. In this work we use human meshes exported from the [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) dataset and scans of indoor office environments from the [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset. However, if you would like to use other meshes, please download and configure them yourself and update the parameters in `central_params.py` to point to your data installation.

## Setup
### Install Anaconda, gcc, g++, libassimp-dev
```
# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh

# Install gcc and g++ if you don't already have them
sudo apt-get install gcc
sudo apt-get install g++

# Install libassimp-dev if you don't already have it
sudo apt-get install libassimp-dev
```

### Setup A Virtual Environment
```
conda env create -f environment.yml # this might end with "Pip failed" which is fine
conda activate socnavbench
```

#### Install pip/conda packages
In the terminal (and in the virtual environment from above `socnavbench`) run:
```
chmod a+x get_packages.sh
bash get_packages.sh # make sure the socnavbench conda environment is active!
```
The script will inform you of all packages being installed and their status, they can also all be install manually


#### Patch the OpenGL Installation
In the terminal run the following commands.
```
1. /PATH/TO/SocNavBench/socnav
2. bash patches/apply_patches_3.sh
# NOTE: after running get_packages.sh you should see:
# HUNK #3 succeeded at 401 (offset 1 line).
# Hunk #4 succeeded at 407 (offset 1 line).
```
If the script fails there are instructions in [`apply_patches_3.sh`](socnav/patches/apply_patches_3.sh) describing how to manually apply the patch. 

### Manually patch pyassimp bug
Additionally, this version of `pyassimp` has a bug which can be fixed by following [this commit](https://github.com/assimp/assimp/commit/b6d3cbcb61f4cc4c42678d5f183351f95c97c8d4) and simply changing `isinstance(obj,int)` to `isinstance(obj, (int, str, bytes))` on line 98 of `anaconda3/envs/socnavbench/lib/python3.6/site-packages/pyassimp/core.py`. Then try running the patches again, or manually (not recommended).


## Run the SocNavBench tests
To get you started we've included `tests`, which contains the main code example for rendering the central simulaton and various unit tests. 

```
# From the base SocNavBench directory

# Ensure the unit tests succeed with 
python3 tests/all_unit_tests.py 

# test the socnav simulator itself with 
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_socnav.py
...
# In a seperate shell (as a separate executable):
python3 joystick/test_example_joystick.py 
```
