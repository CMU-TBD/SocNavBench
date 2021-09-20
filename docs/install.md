# Installation

## Download and Configure Data

### Download SMPL data & Render human meshes
Follow the instructions in [`surreal/README.md`](../surreal/README.md) to correctly install the human meshes.

### Download enrivonment data
To download our custom curated maps (edited from the SD3DIS models) follow this link to our [official online drive](https://drive.google.com/drive/folders/1LAySlmE9dwrTghnL3Y5gE62K5cDJkPm1?usp=sharing) and download the `stanford_builder_parser_dataset` and place it in the `sd3dis` directory as is. 

However, if you'd like to download and configure all the original maps, follow the instructions in [`sd3dis/README.md`](../sd3dis/README.md) to correctly install the building/area meshes. 

Note: `SocNavBench` is independent of the actual indoor office environment and human meshes used. In this work we use human meshes exported from the [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) dataset and scans of indoor office environments from the [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset.

## Setup
### Install Anaconda, gcc, g++, libassimp-dev
```bash
# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh

# Install gcc and g++ if you don't already have them
sudo apt install gcc g++ libassimp-dev ffmpeg
```

### Setup A Virtual Environment
```bash
conda env create -f environment.yml # this will install all required dependencies
conda activate socnavbench
```

#### Patch the OpenGL Installation
In the terminal run the following commands.
```bash
cd /PATH/TO/SocNavBench/socnav
bash patches/apply_patches_3.sh
# NOTE: after running get_packages.sh you should see:
# HUNK #3 succeeded at 401 (offset 1 line).
# Hunk #4 succeeded at 407 (offset 1 line).
```
If the script fails there are instructions in [`apply_patches_3.sh`](../socnav/patches/apply_patches_3.sh) describing how to manually apply the patch. 

### Manually patch pyassimp bug
Additionally, this version of `pyassimp` has a bug which can be fixed by following [this commit](https://github.com/assimp/assimp/commit/b6d3cbcb61f4cc4c42678d5f183351f95c97c8d4) and simply changing `isinstance(obj,int)` to `isinstance(obj, (int, str, bytes))` on line (approx) 91 of `anaconda3/envs/socnavbench/lib/python3.6/site-packages/pyassimp/core.py`. Then try running the patches again, or manually (not recommended).


## Generate intermediate map files
The "traversible" files (`.pkl`) are generated once per map (building `.obj`) and should all be initially generated with this script. If you add custom maps you can add their name to the line including the maps (`maps = ["DoubleHotel", "ETH", "Hotel", "Univ", "Zara"]`) since these are the only ones we provide. 
```bash
# From the base SocNavBench directory
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/generate_traversibles.py
```
## Run the SocNavBench tests
To get you started we've included `tests`, which contains the main code example for testing the simulator mechanics (`test_socnav.py`) as well as testing multiple episodes (`test_episodes.py`) which can both be configured in `schematic` or `full-render` mode.

```bash
# From the base SocNavBench directory

# Ensure the unit tests succeed with 
PYTHONPATH='.' python3 tests/all_unit_tests.py 

# test the socnav simulator itself with 
# (the PYOPENGL env var is only used for the 3d renderer)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_socnav.py
...
# In a seperate shell (as a separate executable):
PYTHONPATH='.' python3 joystick/joystick_client.py
```

## Run the multi-robot renderer
Assuming you have configured the desired robots to "render" side by side in (`user_params.ini`)[params/user_params.ini] under `[renderer_params]` (see `draw_parallel_robots_params_by_algo`) you will be able to render multiple robot trajectories side-by-side via.

```bash
# assuming all the joysticks denoted in run_multi_robot.sh have been set up correctly on your installation
./run_multi_robot.sh

# once it completes
PYTHONPATH='.' python3 tests/test_multi_robot.py
```
![multi-robot-demo](figures/multi-robot-demo.gif)

For more information see [`usage.md`](usage.md)'s section **Multi-robot mode**