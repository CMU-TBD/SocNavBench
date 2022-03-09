# Installation (Linux, x64)

# 1. Download SMPL data & Render human meshes
Follow the instructions in [`surreal/README.md`](../surreal/README.md) to correctly install the human meshes.

# 2. Download environment data
To download our custom curated maps (edited from the SD3DIS models) follow this link to our [official online drive](https://drive.google.com/drive/folders/1LAySlmE9dwrTghnL3Y5gE62K5cDJkPm1?usp=sharing) and download the `stanford_builder_parser_dataset` and place it in the `sd3dis` directory as is. 

However, if you'd like to download and configure all the original maps, follow the instructions in [`sd3dis/README.md`](../sd3dis/README.md) to correctly install the building/area meshes. 

Note: `SocNavBench` is independent of the actual indoor office environment and human meshes used. In this work we use human meshes exported from the [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) dataset and scans of indoor office environments from the [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset.

# 3. Install system dependencies
```bash
sudo apt install gcc g++ libassimp-dev ffmpeg
```

# 4. Setup A Virtual Environment
Install Anaconda for Linux: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
```bash
conda env create -f environment.yml # this will install all python dependencies
conda activate socnavbench
```

# 5. Patch OpenGL and Pyassimp
In the terminal run the following commands.
```bash
bash ./socnav/patches/apply_patches.sh
```

# 6. Generate intermediate map files
The "traversible" files (`.pkl`) are generated once per map (building `.obj`) and should all be initially generated with this script. If you add custom maps you can add their name to the line including the maps (`maps = ["DoubleHotel", "ETH", "Hotel", "Univ", "Zara"]`) since these are the only ones we provide. 
```bash
# From the base SocNavBench directory
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/generate_traversibles.py
```

# 7. [Optional]Run the SocNavBench tests
To get you started we've included `tests`, which contains the main code example for testing the simulator mechanics (`test_socnav.py`) as well as testing multiple episodes (`test_episodes.py`) which can both be configured in `schematic` or `full-render` mode.

```bash
# From the base SocNavBench directory

# Ensure the unit tests succeed with 
PYTHONPATH='.' python3 tests/all_unit_tests.py 

# test the socnav simulator itself with 
# (the PYOPENGL env var is only used for the 3d renderer)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_socnav.py
...
# In a separate shell (as a separate executable):
PYTHONPATH='.' python3 joystick/joystick_client.py
```

# Now what?
To get started with using SocNavBench for your own social navigation benchmarking, see [`usage.md`](usage.md)