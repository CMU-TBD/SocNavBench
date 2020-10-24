## Generate Human Walking Meshes From SURREAL

The SocNavBench 3D renderer uses photorealistic textured meshes from the [SURREAL](https://www.di.ens.fr/willow/research/surreal/) dataset. Please follow the following setup instructions; #1 and #2 are modified from the [SURREAL GitHub page](https://github.com/gulvarol/surreal).

## 1. Accept the SURREAL dataset license

The links to license terms and download procedure are available here:

https://www.di.ens.fr/willow/research/surreal/data/

Once you receive the credentials to download the dataset, you will have a personal username and password. 

## 2. Create your own synthetic data
### 2.1. Preparation
#### 2.1.1. SMPL data (~2.6 GB)
a) With the same credentials as with the SURREAL dataset, you can download the necessary SMPL data and place it in `/PATH/TO/SocNavBench/surreal/download/SURREAL/smpl_data`.

``` 
./download_smpl_data.sh /PATH/TO/SocNavBench/surreal/download yourusername yourpassword
```

b) You need to download SMPL for MAYA from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following two files (~ 40 MB).

```
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

Place these two files under `/PATH/TO/SocNavBench/surreal/download/SURREAL/smpl_data` folder.

The folder `/PATH/TO/SocNavBench/surreal/download/SURREAL/smpl_data` should contain the folling files and folders:
```
/PATH/TO/SocNavBench/surreal/download/SURREAL/smpl_data/
    - basicModel_f_lbs_10_207_0_v1.0.2.fbx
    - basicModel_m_lbs_10_207_0_v1.0.2.fbx
    - female_beta_stds.npy
    - male_beta_stds.npy
    - smpl_data.npz
    - textures/
```


#### 2.1.2. Blender
You need to download [Blender](http://download.blender.org/release/) and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:

``` 
# Install blender 2.78
wget http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2

# Un-TAR Blender
tar xjf blender-2.78a-linux-glibc211-x86_64.tar.bz2 

# Export the BLENDER_PATH
export BLENDER_PATH='/path/to/blender/blender-2.78-linux-glibc219-x86_64'

# Install pip
wget https://bootstrap.pypa.io/get-pip.py
$BLENDER_PATH/2.78/python/bin/python3.5m get-pip.py

# Make sure you have libglu1
sudo apt-get install libglu1

# Install scipy
$BLENDER_PATH/2.78/python/bin/python3.5m -m pip install scipy
```

Note: Installation of pip may fail in Blender 2.78a (this is a known issue)
If this happens Blender 2.79a should work. You can install it from here (and then repeat the above steps substituting 2.79a for 2.78).
```
wget https://download.blender.org/release/Blender2.79/blender-2.79a-linux-glibc219-x86_64.tar.bz2
```

##### Check the scipy installation
```
$BLENDER_PATH/blender -b -t 1 -P check_numpy.py
```
If scipy & numpy are installed correctly the the script should print "Success". You may see an error similar to:
```
ImportError: Something is wrong with the numpy installation. While importing we detected an older version of numpy
```
To fix this run the following
```
# Uninstall numpy & scipy
$BLENDER_PATH/2.78/python/bin/python3.5m -m pip uninstall scipy
$BLENDER_PATH/2.78/python/bin/python3.5m -m pip uninstall numpy

# Manually delete blender's default numpy installation
rm -rf $BLENDER_PATH/2.78/python/lib/python3.5/site-packages/numpy/

# Reinstall scipy (numpy will also be reinstalled in the process)
$BLENDER_PATH/2.78/python/bin/python3.5m -m pip install scipy

```


## 3. Custom Instructions for SocNavBench Data Generation

### Make sure your data is organized correctly

### Edit the config file
In the directory `/PATH/TO/SocNavBench/surreal/code` update the following line in the file called `"config"`
```
smpl_data_folder   = '/PATH/TO/SocNavBench/surreal/download/SURREAL/smpl_data'
```

### Test the installation
```
cd /PATH/TO/SocNavBench/surreal/code
$BLENDER_PATH/blender -b -t 1 -P export_human_meshes.py -- --idx 2 --ishape 0 --stride 59 --gender female --body_shape 1000 --outdir test_human_mesh_generation
```
The test should create the following directory structure:
```
test_human_mesh_generation/
    - velocity_0.000_m_s/
    - velocity_0.200_m_s/
    - velocity_0.500_m_s/
        - pose_2_ishape_0_stride_59/
            - body_shape_1000/
                - female/ # (here i is in [1, 2, 3])
                    - human_centering_info_i.pkl
                    - human_mesh_i.mtl
                    - human_mesh_i.obj
    - velocity_0.600_m_s/
        - pose_2_ishape_0_stride_59/
            - body_shape_1000/
                - female/  # (here i is in [4, 5, 6, 7, 8, 18, 19])
                    - human_centering_info_i.pkl 
                    - human_mesh_i.mtl
                    - human_mesh_i.obj
```
The human_mesh_i.obj (mesh of the corresponding human body), and human_centering_info_i.pkl (information to canonically center and position each human) files will be used in the SocNavBench human mesh dataset.

### Generate the Human Mesh Models for SocNavBench
Note: Full data generation takes around ~4 hours & 5 GB of space.
```
cd /PATH/TO/SocNavBench/surreal/code
sh generate_meshes.sh
```

Human meshes will be saved in `/PATH/TO/SocNavBench/surreal/code/human_meshes`.
Human textures will be saved in `/PATH/TO/SocNavBench/surreal/code/human_textures`

## Foundations
This page is built upon the [`surreal/README.md`](https://github.com/vtolani95/HumANav-Release/blob/master/surreal/README.md) in the *Human Active Navigation* ([HumANav](https://github.com/vtolani95/HumANav-Release)) codebase.
