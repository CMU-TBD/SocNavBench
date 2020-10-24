#### The following script will save meshes in .obj format of a human
#### with a specified body shape and gender. Human poses come from
#### mocap data from real human subjects. This script is heavily
#### based on code from the SURREAL dataset: https://github.com/gulvarol/surreal
### This particular file is based on main_part1.py
#### https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py


import sys
import os
import random
import math
import bpy
import numpy as np
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam
import pickle

sys.path.insert(0, ".")

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None

sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

# Map from the heuristically computed velocity to
# a more visually realistic velocity (i.e. so the spread in the
# humans legs is roughly correlated with its speed)

    # 0.0 m/s -> 0.0 m/s (bin 0 -> bin 0)
    # .2 & .4 m/s -> .2 m/s (bins 1, 2 -> bin 1)
    # .6 m/s -> .5 m/s (bin 3 -> bin 2)
    # .8 m/s -> .6 m/s (bin 4 -> bin 3)
    # > .8 m/s -> N/A (We only keep speeds up to .6 m/s (after rebinning))

rebin_map = {0: 0, 1: 1, 2: 1, 3: 2, 4:3}

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def init_scene(scene, params, gender='female'):
    # load fbx model
    bpy.ops.import_scene.fbx(filepath=join(params['smpl_data_folder'], 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
                             axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '%s_avg' % gender[0] 
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob

    cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
                                 (0., -1, 0., -1.0),
                                 (-1., 0., 0., 0.),
                                 (0.0, 0.0, 0.0, 1.0)))

    # The following camera settings are irrelevant
    # as we are saving actual meshes (not images)
    # 90 FOV, z_near = .01, focal length = .01
    # (or image plane =.02x.02 [2*.01 since 45, 45, 90 triangle])
    cam_ob.rotation_mode = 'XYZ'
    cam_ob.rotation_euler[2] = math.radians(-30)
    cam_ob.data.angle = math.radians(90)
    cam_ob.data.clip_start = .01
    cam_ob.data.sensor_height = 20
    cam_ob.data.sensor_width = 20

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_material_index  = True

    # set render size
    scn.render.resolution_x = params['resy']
    scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    return(ob, obname, arm_ob, cam_ob)

# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    if frame is not None:
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

def get_bone_locs(obname, arm_ob, scene, cam_ob):
    n_bones = 24
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))
    bone_locations_2d = np.empty((n_bones, 2))
    bone_locations_3d = np.empty((n_bones, 3), dtype='float32')

    # obtain the coordinates of each bone head in image space
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.head)
        co_3d = arm_ob.matrix_world * bone.head
        bone_locations_3d[ibone] = (co_3d.x,
                                 co_3d.y,
                                 co_3d.z)
        bone_locations_2d[ibone] = (round(co_2d.x * render_size[0]),
                                 round(co_2d.y * render_size[1]))
    return(bone_locations_2d, bone_locations_3d)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, cam_ob)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)

# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)

    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))
    
    name = sorted(cmu_keys)[idx % len(cmu_keys)]
    
    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms, fshapes, name)

import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

def main():
    # time logging
    global start_time
    start_time = time.time()

    import argparse
    
    # parse commandline arguments
    #log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')
    parser.add_argument('--gender', type=str,
                        help='gender {male, female}')
    parser.add_argument('--body_shape_idx', type=int,
                        help='body shape idx (height, weight etc.) < 1682 for female, < 1360 for male)')
    parser.add_argument('--outdir', type=str, help='out directory')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
   
    idx = args.idx
    ishape = args.ishape
    stride = args.stride
    gender = args.gender
    body_shape_idx = args.body_shape_idx
    outdir = args.outdir

    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)
    log_message("gender: %s" % gender)
    log_message("body_shape_idx: %d" % body_shape_idx)

    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50

    # Custom checks for specially added fields (gender & body_shape_idx)
    if gender is 'male':
        assert body_shape_idx < 1360
    elif gender is 'female':
        assert body_shape_idx < 1682
    else:
        assert(gender in ['male', 'female'])
    
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    (runpass, idx) = divmod(idx, len(idx_info))
    
    log_message("runpass: %d" % runpass)
    log_message("output idx: %d" % idx)
    idx_info = idx_info[idx]
    log_message("sequence: %s" % idx_info['name'])
    log_message("nb_frames: %f" % idx_info['nb_frames'])
    log_message("use_split: %s" % idx_info['use_split'])

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']

    # compute number of cuts
    nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
    log_message("Max ishape: %d" % (nb_ishape - 1))

    if ishape == None:
        exit(1)

    assert(ishape < nb_ishape)

    # name is set given idx
    name = idx_info['name']
    output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    params['output_path'] = output_path
    tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    params['tmp_path'] = tmp_path

    # check if already computed
    #  + clean up existing tmp folders if any
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)

    # >> don't use random generator before this point <<

    # initialize RNG with seeds from sequence id
    import hashlib
    s = "synth_data:%d:%d:%d" % (idx, runpass,ishape)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)
    
    if(output_types['vblur']):
        vblur_factor = np.random.normal(0.5, 0.5)
        params['vblur_factor'] = vblur_factor
    
    log_message("Setup Blender")

    genders = {0: 'female', 1: 'male'}

    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    # grab clothing names
    log_message("clothing: %s" % clothing_option)
    with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % ( gender, idx_info['use_split'] ) ) ) as f:
        txt_paths = f.read().splitlines()

    # if using only one source of clothing
    if clothing_option == 'nongrey':
        txt_paths = [k for k in txt_paths if 'nongrey' in k]
    elif clothing_option == 'grey':
        txt_paths = [k for k in txt_paths if 'nongrey' not in k]

    # random clothing texture
    cloth_img_name = choice(txt_paths)
    cloth_img_name = join(smpl_data_folder, cloth_img_name)

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

    log_message("Initializing scene")
    camera_distance = 4.0  # Not rendering images so camera distance can be any #
    params['camera_distance'] = camera_distance
    ob, obname, arm_ob, cam_ob = init_scene(scene, params, gender)

    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob

    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
    orig_cam_loc = cam_ob.location.copy()

    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)
    
    log_message("Loaded body data for %s" % name)
    
    nb_fshapes = len(fshapes)
    
    # Force the train split
    fshapes = fshapes[:int(nb_fshapes*0.8)]
    
    shape = fshapes[body_shape_idx]
    
    # example shapes
    #shape = np.zeros(10) #average
    #shape = np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]) #fat
    #shape = np.array([-2.26781107,  0.88158132, -0.93788176, -0.23480508,  1.17088298,  1.55550789,  0.44383225,  0.37688275, -0.27983086,  1.77102953]) #thin
    #shape = np.array([ 0.00404852,  0.8084637 ,  0.32332591, -1.33163664,  1.05008727,  1.60955275,  0.22372946, -0.10738459,  0.89456312, -1.22231216]) #short
    #shape = np.array([ 3.63453289,  1.20836171,  3.15674431, -0.78646793, -1.93847355, -0.32129994, -0.97771656,  0.94531640,  0.52825811, -0.99324327]) #tall

    ndofs = 10

    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

    data = cmu_parms[name]
    
    fbegin = ishape*stepsize*stride
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    
    log_message("Computing how many frames to allocate")
    N = len(data['poses'][fbegin:fend:stepsize])
    log_message("Allocating %d frames in mat file" % N)

    # force recomputation of joint angles unless shape is all zeros
    curr_shape = np.zeros_like(shape)

    matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
    log_message('Working on %s' % matfile_info)

    
    # allocate
    dict_info = {}
    dict_info['bg'] = np.zeros((N,), dtype=np.object) # background image path
    dict_info['camLoc'] = np.empty(3) # (1, 3)
    dict_info['clipNo'] = ishape +1
    dict_info['cloth'] = np.zeros((N,), dtype=np.object) # clothing texture image path
    dict_info['gender'] = np.empty(N, dtype='uint8') # 0 for male, 1 for female
    dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32') # 2D joint positions in pixel space
    dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32') # 3D joint positions in world coordinates
    dict_info['light'] = np.empty((9, N), dtype='float32')
    dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32') # joint angles from SMPL (CMU)
    dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
    dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
    dict_info['zrot'] = np.empty(N, dtype='float32')
    dict_info['camDist'] = camera_distance
    dict_info['stride'] = stride

    # Note: Necessary for the HumANav dataset to canonically center the human.
    dict_info['rightFootPos'] = np.empty((3, N), dtype='float32')
    dict_info['rightToePos'] = np.empty((3, N), dtype='float32')
    dict_info['leftFootPos'] = np.empty((3, N), dtype='float32')
    dict_info['leftToePos'] = np.empty((3, N), dtype='float32')


    if name.replace(" ", "").startswith('h36m'):
        dict_info['source'] = 'h36m'
    else:
        dict_info['source'] = 'cmu'

    if(output_types['vblur']):
        dict_info['vblur_factor'] = np.empty(N, dtype='float32')

    # for each clipsize'th frame in the sequence
    get_real_frame = lambda ifr: ifr
    random_zrot = 0
    reset_loc = False
    curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                       cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])
    random_zrot = 2*np.pi*np.random.rand()
    
    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    # Needed to extracting toe and foot position & direction
    with open('pkl/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)

    # Where the meshes and centering information is stored
    base_dir = outdir
    vs = np.arange(0, 1.85, .2)

    # VS corresponds to the approximate velocity of the human (computer heuristically from mocap data), but after examining
    # the actual data we decided to rebin the estimated human velocity for more realistic visual cues.
    rebinned_vs = np.array([0., .2, .5, .6])
    velocity_folders = make_velocity_dirs(base_dir, rebinned_vs)
    pose_ishape_stride_str = 'pose_{:d}_ishape_{:d}_stride_{:d}'.format(idx, ishape, stride)
    body_shape_str = 'body_shape_{:d}'.format(body_shape_idx)
    gender_str = gender

    # create a keyframe animation with pose, translation, blendshapes and camera motion
    # LOOP TO CREATE 3D ANIMATION
    dt = 1./30.
    prev_human_pos_3 = None
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))

        dict_info['shape'][:, iframe] = shape[:ndofs]
        dict_info['pose'][:, iframe] = pose
        dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
        if(output_types['vblur']):
            dict_info['vblur_factor'][iframe] = vblur_factor

        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        dict_info['zrot'][iframe] = random_zrot

        scene.update()

        # Bodies centered only in each minibatch of clipsize frames
        if seq_frame == 0 or reset_loc: 
            reset_loc = False
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            cam_ob.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
            cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))
            dict_info['camLoc'] = np.array(cam_ob.location)

        # Compute the human (x, y, theta) position based on the location and orientation
        # of its feet
        human_pos_3 = compute_human_pos_3(scene, vsegm, ob)
        human_speed = compute_human_speed(trans, prev_human_pos_3, dt=dt)
        prev_human_pos_3 = trans
        centering_data = {'human_pos_3': human_pos_3}

        # If we can't estimate the speed from finite differencing (i.e. at the first timestep)
        # do not save the mesh
        if human_speed is not None:
            print(human_speed)
            # Compute the closest velocity bin and the corresponding folder name
            # it will be None if the human_speed is not in the range [vmin, vmax]
            velocity_folder = compute_velocity_folder(human_speed, vs, rebinned_vs, velocity_folders, vmax=1.85, vmin=0.0)

            if velocity_folder is not None:
                human_data_output_folder = os.path.join(velocity_folder,
                                                        pose_ishape_stride_str,
                                                        body_shape_str, gender_str)
                if not os.path.exists(human_data_output_folder):
                    os.makedirs(human_data_output_folder)

                centeringFile = os.path.join(human_data_output_folder, 'human_centering_info_{:d}.pkl'.format(seq_frame))
                with open(centeringFile, 'wb') as f:
                    pickle.dump(centering_data, f)

                # Exporting Human Mesh Here for Use in HumANav
                meshFile = os.path.join(human_data_output_folder, 'human_mesh_{:d}.obj'.format(seq_frame))
                bpy.ops.export_scene.obj(filepath=meshFile,
                                         keep_vertex_order=True, group_by_object=True)
    os._exit(0)

def compute_velocity_folder(human_speed, vs, rebinned_vs, rebinned_velocity_folders, vmax, vmin):
    if human_speed > vmax or human_speed < vmin:
        return None
    abs_diff = np.abs(vs-human_speed)
    min_idx = np.argmin(abs_diff)

    
    try:
        rebinned_min_idx = rebin_map[min_idx]
    except KeyError:
        # Ignore all heuristically computed velocities above .8 m/s
        return None

    return rebinned_velocity_folders[rebinned_min_idx]

def make_velocity_dirs(base_dir, vs):
    velocity_folders = []
    for v in vs:
        velocity_folder = os.path.join(base_dir, 'velocity_{:.3f}_m_s'.format(v))
        velocity_folders.append(velocity_folder)
        if not os.path.exists(velocity_folder):
            os.makedirs(velocity_folder)
    return velocity_folders

def compute_human_speed(human_pos_3, prev_human_pos_3, dt):
    if prev_human_pos_3 is None:
        return None
    else:
        xy_diff_2 = (human_pos_3-prev_human_pos_3)[:2]
        return np.linalg.norm(xy_diff_2)/dt

def compute_human_pos_3(scene, vsegm, ob):
    """
    Convert the human to a mesh and compute the
    (x, y, theta) location of the human on the ground
    plane as the midpoint between the feet facing the
    mean direction between the two direction of the two
    feet.

    This information can be used to canonically center the human
    (i.e. in the HumANav dataset)
    """
    me = ob.to_mesh(scene, True, 'PREVIEW')
    vrtxs = np.array(me.vertices.items())
    coors = np.array([vrtx.co for vrtx in vrtxs[:, 1]])

    # Compute Right Foot and Toe Center
    rightFootCoors = coors[vsegm['rightFoot']]
    rightToeCoors = coors[vsegm['rightToeBase']]
    rightFootCenter = np.mean(rightFootCoors, axis=0)
    rightToeCenter = np.mean(rightToeCoors, axis=0)

    # Compute Left Foot and Toe Center
    leftFootCoors = coors[vsegm['leftFoot']]
    leftToeCoors = coors[vsegm['leftToeBase']]
    leftFootCenter = np.mean(leftFootCoors, axis=0)
    leftToeCenter = np.mean(leftToeCoors, axis=0)

    # Compute the human center
    rightMidpoint = (rightFootCenter + rightToeCenter)/2.
    leftMidpoint = (leftFootCenter + leftToeCenter)/2.
    humanCenter = (rightMidpoint + leftMidpoint)/2.

    rightFootCoors = np.stack([rightFootCenter, rightToeCenter], axis=1)
    leftFootCoors = np.stack([leftFootCenter, leftToeCenter], axis=1)

    # Compute the human direction
    rightFootDir = rightFootCoors[:, 1] - rightFootCoors[:, 0]
    leftFootDir = leftFootCoors[:, 1] - leftFootCoors[:, 0]
    humanDir = (rightFootDir + leftFootDir)/2.
    theta = np.arctan2(humanDir[1], humanDir[0])
    human_pos_3 = np.array([humanCenter[0], humanCenter[1], theta])

    return human_pos_3


if __name__ == '__main__':
    main()
