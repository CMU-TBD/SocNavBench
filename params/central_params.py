import configparser
from dotmap import DotMap
import numpy as np
import os

# first thing to do is create a config parser
cwd = os.getcwd()

# then read in params for user editable, user-non-editable, episodes, and dataset
user_config = configparser.ConfigParser()
user_config.read(os.path.join(cwd, 'params/user_params.ini'))
# get global randomness seed
seed = user_config['socnav_params'].getint('seed')

# read params file for default configurations of SocNavBench
default_config = configparser.ConfigParser()
default_config.read(os.path.join(cwd, 'params/.default_params.ini'))

# read params file for episodes configs
episodes_config = configparser.ConfigParser()
episodes_config.read(os.path.join(cwd, 'params/episode_params_val.ini'))

# read params file for prerecorded pedestrian datasets
dataset_config = configparser.ConfigParser()
dataset_config.read(os.path.join(cwd, 'params/dataset_params.ini'))


def create_socnav_params():
    p = DotMap()
    socnav_p = user_config['socnav_params']
    p.seed = seed
    p.render_3D = (socnav_p.get('render_mode') == 'full-render')
    p.dataset_dir = socnav_p.get('dataset_dir')
    p.socnav_dir = get_path_to_socnav()
    p.traversible_dir = get_traversible_dir()
    p.sbpd_data_dir = get_sbpd_data_dir()
    if p.render_3D:
        # only need sbpd & surreal if rendering in full-render mode
        # SBPD Data Directory
        # Surreal Parameters
        surr_p = default_config['surreal_params']
        p.surreal = \
            DotMap(mode=surr_p['mode'],
                   data_dir=get_surreal_mesh_dir(),
                   texture_dir=get_surreal_texture_dir(),
                   body_shapes_train=eval(surr_p.get('body_shapes_train')),
                   body_shapes_test=eval(surr_p.get('body_shapes_test')),
                   compute_human_traversible=surr_p.getboolean(
                       'compute_human_traversible'),
                   render_humans_in_gray_only=surr_p.getboolean(
                       'render_humans_in_gray_only')
                   )
    p.camera_params = create_camera_params()
    p.building_params = create_building_params(p.render_3D)
    return p


def create_robot_params():
    p = DotMap()
    # Load the dependencies
    rob_p = user_config['robot_params']
    p.port = rob_p.getint('port')
    p.repeat_freq = rob_p.getint('repeat_freq')
    p.physical_params = \
        DotMap(radius=rob_p.getfloat('radius_cm') / 100.0,
               base=rob_p.getfloat('distance_from_ground_cm'),
               height=rob_p.getfloat('chassis_height_cm'),
               sensor_height=rob_p.getfloat('sensor_height_cm'),
               camera_elevation_degree=rob_p.getfloat(
                   'camera_elevation_degree'),
               delta_theta=rob_p.getfloat('delta_theta'))
    p.use_system_dynamics = \
        user_config['joystick_params'].getboolean('use_system_dynamics')
    return p


def create_joystick_params():
    p = DotMap()
    # get same port as the robot
    p.port = user_config['robot_params'].getint('port')
    joystick_p = user_config['joystick_params']
    p.dt = joystick_p.getfloat('dt')
    p.use_system_dynamics = joystick_p.getboolean('use_system_dynamics')
    p.use_random_planner = joystick_p.getboolean('use_random_planner')
    p.track_vel_accel = joystick_p.getboolean('track_vel_accel')
    p.print_data = joystick_p.getboolean('print_data')
    p.track_sim_states = joystick_p.getboolean('track_sim_states')
    p.write_pandas_log = joystick_p.getboolean('write_pandas_log')
    p.generate_movie = joystick_p.getboolean('generate_movie')
    return p


def create_dataset(dataset_name: str):
    p = DotMap()
    dataset_p = dataset_config[dataset_name]
    p.name = dataset_name
    p.file_name = dataset_p.get('file_name')
    p.fps = dataset_p.getint('fps')
    # p.ped_range = eval(dataset_p.get('ped_range'))
    # (starts of the datasets are located in episode_params)
    # p.start_t = eval(dataset_p.get('start_t'))
    p.spawn_delay_s = dataset_p.getfloat('spawn_delay_s')
    p.offset = eval(dataset_p.get('offset'))
    p.swapxy = dataset_p.getboolean('swapxy')
    p.flipxn = dataset_p.getboolean('flipxn')
    p.flipyn = dataset_p.getboolean('flipyn')
    return p


def create_datasets_params(datasets: list):
    # list of dotmaps for pedestrians datasets in use
    pedestrian_datasets = []
    for pdds in datasets:
        pedestrian_datasets.append(create_dataset(pdds))
    return pedestrian_datasets


def create_test_params(test: str):
    p = DotMap()
    test_p = episodes_config[test]
    p.name = test
    p.map_name = test_p.get('map_name')
    p.pedestrian_datasets = \
        create_datasets_params(eval(test_p.get('pedestrian_datasets')))
    p.datasets_start_t = eval(test_p.get('datasets_start_t'))
    p.ped_ranges = eval(test_p.get('ped_ranges'))
    p.agents_start = eval(test_p.get('agents_start'))
    p.agents_end = eval(test_p.get('agents_end'))
    p.robot_start_goal = eval(test_p.get('robot_start_goal'))
    p.max_time = test_p.getfloat('max_time')
    p.write_episode_log = test_p.getboolean('write_episode_log')
    return p


def create_episodes_params():
    p = DotMap()
    # Load the dependencies
    epi_p = episodes_config['episodes_params']
    p.without_robot = epi_p.getboolean('without_robot')
    # NOTE: uses a dictionary of DotMaps to use string notation
    tests = eval(epi_p.get('tests'))
    if len(tests) == 0:
        tests = episodes_config.sections()[1:]
    test_dict = {}
    for t in tests:
        test_dict[t] = create_test_params(test=t)
    p.tests = test_dict
    return p


def create_planner_params():
    p = DotMap()

    # Load the dependencies
    p.control_pipeline_params = create_control_pipeline_params()

    from planners.sampling_planner import SamplingPlanner
    # Default of a planner
    p.planner = SamplingPlanner
    return p


def create_waypoint_params():
    p = DotMap()
    from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
    p.grid = ProjectedImageSpaceGrid

    # Load the dependencies
    wayp_p = user_config['waypoint_params']

    p.num_waypoints = wayp_p.getint('num_waypoints')
    p.num_theta_bins = wayp_p.getint('num_theta_bins')

    p.bound_min = eval(wayp_p.get('bound_min'))
    p.bound_max = eval(wayp_p.get('bound_max'))

    camera_params = create_camera_params()
    robot_params = create_robot_params().physical_params

    # Ensure square image and aspect ratio = 1
    # as ProjectedImageSpaceGrid assumes this
    assert(camera_params.width == camera_params.height)
    assert(camera_params.fov_horizontal == camera_params.fov_vertical)

    # Additional parameters for the projected grid from the image space to the world coordinates
    p.projected_grid_params = DotMap(
        # Focal length in meters
        # OpenGL default uses the near clipping plane
        f=camera_params.z_near,

        # Half-field of view
        fov=np.deg2rad(camera_params.fov_horizontal / 2.),

        # Height of the camera from the ground in meters
        h=robot_params.sensor_height / 100.,

        # Downwards tilt of the robot camera
        tilt=np.deg2rad(-robot_params.camera_elevation_degree),
    )

    return p


def create_system_dynamics_params():
    p = DotMap()
    from systems.dubins_v2 import DubinsV2
    p.system = DubinsV2

    # Load the user editable params
    dyn_p = user_config['dynamics_params']
    # non-user editable params
    dyn_p2 = default_config['dynamics_params']

    p.dt = user_config['simulator_params'].getfloat('dt')

    p.v_bounds = eval(dyn_p.get('v_bounds'))
    assert(p.v_bounds[1] > p.v_bounds[0])
    p.w_bounds = eval(dyn_p.get('w_bounds'))
    assert(p.w_bounds[1] > p.w_bounds[0])

    p.linear_acc_max = dyn_p2.getfloat('linear_acc_max')
    p.angular_acc_max = dyn_p2.getfloat('angular_acc_max')

    p.simulation_params = \
        DotMap(simulation_mode=dyn_p2.get('simulation_mode'),
               noise_params=DotMap(is_noisy=dyn_p2.getboolean('is_noisy'),
                                   noise_type=dyn_p2.get('noise_type'),
                                   noise_lb=eval(dyn_p2.get('noise_lb')),
                                   noise_ub=eval(dyn_p2.get('noise_ub')),
                                   noise_mean=eval(dyn_p2.get('noise_mean')),
                                   noise_std=eval(dyn_p2.get('noise_std'))
                                   )
               )
    return p


def create_control_pipeline_params():
    p = DotMap()

    p.system_dynamics_params = create_system_dynamics_params()
    p.waypoint_params = create_waypoint_params()

    # Load the dependencies
    cp_p = user_config['control_pipeline_params']
    cp_p2 = default_config['control_pipeline_params']

    from control_pipelines.control_pipeline_v0 import ControlPipelineV0
    p.pipeline = ControlPipelineV0

    # The directory for saving the control pipeline files
    p.dir = os.path.join(base_data_dir(), 'control_pipelines')

    # The time interval between updates, global to system dynamics
    p.dt = p.system_dynamics_params.dt

    # Spline parameters
    from trajectory.spline.spline_3rd_order import Spline3rdOrder
    p.spline_params = \
        DotMap(spline=Spline3rdOrder,
               max_final_time=cp_p.getfloat('max_final_time'),
               epsilon=1e-5)
    p.minimum_spline_horizon = cp_p.getfloat('minimum_spline_horizon')

    # LQR setting parameters
    from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
    q_coeffs = eval(cp_p2.get('quad_coeffs'))
    l_coeffs = eval(cp_p2.get('linear_coeffs'))
    p.lqr_params = \
        DotMap(cost_fn=QuadraticRegulatorRef,
               quad_coeffs=np.array(q_coeffs, dtype=np.float32),
               linear_coeffs=np.array(l_coeffs, dtype=np.float32)
               )

    # Velocity binning parameters
    p.binning_parameters = \
        DotMap(num_bins=cp_p.getint('num_bins'),
               min_speed=p.system_dynamics_params.v_bounds[0],
               max_speed=p.system_dynamics_params.v_bounds[1])

    p.convert_K_to_world_coordinates = \
        cp_p2.getboolean('convert_K_to_world_coordinates')
    p.discard_LQR_controller_data = \
        cp_p2.getboolean('discard_LQR_controller_data')
    p.discard_precomputed_lqr_trajectories = \
        cp_p2.getboolean('discard_precomputed_lqr_trajectories')
    p.track_trajectory_acceleration = \
        cp_p2.getboolean('track_trajectory_acceleration')
    p.verbose = cp_p2.getboolean('verbose')
    return p


def create_simulator_params():
    p = DotMap()
    sim_p = user_config['simulator_params']
    p.dt = sim_p.getfloat('dt')
    p.keep_episode_running = sim_p.getboolean('keep_episode_running')
    p.block_joystick = sim_p.getboolean('block_joystick')
    p.delta_t_scale = sim_p.getfloat('delta_t_scale')
    p.socnav_params = create_socnav_params()
    p.img_scale = sim_p.getfloat('img_scale')
    p.max_frames = sim_p.getint('max_frames')
    # bound by 0 <= X <= 1
    p.fps_scale_down = max(0.0, min(1.0, sim_p.getfloat('fps_scale_down')))
    # sbpd simulator params:
    p.render_3D = p.socnav_params.render_3D
    # Load obstacle map params
    p.obstacle_map_params = create_obstacle_map_params()
    # much faster to only render the topview rather than use the 3D renderer
    from utils.utils import color_blue, color_reset
    if p.render_3D:
        print("%sRender mode: Full Render (TOPVIEW, RGB, and DEPTH)%s" %
              (color_blue, color_reset))
    else:
        print("%sRender mode: Schematic view (TOPVIEW only)%s" %
              (color_blue, color_reset))
    p.verbose_printing = sim_p.getboolean('verbose_printing')
    p.clear_files = sim_p.getboolean('clear_files')
    p.record_video = sim_p.getboolean('record_video')
    return p


def create_agent_params(with_planner=False, with_obstacle_map=False):
    p = DotMap()
    agent_p = user_config["agent_params"]
    agent_p2 = default_config["agent_params"]

    p.radius = agent_p.getfloat('radius')

    p.record_video = agent_p2.getboolean('record_video')
    p.save_trajectory_data = agent_p2.getboolean('save_trajectory_data')
    p.collision_cooldown_amnt = agent_p.getint('collision_cooldown_amnt')
    p.pause_on_collide = agent_p.getboolean('pause_on_collide')
    assert(p.collision_cooldown_amnt > 0)
    # Load system dynamics params
    p.system_dynamics_params = create_system_dynamics_params()
    if with_planner:
        p.episode_horizon_s = agent_p.getfloat('episode_horizon_s')
        p.control_horizon_s = agent_p.getfloat('control_horizon_s')

        # Load the dependencies
        p.planner_params = create_planner_params()

        # Time discretization step
        dt = p.planner_params.control_pipeline_params.system_dynamics_params.dt

        # Whether or not to track acceleration
        p.track_accel = p.planner_params.control_pipeline_params.track_trajectory_acceleration

        # Updating horizons
        p.episode_horizon = int(np.ceil(p.episode_horizon_s / dt))
        p.control_horizon = int(np.ceil(p.control_horizon_s / dt))
        p.dt = dt

    if with_obstacle_map:
        p.obstacle_map_params = create_obstacle_map_params()

    # Define the Objectives

    # Obstacle Avoidance Objective
    p.avoid_obstacle_objective = \
        DotMap(obstacle_margin0=agent_p2.getfloat('obstacle_margin0'),
               obstacle_margin1=agent_p2.getfloat('obstacle_margin1'),
               power=agent_p2.getfloat('power_obstacle'),
               obstacle_cost=agent_p2.getfloat('obstacle_cost'))
    # Angle Distance parameters
    p.goal_angle_objective = \
        DotMap(power=agent_p2.getfloat('power_angle'),
               angle_cost=agent_p2.getfloat('angle_cost'))
    # Goal Distance parameters
    p.goal_distance_objective = \
        DotMap(power=agent_p2.getfloat('power_goal'),
               goal_cost=agent_p2.getfloat('goal_cost'),
               goal_margin=agent_p2.getfloat('goal_margin'))

    # Personal Space cost parameters
    p.personal_space_objective = \
        DotMap(power=1,
               psc_scale=10
               )

    p.objective_fn_params = DotMap(obj_type=agent_p2.get('obj_type'))
    p.goal_margin = p.goal_distance_objective.goal_margin
    p.goal_dist_norm = p.goal_distance_objective.power  # Default is l2 norm
    p.episode_termination_reasons = ['Timeout',
                                     'Obstacle Collision',
                                     'Pedestrian Collision',
                                     'Success']
    p.episode_termination_colors = ['b', 'r', 'g']
    p.waypt_cmap = 'winter'
    p.num_validation_goals = agent_p2.getint('num_validation_goals')
    return p


def create_obstacle_map_params():
    p = DotMap()

    # Load the dependencies
    obst_p = default_config['obstacle_map_params']
    # p.renderer_params = create_base_params()

    from obstacles.sbpd_map import SBPDMap
    p.obstacle_map = SBPDMap

    # Size of map
    # Same as for SocNav FMM Map of Area3
    p.map_size_2 = np.array(eval(obst_p.get('map_size_2')))

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    p.dx = obst_p.getfloat('dx')

    # Origin is always 0,0 for SBPD
    p.map_origin_2 = eval(obst_p.get('map_origin_2'))

    # Threshold distance from the obstacles to sample the start and the goal positions.
    p.sampling_thres = obst_p.getint('sampling_thres')

    # Number of grid steps around the start position to use for plotting
    p.plotting_grid_steps = obst_p.getint('plotting_grid_steps')
    return p


def create_map_params():
    p = DotMap()
    # NOTE: this is very much subject to change with diff maps

    # goal_pos_n2 = np.array([[9., 15.]])
    p.goal_pos_n2 = np.array([[13.0, 8.0]])

    # pos_nk2 = np.array([[[8., 16.], [8., 12.5], [18., 16.5]]], dtype=np.float32)
    p.pos_nk2 = np.array(
        [[[8., 9.], [8., 12.5], [18., 12.5]]], dtype=np.float32)

    # p.test_goal_ang_obj_ans = [19.634956, 29.616005, 74.31618]
    p.test_goal_ang_obj_ans = [5.0869893, 18.3243617, 60.214370]

    # p.test_goal_dist_ans = [49.088074, 179.12201, 2071.5808]
    p.test_goal_dist_ans = [644.88555, 1126.650778, 1126.65109]

    # p.test_obst_map_ans = [1.252921, 1.5730935, 1.7213388]
    p.test_obst_map_ans = [100.0, 0.5900573134422302, 100.0]

    # p.test_obst = [0., 0., 0.]
    p.test_obs_obj_ans = [0., 0., 10.345789710051179]

    return p


def create_building_params(full_render: bool = False):
    p = DotMap()
    # Load the dependencies
    build_p = user_config['building_params']
    build_p2 = default_config['building_params']
    p.building_name = build_p.get('building_name')
    p.building_thresh = build_p.getint('building_thresh')
    p.dataset_name = build_p2.get('dataset_name')
    if full_render:
        # always load meshes if running a full-render
        p.load_meshes = True
    else:
        p.load_meshes = build_p2.getboolean('load_meshes')
    p.load_traversible_from_pickle_file = \
        build_p2.getboolean('load_traversible')
    p.flip = False
    return p


def create_camera_params():
    p = DotMap()
    camera_p = user_config['camera_params']
    p.modalities = eval(camera_p.get('modalities'))
    p.width = camera_p.getfloat('width')
    p.height = camera_p.getfloat('height')
    p.z_near = camera_p.getfloat('z_near')
    p.z_far = camera_p.getfloat('z_far')
    p.fov_horizontal = camera_p.getfloat('fov_horizontal')
    p.fov_vertical = camera_p.getfloat('fov_vertical')
    p.img_channels = camera_p.getint('img_channels')
    p.im_resize = camera_p.getfloat('im_resize')
    p.max_depth_meters = camera_p.getfloat('max_depth_meters')
    return p


"""BEGIN DIRECTORY FUNCTIONS"""


def get_path_to_socnav():
    # can use a literal string in params.ini as the path
    # PATH_TO_HUMANAV = config['base_params']['base_directory']
    # or just use the relative path
    PATH_TO_HUMANAV = os.getcwd()
    if(not os.path.exists(PATH_TO_HUMANAV)):
        # the main directory should be the parent of params/central_params.py
        PATH_TO_HUMANAV = os.path.join(os.path.dirname(__file__), '..')
        if(os.path.exists(PATH_TO_HUMANAV)):
            return PATH_TO_HUMANAV
        print('\033[31m', "ERROR: Failed to find SocNavBench installation at",
              PATH_TO_HUMANAV, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_HUMANAV


def base_data_dir():
    PATH_TO_BASE_DIR = os.path.join(get_path_to_socnav(), 'wayptnav_data')
    if(not os.path.exists(PATH_TO_BASE_DIR)):
        print('\033[31m', "ERROR: Failed to find the wayptnav_data dir at",
              PATH_TO_BASE_DIR, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_BASE_DIR


def get_sbpd_data_dir():
    PATH_TO_SBPD = os.path.join(
        get_path_to_socnav(), 'sd3dis/stanford_building_parser_dataset')
    if(not os.path.exists(PATH_TO_SBPD)):
        print('\033[31m', "ERROR: Failed to find sd3dis installation at",
              PATH_TO_SBPD, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SBPD


def get_traversible_dir():
    PATH_TO_TRAVERSIBLES = os.path.join(get_sbpd_data_dir(), 'traversibles')
    if(not os.path.exists(PATH_TO_TRAVERSIBLES)):
        print('\033[31m', "ERROR: Failed to find traversible directory at",
              PATH_TO_TRAVERSIBLES, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_TRAVERSIBLES


def get_surreal_mesh_dir():
    PATH_TO_SURREAL_MESH = os.path.join(
        get_path_to_socnav(), 'surreal/code/human_meshes')
    if(not os.path.exists(PATH_TO_SURREAL_MESH)):
        print('\033[31m', "ERROR: Failed to find SURREAL meshes at",
              PATH_TO_SURREAL_MESH, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SURREAL_MESH


def get_surreal_texture_dir():
    PATH_TO_SURREAL_TEXTURES = os.path.join(
        get_path_to_socnav(), 'surreal/code/human_textures')
    if(not os.path.exists(PATH_TO_SURREAL_TEXTURES)):
        print('\033[31m', "ERROR: Failed to find SURREAL textures at",
              PATH_TO_SURREAL_TEXTURES, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SURREAL_TEXTURES


"""BEGIN OTHER UTILS"""


def get_seed():
    return seed
