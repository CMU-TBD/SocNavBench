import configparser
from dotmap import DotMap
import numpy as np
import os
import pickle
import sys
socnavdir = os.path.abspath("..")
sys.path = [k for k in sys.path if 'carla' not in k]
sys.path.append(socnavdir)
# print(sys.path)
from obstacles.sbpd_map import SBPDMap


def create_test_params(test: str):
    p = DotMap()
    test_p = episodes_config[test]
    p.name = test
    p.map_name = test_p.get('map_name')
    p.pedestrian_datasets = test_p.get('pedestrian_datasets')
    p.datasets_start_t = eval(test_p.get('datasets_start_t'))
    p.ped_ranges = eval(test_p.get('ped_ranges'))
    p.agents_start = eval(test_p.get('agents_start'))
    p.agents_end = eval(test_p.get('agents_end'))
    p.robot_start_goal = eval(test_p.get('robot_start_goal'))
    p.max_time = test_p.getfloat('max_time')
    p.write_episode_log = test_p.getboolean('write_episode_log')
    return p


def get_config(socnavdir, map_name):
    """
    Return the resolution and traversible of the SBPD building
    """
    traversible_dir = os.path.join(socnavdir, 'sd3dis', 'stanford_building_parser_dataset', 'traversibles')
    traversible_dir = os.path.join(traversible_dir,
                                   map_name)
    filename = os.path.join(traversible_dir, 'data.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    resolution = data['resolution']
    traversible = data['traversible']
    return resolution, traversible


if __name__ == "__main__":
    # get global randomness seed for replicable random episodes
    seed = 999; np.random.seed(seed)
    num_episodes_sample = 10  # number of random episodes to sample

    # first thing to do is create a config parser
    cwd = os.getcwd()
    # read params file for episodes configs
    episodes_config = configparser.ConfigParser()
    episodes_config.read(os.path.join(cwd, '../params/episode_params_val.ini'))

    # empty config parser for output
    random_episodes_config = configparser.ConfigParser()
    random_episodes_config['episodes_params'] = {'tests': '[]', 'without_robot':'False'}

    # load the set of hand designed episodes
    p = DotMap()
    epi_p = episodes_config['episodes_params']
    p.without_robot = epi_p.getboolean('without_robot')
    # NOTE: uses a dictionary of DotMaps to use string notation
    tests = eval(epi_p.get('tests'))
    assert(len(tests) > 0)

    # construct N sampled eps
    for i in range(num_episodes_sample):
        # choose among the loaded eps
        sampled_test = np.random.choice(tests)
        sampled_test_dm = create_test_params(test=sampled_test)
        print(sampled_test)
        print(sampled_test_dm)
        print(episodes_config[sampled_test].get('pedestrian_datasets'))
        # get the corresponding map
        map_name = sampled_test_dm.map_name
        # load traversible from map
        resolution, traversible = get_config(socnavdir, map_name)
        obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                     map_origin_2=[0., 0.],
                                     sampling_thres=2,
                                     plotting_grid_steps=100)
        obstacle_map = SBPDMap(obstacle_map_params,
                               renderer=0, res=resolution,
                               map_trav=traversible)
        # sample freespace start and goal from map
        sample_start = np.squeeze(obstacle_map.sample_point_112(np.random))
        sample_start_orientation = 2 * np.pi * np.random.random()
        sample_start_list = [sample_start[0], sample_start[1], sample_start_orientation]

        sample_goal = np.squeeze(obstacle_map.sample_point_112(np.random))
        sample_goal_orientation = 2 * np.pi * np.random.random()
        sample_goal_list = [sample_goal[0], sample_goal[1], sample_goal_orientation]

        # if you want to visualize the start and goal
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        obstacle_map.render(ax)
        ax.plot(sample_start[0], sample_start[1], 'r.')
        ax.plot(sample_goal[0], sample_goal[1], 'r*')
        plt.show(); break

        new_test_name = sampled_test + str(i)
        random_episodes_config[new_test_name] = episodes_config[sampled_test]
        random_episodes_config[new_test_name]['robot_start_goal'] = \
            str([sample_start_list, sample_goal_list])

    # format and print generated episode config
    randomeps_filename = 'randomeps_N' + str(num_episodes_sample) + '_R' + str(seed) + '.cfg'
    with open(randomeps_filename, 'w') as configfile:
        random_episodes_config.write(configfile)

    exit()
