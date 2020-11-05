import os
import pickle as pkl
from collections import defaultdict
from simulators.simulator import Simulator


def collate_episode_scores(central_sim: Simulator, params=None):
    """
    Takes in the outputs of generate_episode_score_report
    And digests them into overall scores

    :return:
    """

    p = params

    # read in and collect tuples of (test name, score_dict)
    scores_list_success = []
    scores_list_fail = []

    for i, episode_name in enumerate(list(p.episode_params.tests.keys())):
        episode = p.episode_params.tests[episode_name]
        score_filename = os.path.join(p.output_directory,
                                      "episode_score_%s.pkl" % episode_name)
        with open(score_filename, 'rb') as f:
            score_dict = pkl.load(f)
            if score_dict["success"]:
                scores_list_success.append((episode_name, score_dict))
            else:
                scores_list_fail.append((episode_name, score_dict))

    agg_score_dict_succ = defaultdict(list)
    agg_score_dict_fail = defaultdict(list)
    # do something for all the successful episodes

    # do something for all scores that are a single value
    for test_name, score_dict in scores_list_success:
        # go through the scores for each episode and aggregate
        for key in score_dict.keys():
            score = score_dict[key]
            if isinstance(score, float):
                agg_score_dict_succ[key] += [score]

    # do something for all the things that are a distribution/collection of values
    for test_name, score_dict in scores_list_fail:
        # go through the scores for each episode and aggregate
        for key in score_dict.keys():
            score = score_dict[key]
            if isinstance(score, float):
                agg_score_dict_fail[key] += [score]

    # do somethings differently for the unsuccessful episodes
    # at min, count the different failure reasons
    #

    scores = []
    return scores
