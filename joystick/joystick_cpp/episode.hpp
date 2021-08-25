#ifndef EPISODE_H
#define EPISODE_H

#include "agents.hpp"
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

/**
 * @brief Container for the "environment" which holds the traversibles along with
 * the map scale.
 * */
struct env_t
{
    float dx_scale;
    vector<float> room_center;
    vector<vector<int>> building_traversible;
    vector<vector<int>> human_traversible;
    env_t()
    {
        dx_scale = 0;
        building_traversible = {};
        human_traversible = {};
    }
    /**
     * @brief Update the fields in the environment
     * */
    void update_environment(vector<vector<int>> &building_trav, vector<vector<int>> &human_trav, float scale)
    {
        dx_scale = scale;
        building_traversible = building_trav;
        human_traversible = human_trav;
    }
};

class Episode
{
  public:
    Episode()
    {
    }

    /** @brief Constructor for Episode instance
     * @param[in] name The title of the episode
     * @param[in] building_trav The traversible of the building in the environment
     * @param[in] human_trav The traversible of the humans in the environment
     * @param[in] scale The map scale of the environment
     * @param[in] peds The agents (pedestrians) in the environment
     * @param[in] t_budget The maximum time allocated for this episode test
     * @param[in] r_start The starting config of the robot
     * @param[in] r_goal The goal config of the robot
     * */
    Episode(string &name, vector<vector<int>> &building_trav, vector<vector<int>> &human_trav, float scale,
            unordered_map<string, AgentState> &a, float t_budget, vector<float> &r_start, vector<float> &r_goal)
    {
        title = name;
        env.update_environment(building_trav, human_trav, scale);
        agents = a;
        max_time_s = t_budget;
        robot_start = r_start;
        robot_goal = r_goal;
    }
    /*** @brief getter for the title of the episode*/
    string get_title() const
    {
        return title;
    }

    /**
     * @brief getter for the robot's start config
     * @returns vector<float> Config containing (x, y, theta)
     */
    vector<float> get_robot_start() const
    {
        return robot_start;
    }

    /**
     * @brief getter for the robot's goal config
     * @returns vector<float> Config containing (x, y, theta)
     */
    vector<float> get_robot_goal() const
    {
        return robot_goal;
    }

    /**
     * @brief getter for an agent map (only pedestrians)
     * @returns An unordered_map mapping pedestrian names to their AgentStates
     */
    unordered_map<string, AgentState> get_agents() const
    {
        return agents;
    }

    /*** @brief getter for the maximum time allowed for this episode*/
    const float get_time_budget() const
    {
        return max_time_s;
    }

    /*** @brief getter for the environment data
     * @returns env_t A struct containing the building & human traversible,
     * room scale, and room center. */
    env_t get_environment() const
    {
        return env;
    }

    /**
     * @brief Construct en Episode instance from its serialized json representation
     *
     * @param[in] metadata (const &json) : the json data to take in
     * @returns Episode Instance with the corresponding fields
     * */
    static Episode construct_from_json(const json &metadata)
    {
        // gather data from json
        string title = metadata["episode_name"];

        // create a json object of the environment metadata
        auto &env_json = metadata["environment"];

        // gather the map (building) traversible
        vector<vector<int>> map_trav = env_json["map_traversible"];

        // gather the human traversible (not currently supported)
        vector<vector<int>> h_trav = {};

        // default for sd3dis data (5cm per pixel)
        float dx_m = 0.05;
        // TODO: fix map_scale not parsing through json correctly
        // float dx_m = env["map_scale"];

        // pedestrian data as a map between name and agent
        unordered_map<string, AgentState> agents = AgentState::construct_from_dict(metadata["pedestrians"]);

        // maximum time budgeted for this episode
        float max_time = metadata["episode_max_time"];

        // NOTE there is an assumption that there is only one robot in the
        // simulator at once, and its *name* is "robot_agent"
        auto &robots = metadata["robots"];
        auto &robot = robots["robot_agent"];

        // robot config as a 3D vector of (x, y, theta)
        vector<float> r_start = robot["start_config"];
        vector<float> r_goal = robot["goal_config"];

        return Episode(title, map_trav, h_trav, dx_m, agents, max_time, r_start, r_goal);
    }

    /**
     * @brief Prints quick information about the current episode.
     * Useful for debugging
     */
    void print() const
    {
        cout << "Episode: " << get_title() << endl;
        cout << "Max time: " << get_time_budget() << endl;
        float start_x = get_robot_start()[0];
        float start_y = get_robot_start()[1];
        float start_theta = get_robot_start()[2];
        cout << "Robot start: " << start_x << ", " << start_y << ", " << start_theta << endl;
        float goal_x = get_robot_goal()[0];
        float goal_y = get_robot_goal()[1];
        float goal_theta = get_robot_goal()[2];
        cout << "Robot goal: " << goal_x << ", " << goal_y << ", " << goal_theta << endl;
    }

  private:
    /* name of the episode, eg: "test_socnav" */
    string title;
    /* environment struct holding the human/building traversible & map metadata */
    env_t env;
    /*  dictionary with <key, val> pair being agent name (str) & agent (AgentState) */
    unordered_map<string, AgentState> agents;
    /* maximum time budgeted for this episode in seconds */
    float max_time_s;
    /* pos3's for the robot configs (x, y, theta) */
    vector<float> robot_start, robot_goal;
};

#endif