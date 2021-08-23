#ifndef AGENTS_H
#define AGENTS_H

#include <string>
#include <vector>
// external json parser (https://github.com/nlohmann/json)
#include "json.hpp"

using namespace std;
using json = nlohmann::json; // for convenience

/**
 * @brief State class for the Pedestrians representation.
 * @param[out] name The agent's name
 * @param[out] current_config The agent's current "config" (x, y, theta)
 * @param[out] radius The radius of the agent
 */
class AgentState
{
  public:
    AgentState()
    {
    }

    /**
     * @brief Constructor for the Pedestrian's AgentState representation.
     * @param[out] name The agent's name
     * @param[out] current_config The agent's current "config" (x, y, theta)
     * @param[out] radius The radius of the agent
     */
    AgentState(string &n, vector<float> &current_pos3, float r)
    {
        name = n;
        current_config = current_pos3;
        radius = r;
    }

    /** @brief getter for the agent's name */
    string get_name() const
    {
        return name;
    }

    /** @brief getter for the agent's radius */
    float get_radius() const
    {
        return radius;
    }

    /* @brief getter for the agent's current config (x, y, theta) */
    vector<float> get_current_config() const
    {
        return current_config;
    }

    /**
     * @brief construct an AgentState out of its serialized json form
     * @param[in] data The json object that holds the serialized AgentState info
     * @returns AgentState instance containing all the fields from the json
     **/
    static AgentState construct_from_json(const json &data)
    {
        string name = data["name"];
        vector<float> current_pos3 = data["current_config"];
        float radius = data["radius"];
        return AgentState(name, current_pos3, radius);
    }

    /**
     * @brief construct a hash map out of a serialized json dictionary
     * @param[in] data The json object that holds the serialized AgentState dictionary
     * @returns pedestrians The map from name to AgentState of all the pedestrians
     **/
    static unordered_map<string, AgentState> construct_from_dict(const json &data)
    {
        unordered_map<string, AgentState> pedestrians;
        for (auto &ped : data.items())
        {
            string name = ped.key(); // indexed by name (string)
            auto new_agent = AgentState::construct_from_json(ped.value());
            pedestrians.insert({name, new_agent});
        }
        return pedestrians;
    }

  private:
    /* The agent's name */
    string name;

    /* The current config of the agent (x, y, theta) */
    vector<float> current_config;

    /* The radius of the agent */
    float radius;
};

#endif
