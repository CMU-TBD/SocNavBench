#ifndef SIMSTATE_H
#define SIMSTATE_H

#include "agents.hpp"
#include <unordered_map>

/**
 * @brief A snapshot of the simulator at a particular time
 * @param[out] robot The robot's AgentState in the simulator
 * @param[out] robot_on The robot's status
 * @param[out] simulator_time The time of capture for this SimState
 * @param[out] pedestrians All the agents in the simulator in a dictionary
 * @param[out] termination_cause The reason why the robot terminated if applicable
 */
class SimState
{
  public:
    SimState()
    {
    }
    /** @brief Constructor for SimState instances
     * @param[in] robot_agent The AgentState of the robot in the simulator
     * @param[in] rob_on The status of the robot (on or off)
     * @param[in] sim_time The time in the simulator when this SimState occured
     * @param[in] peds The map of pedestrians (name -> AgentState) in the simulator
     * @param[in] term_cause The termination cause of the robot if applicable
     * */
    SimState(AgentState &r, bool rob_on, float sim_time, unordered_map<string, AgentState> &peds, string &term_cause)
    {
        robot = r;
        robot_on = rob_on;
        simulator_time = sim_time;
        pedestrians = peds;
        termination_cause = term_cause;
    }

    /* @brief getter for the robot AgentState instance in this SimState */
    const AgentState get_robot() const
    {
        return robot;
    }

    /* @brief getter for the power-status of the robot, powered on or off */
    const bool get_robot_status() const
    {
        return robot_on;
    }

    /* @brief getter for the simulator time for when this SimState was captured */
    const float get_sim_t() const
    {
        return simulator_time;
    }

    /* @brief getter for the Robot's termination cause if applicable */
    const string get_termination_cause() const
    {
        return termination_cause;
    }

    /* @brief getter for the SimState's pedestrian map (name -> AgentState) */
    const unordered_map<string, AgentState> get_pedestrians() const
    {
        return pedestrians;
    }

    /**
     * @brief Constructs a SimState instance from a json serialization
     * @param[in] json object to deserialize (parse) and interpret
     * @returns SimState with all the corresponding fields from the json object
     */
    static SimState construct_from_json(const json &json_data)
    {
        // first and foremost, every python sim_state instance has `robot_on`
        // and `sim_t` representing the robot's status and capture time respectively
        bool rob_on = json_data["robot_on"];
        float sim_t = json_data["sim_t"];

        // however, some variables may or may not be included:

        // Only included if the robot has terminated
        string term_cause;
        // Only included if the robot is still running
        AgentState rob;
        // Only included if the robot is still running
        unordered_map<string, AgentState> peds;

        // case on robot's status
        if (rob_on)
        {
            // currently only one robot exists (and its name is "robot_agent")
            peds = AgentState::construct_from_dict(json_data["pedestrians"]);
            auto &sim_robots = json_data["robots"];
            auto &robot_json = sim_robots["robot_agent"];
            rob = AgentState::construct_from_json(robot_json);
        }
        else
            term_cause = json_data["termination_cause"];
        return SimState(rob, rob_on, sim_t, peds, term_cause);
    }

  private:
    /* Robot's AgentState (currently only one robot is supported) */
    AgentState robot;
    /* Status of the robot, powered on (true) or off (false) */
    bool robot_on;
    /* Time of capture of this particular SimState instance */
    float simulator_time;
    /* Mapping from pedestrian's name to their AgentState class */
    unordered_map<string, AgentState> pedestrians;
    /* Reason for termination if applicable: "Success", "Collision", "Timeout" */
    string termination_cause;
};

#endif
