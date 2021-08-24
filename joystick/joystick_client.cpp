#include <cstdlib> // rand()
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
// external json parser (https://github.com/nlohmann/json)
#include "joystick_cpp/agents.hpp"
#include "joystick_cpp/episode.hpp"
#include "joystick_cpp/json.hpp"
#include "joystick_cpp/sim_state.hpp"
#include "joystick_cpp/sockets.hpp"

using namespace std;
using json = nlohmann::json; // for convenience

/**
 * @brief wrapper of listen_once that parses the data into episode names
 * @param[in] episodes The vector of episode names to return
 */
void get_all_episode_names(vector<string> &episodes);

/**
 * @brief wrapper of listen_once that parses the data of an episode's metadata
 * @param[in] ep_name The name of the episode that will be created
 * @param[out] ep The episode
 */
void get_episode_metadata(const string &title, Episode &ep);

/**
 * @brief run the sense() plan() act() loop of the joystick controller
 * @param[in] ep The episode metadata in the form of an Episode class
 */
void update_loop(const Episode &ep);

/**
 * @brief main process, runs through the handshake and episode loops
 */
int main(int argc, char *argv[])
{
    srand(1); // seed random number generator
    cout << "Demo Joystick Interface in C++ (Random planner)" << endl;
    /// TODO: add suport for reading .ini param files from C++
    cout << "Initiated joystick locally (AF_UNIX) at \"" << SEND_ID << " & \"" << RECV_ID << "\"" << endl;
    // establish socket that sends data to robot
    if (init_send_conn(sender_addr, sender_fd) < 0)
        return -1;
    // establish socket that receives data from robot
    if (init_recv_conn(receiver_addr, receiver_fd) < 0)
        return -1;
    vector<string> episode_names;
    get_all_episode_names(episode_names);
    // run the episode loop on individual episodes
    for (auto &title : episode_names)
    {
        Episode current_episode;
        get_episode_metadata(title, current_episode);
        // would-be init control pipeline
        update_loop(current_episode);
    }
    cout << "\033[32m"
         << "Finished all episodes"
         << "\033[00m" << endl;
    // once completed all episodes, close socket connections
    close_sockets(sender_fd, receiver_fd);
    return 0;
}

/**
 * @brief sense action updating the robot status, sim state history and time
 * @param[out] robot_on The status of the robot (on or off)
 * @param[out] frame the current time frame of the simulator
 * @param[out] hist The sim_state history indexed by frame
 * @param[in] max_time The maximum time allocated for this episode
 */
void joystick_sense(bool &robot_on, int &sim_time, unordered_map<int, SimState> &hist, const float max_time);

/**
 * @brief planning algorithm of the robot given the sim history and time
 * @param[in] robot_on The status of the robot (on or off)
 * @param[in] frame the current time frame of the simulator
 * @param[in] hist The sim_state history indexed by frame
 */
void joystick_plan(const bool robot_on, const int sim_t, const unordered_map<int, SimState> &hist);

/**
 * @brief sending act commands to the robot to execute
 * @param[in] robot_on The status of the robot (on or off)
 * @param[in] frame the current time frame of the simulator
 * @param[in] hist The sim_state history indexed by frame
 */
void joystick_act(const bool robot_on, const int sim_t, const unordered_map<int, SimState> &hist);

/**
 * @brief run the sense() plan() act() loop of the joystick controller
 * @param[in] ep The episode metadata in the form of an Episode class
 */
void update_loop(const Episode &ep)
{
    unordered_map<int, SimState> sim_state_hist;
    bool robot_on = true;
    int frame = 0;
    const float max_t = ep.get_time_budget();
    cout << "\033[35m"
         << "Starting episode: " << ep.get_title() << "\033[00m" << endl;
    while (robot_on)
    {
        // gather information about the world state based off the simulator
        joystick_sense(robot_on, frame, sim_state_hist, max_t);
        // create a plan for the next steps of the trajectory
        joystick_plan(robot_on, frame, sim_state_hist);
        // send commands to the robot to execute
        joystick_act(robot_on, frame, sim_state_hist);
    }
    cout << "\n\033[32m"
         << "Finished episode: " << ep.get_title() << "\033[00m" << endl;
}

/**
 * @brief sense action updating the robot status, sim state history and time
 * @param[out] robot_on The status of the robot (on or off)
 * @param[out] frame the current time frame of the simulator
 * @param[out] hist The sim_state history indexed by frame
 * @param[in] max_time The maximum time allocated for this episode
 */
void joystick_sense(bool &robot_on, int &frame, unordered_map<int, SimState> &hist, const float max_time)
{
    vector<char> raw_data;
    // send keyword (trigger sense action) and await response
    if (send_to_robot("sense") >= 0 && listen_once(raw_data) >= 0)
    {
        // process the raw_data into a sim_state
        json sim_state_json = json::parse(raw_data);
        SimState new_state = SimState::construct_from_json(sim_state_json);
        // the new time from the simulator (converted to frame:int for hashing)
        frame = round(new_state.get_sim_t() / 0.05); // default dt = 0.05 (sim tick)
        // update robot running status
        robot_on = new_state.get_robot_status();
        // add print output:
        cout.setf(ios::fixed, ios::floatfield);
        cout.precision(3);
        cout << "\033[36m"
             << "\33[2K" // clear old line
             << "Updated state of the world for time = " << new_state.get_sim_t() << " out of " << max_time
             << "\033[00m\r" << flush;
        // add new sim_state to the history
        hist[frame] = new_state;
    }
    else // connection failure, power off the robot
        robot_on = false;
}

/**
 * @brief planning algorithm of the robot given the sim history and time
 * @param[in] robot_on The status of the robot (on or off)
 * @param[in] frame the current time frame of the simulator
 * @param[in] hist The sim_state history indexed by frame
 */
void joystick_plan(const bool robot_on, const int sim_t, const unordered_map<int, SimState> &hist)
{
    // This is left blank as a random planner for now
    return;
}

/**
 * @brief wrapper of send_to_robot that packages velocity cmds into json
 * @param[in] v The linear velocity component of the command
 * @param[in] w The angular velocity component of the command
 */
void send_cmd(const float v, const float w)
{
    // NOTE: this joystick currently only supports system-dynamics commands
    // and therefore only sends velocity commands
    json message;
    // Recall, commands are sent as list of lists where inner lists
    // form the commands v & w, but the outer list contains these commands
    // in case multiple should be sent across a single update (ie. when
    // simulator dt and joystick dt don't match)
    message["j_input"] = {{v, w}};
    send_to_robot(message.dump());
}

/**
 * @brief sending act commands to the robot to execute
 * @param[in] robot_on The status of the robot (on or off)
 * @param[in] frame the current time frame of the simulator
 * @param[in] hist The sim_state history indexed by frame
 */
void joystick_act(const bool robot_on, const int frame, const unordered_map<int, SimState> &hist)
{
    string termination_cause;
    if (hist.find(frame) != hist.end())
    {
        const SimState *sim_state = &hist.at(frame); // pointer (not deepcopy)
        termination_cause = sim_state->get_termination_cause();
    }
    else
        termination_cause = "Disconnection";
    if (robot_on)
    {
        // Currently send random commands
        const float max_v = 1.2;
        const float max_w = 1.1;
        const int p = 100; // 2 decimal places of precision
        // between [0 and max_V]
        float rand_v = ((rand() % int(p * max_v)) / float(p));
        // between [-max_w and max_w]
        float rand_w = ((rand() % int(2 * p * max_w)) / float(p)) - max_w;
        send_cmd(rand_v, rand_w);
    }
    else
    {
        cout << "\nPowering off joystick, robot terminated with: " << termination_cause << endl;
    }
}

/**
 * @brief wrapper of listen_once that parses the data into episode names
 * @param[in] episodes The vector of episode names to return
 */
void get_all_episode_names(vector<string> &episodes)
{
    int data_len;
    vector<char> raw_data;
    data_len = listen_once(raw_data);
    // parse the episode names from the raw data
    json ep_data = json::parse(raw_data);
    cout << "Received episodes: [";
    for (auto &s : ep_data["episodes"])
    {
        cout << s << ", ";
        episodes.push_back(s);
    }
    cout << "]" << endl;
}

/**
 * @brief wrapper of listen_once that parses the data of an episode's metadata
 * @param[in] ep_name The name of the episode that will be created
 * @param[out] ep The episode
 */
void get_episode_metadata(const string &ep_name, Episode &ep)
{
    cout << "Waiting for episode: " << ep_name << endl;
    int ep_len;
    vector<char> raw_data;
    ep_len = listen_once(raw_data);
    // parse the episode_names raw data from the connection
    json metadata = json::parse(raw_data);
    // TODO: move to static Episode class
    ep = Episode::construct_from_json(metadata);
    // notify the robot that all the metadata has been obtained
    // to begin the simualtion
    send_to_robot("ready");
}