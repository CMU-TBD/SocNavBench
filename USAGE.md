# Usage and Information
## Overall Structure of the Simulator
The `Simulator` used in `SocNavBench` runs through a single episode to execute and measure the `RobotAgent`'s planning algorithm for a particular episode. An episode consists of the parameters for a particular simulation, such as the pedestrians, environment, and robot's start/goal positions. In order to measure arbitrary planning algorithms we provide a `Joystick` API that translates data to and from the simulator to control the robot. 

All our agents undergo a `sense()->plan()->act()` cycle to perceive and interact with the world. 

The simulator can be run in synchronous and asynchronous modes. In synchronous-mode the joystick will block on the arrival of data from its `sense()` call, and the `RobotAgent` will equivalently block on the actions/commands sent from the joystick, making their communication transaction 1:1 with the simulator time. In asynchronous-mode the simulator will run in real time and the joystick's planning algorithm will have to keep up. 

## Top Level Overview
![Structure Graphic](https://docs.google.com/drawings/d/e/2PACX-1vSKzu4L14_2Y6XrHz5HTfNXPPkpJShYqjE_G3wN8tBz4a7bBrhjSYl1HHVASgzX8L0-wV9V7PT2g55j/pub?w=1107&h=614)
## More Detailed Overview
![Detailed Structure](https://docs.google.com/drawings/d/e/2PACX-1vRlskIHFdvq8ZDlo0tUN5Z_BPCA87UldJgg6dgsbaMBOmkXgVecxNijFm9fxjgJSO-yj16HcGUaaK0G/pub?w=1075&h=798)

## Running `SocNavBench`
To start the main process for `SocNavBench` enter the main directory and run the first command below (1) to see `Waiting for joystick connection...` Then run the second command (2) as a separate executable (ie. in another shell instance) to start the `Joystick` process.
```
# The command to start the simulator (1)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_episodes.py

# The command to start the joystick executable (2)
python3 joystick/joystick_client.py

# now the two executables will complete the connection handshake and run side-by-side
```
Note that the first time `SocNavBench` is run on a specific map it will generate a `traversible` (bitmap of non-obstructed areas in the map) that will be used for the simulation's environment. This traversible is then serialized under `SocNavBenchmark/sd3dis/stanford_building_parser_dataset/traversibles/` so it does not get regenerated upon repeated runs on the same map.

## More about the `Joystick` API
In order to communicate with the robot's sense-plan-act cycle from a process external to the simulator we provide this "Joystick" interface. In synchronous mode the `RobotAgent` (and by extension `Simulator`) blocks on the socket-based data transmission between the joystick and the robot, providing 'free thinking time' as the simulator time stops until the robot progresses. To learn more see [`SocNavBench/joystick`](joystick/).

The `Joystick` is expected to complete several key functions:
- `init_send_conn()` which establishes the socket that sends data to the robot.
- `init_recv_conn()` which establishes the socket that receives data from the robot.
- `get_all_episode_names()` which gets all the episodes (by name) that will be executed.
- `get_episode_metadata()` which gets the specific metadata (see `Episodes` below) for the incoming episode.

These functions are provided in `python` and `c++` in `joystick/joystick_py/` and `joystick/joystick_cpp/` respectively.

Users will be responsible for implementing their own:
- `sense()` which requests a JSON-serialized [`sim_state`](simulators/sim_state.py) and parses it.
- `plan()` which uses the new information from the sense to determine next steps/actions to take.
- `act()` which sends commands to the robot to execute in the simulator.

For communications we use the `AF_UNIX` protocol for the fastest communications within a local host machine. The default socket identifiers are `/tmp/socnavbench_joystick_recv` and `/tmp/socnavbench_joystick_send` which may be modified in [`params/user_params.ini`](params/user_params.ini) under `[robot_params]`. 
  
### A starting off point
For joystick implementations we have provided two sample joystick instances in `python` and one in `c++` under [`SocNavBench/joystick/`](joystick/):
- `joystick_random.py` which uses a generic random planner, showcasing one of the simplest uses of the Joystick interface.
- `joystick_planner.py` which uses a basic sampling planner that can take the robot to the goal without obstacle collisions (has no notion of pedestrians).
- `joystick_client.cpp` which is a simple random planner as well, but in `c++`. Highlithgint the language-agnostic nature of the API.

Note that the provided joystick implementations (in python) are still executed by running `joystick_client.py` but is defaulted to using the `joystick_planner` implementation. To use the `joystick_random` implementation you can toggle the flag for `use_random_planner` under `[joystick_params]` in [`user_params.ini`](params/user_params.ini).

Also note that joystick must be run in an external process (but within the same `conda env`). Make sure before running `joystick_client.py` that the conda environment is `socnavbench` (same as for `test_socnav.py` and `test_episodes.py`)

## More about the `Robot`
As depicted in the [`user_params.ini`](params/user_params.ini) param file, the default robot is modeled after a [Pioneer P3DX robot](https://www.generationrobots.com/media/Pioneer3DX-P3DX-RevA.pdf). Since the simulation primaily focuses on the base of the robot, those are the dimensions we use. 

Also note that we are making the assumption that both the system dynamics of the robot and the environment are the same. More information about the system dynamics can be found in the [`SocNavBench/systems`](systems/) directory for the Dubins Car models we use. 

Additionally, we provide the option to not use system dynamics at all and instead have the robot *teleport* to a position within the feasible velocity range. This can be toggled from the `use_system_dynamics` flag under `[joystick_params]` in [`user_params.ini`](params/user_params.ini). This requires a slight modification in the `Joystick` API which will now be expecting a tuple of `(x, y, theta, v)` instead of `(v, w)` linear/angular velocity commands.

Other functionality of the robot includes:
  - Listening for `Joystick` keywords such as `"sense"`, `"ready"`, `"algo: XYZ"`, or `"abandon"` to specify what action to take given the request. 
    - `"sense"` will send a `sim_state` (see below) to the running `Joystick`.
    - `"ready"` notifies the robot that the `"Joystick"` has fully received the episode metadata and is ready to begin the simulation.
    - `"algo: XYZ"` where `XYZ` is the name of an algorithm (such as `"Random", "Sampling"`) to tell the simulator which planning algorithm is being used (optional).
    - `"abandon"` to instantly power off the robot and end its acting.
  - If the message sent to the robot is not a keyword then it is assumed to be commands (or multiple commands) from the `act()` phase.

## More about the `Episodes`
Our episodes consists of all the params used to define a typical scene. See [`episode_params_val.ini`](params/episode_params_val.ini) for examples:
- `map_name` which holds the title of the specific building (in `sd3dis`) to use for this episode 
- `pedestrian_datasets` which holds the names of the datasets to run in this episode (see [`params/dataset_params.ini`](params/dataset_params.ini))
- `datasets_start_t` which holds the starting times for the corresponding pedestrian datasets being used.
- `ped_ranges` which holds the range of pedestrians to use for the corresponding pedestrian dataset being used.
- `agents_start/end` which holds the custom start/goal positions of the autonomous agents (using `SamplingPlanner`)
- `robot_start_goal` which holds the start position and goal position of the robot
  - Note for all our "pos3's" we use (x, y, theta)
- `max_time` which holds the time budget for this episode
- `write_episode_log` which flags the option to write a simulator log at the end of the simulation

To choose which tests to run, edit the `tests` list under `[episode_params]` in [`episode_params_val.ini`](params/episode_params_val.ini) which holds the names of the tests to run.

## More about the `Agents`
Pedestrians used in the simulator can be either autonomous agents or prerecorded agents. The autonomous agents follow the same `SamplingPlanner` planner to reach their goal. However we also include the option to use prerecorded agents which are from an open-source dataset that recorded real human trajectories in various environments. 
- Since the prerecorded agents have a set trajectory as they are spawned, they cannot interact with the environment or change course to avoid obstacles
- We do provide the option under `[agent_params]` in [`user_params.ini`](params/user_params.ini) to toggle `pause_on_collide` which will pause the pedestrians' motion for `collision_cooldown_amnt` seconds (simulator time) after a collision with the robot. This feature applies to both Auto-agents as well as Prerecorded-agents. 

The pedestrian datasets are also a component of the user-editable params under [`dataset_params.ini`](params/dataset_params.ini) which define the following params:
- `file_name` which holds the relative file location of the `.csv` file for the dataset.
- `fps` which holds the framerate that the dataset was captured at.
- `spawn_delay_s` which holds the number of seconds that all agents wait after spawning before moving.
- `offset` which holds the translational (x, y, theta) values for shifting the entire dataset.
- `swapxy` which is used to swap the x and y's from the dataset if the formatting does not match ours
- `flipxn` which is used to negate the x positions
- `flipyn` which is used to negate the y positions

## More about the `Simulator`
The `Simulator` progresses the state of the world in the main `simulate()` loop, which spawns update threads for all the agents in the scene, updates the robot and captures a "snapshot" of the current simulator status in the form of a `sim_state` that is stored for later use.

By default the simulator runs in "synchronous mode", meaning that it will freeze time until the robot (joystick API) responds. In synchronous mode the robot's "thinking time" is free as the simulator blocks until its reception. In asynchronous mode the simulator runs alongside real-world time and does not wait on the robot's input at all. This means the robot could take too long to think and miss the simulator's cue, in this case the robot may repeat the last command sent by the joystick API or do nothing at all. The maximum number of times the robot can repeat commands is set as `max_repeats` under `[robot_params]` in [`params/user_params.ini`](params/user_params.ini). In order to toggle the simulator's synchronicity mode edit the `synchronous_mode` param in `[simulator_params]` in [`params/user_params.ini`](params/user_params.ini).


## More about `sim_states`
Sim-states are what we use to keep track of a snapshot of the simulator from a high level with no information about the past/future. For example, a `sim_state` instance might contain information similar to what a robot would sense using a lidar sensor, that being the environment and the current pos3's (x, y, theta) of the agents (no velocity, acceleration, trajectory, starts/goals, etc.).

Also note that `sim_states` are designed to be easily serializable and deserializable, allowing for simple `JSON` transmissions between the `RobotAgent` and the `Joystick` API. 

More information about the `sim_states` can be found in [`simulators/sim_state.py`](simulators/sim_state.py)


## Visualization
The default rendering mode is `Schematic` which renders only the topview of the episode. The topdown view only uses matplotlib to render a "bird's-eye-view" perspective without needing the intensive OpenGL Swiftshader renderer. However, to visualize the Depth/RGB modes change the `render_mode` parameter in [`params/user_params.ini`](params/user_params.ini) to `full-render`. Note that currently the program does not support parallel image rendering when using the 3D renderer, making it very time consuming. The `schematic` rendering mode utilizes python `multiprocessing` to render frames in paralell, to maximize performance and utilize all your machine's cores, edit `num_render_cores` in [`params/user_params.ini`](params/user_params.ini) under `[simulator_params]`. 

## Rendering Modes:
<img src="https://docs.google.com/drawings/d/e/2PACX-1vTyCc098f0Rk__i8p4xwcMIELorIsQ3BSvN2k-ntomr8olhWEaIWs4EJGJ8MdGTLkvaygODNIuOvHed/pub?w=1288&h=440" alt="drawing" width="100%"/>