# Usage and Information
## Overall Structure of the Simulator
The primary `SocNavBench` program runs through the various episodes provided (see [`episode_params_val.ini`](params/episode_params_val.ini)) and spawns a `CentralSimulator` for each test, the initial states of those simulators are based off the running test. However, in order to start an episode there must also be an external `Joystick` process that is used to send commands, requests, and signals to the robot through a socket communication protocol. The `Joystick` is what users will primarily be interacting with, as it provides the interface for any planning algorithm. 

![Structure Graphic](https://drive.google.com/uc?export=download&id=1FUtc420QOcYp57q-9XqSktABfcfYbiBJ)


## Running `SocNavBench`
To start the main process for `SocNavBench` enter the main directory and run the first command below (1) to see `Waiting for joystick connection...` Now run the second command (2) as a separate executable (ie. in another shell instance) to start the `Joystick` process.
```
# The command to start the simulator (1)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_episodes.py

# The command to start the joystick executable (2)
PYTHONPATH='.' python3 joystick/test_example_joystick.py

# now the two executables will complete the connection handshake and run synchronously
```

## More about the `Joystick`
The main program relies on (and will block on) an inter-process communication channel such as the socket connection between the it and the external "Joystick process". This process will be the "brain" of the robot that plans a route and sends commands based off the information it is given.

The joystick can:
- `sense()` by requesting a json-serialized `sim_state` which holds all the information about that state.
- `plan()` by using the updated information about the current state, such as the current simulator time, agent positions, and environment.
- `act()` by sending specific velocity commands to the robot to execute in the simulator, which the `CentralSimulator` blocks until the commands are sent. 
- To start a joystick executable you can simply run the `test_example_joystick.py` which will work independently of the type of `Joystick` class that is being used. 

As a starting-off point, we've provided two sample classes in `joystick/`:
- `example_joystick.py` which holds `JoystickRandom` uses a generic random planner that showcases one of the lightest uses of the Joystick interface.
- `joystick_planner.py` which holds `JoystikWithPlanner` uses a basic sampling planner that showcases how a typical planner implementation might be integrated with the interface. 


The joystick can be made to run synchronously with the simulator or asynchronously by repeating the last command sent for a number of simulator frames. This can be toggled in [`params/user_params.ini`](params/user_params.ini) by editing the `block_joystick` param in `[simulator_params]`

The communication port is defaulted to 6000, this can be changed by editing `port` in [`params/user_params.ini`](params/user_params.ini) under `[robot_params]`
  - Note that the program actually uses two sockets to ensure bidirectional communications for asynchronous data transmission. We have designited the successor of `port` to be set as the robot receiver port. Therefore in our default case, we are actually using ports 6000 and 6001.

The joystick must be run in an external process (but within the same `conda env`)
- Therefore, make sure before running `test_joystick.py` that the conda environment is `socnavbench` (same as for `test_socnav.py` and `test_episodes.py`)

## More about the `Robot`
As depicted in the [`user_params.ini`](params/user_params.ini) param file, the default robot is modeled after a [Pioneer P3DX robot](https://www.generationrobots.com/media/Pioneer3DX-P3DX-RevA.pdf). Since the simulation primaily focuses on the base of the robot, those are the dimensions we use. 

Also note that we are making the assumption that both the system dynamics of the robot and the environment are the same. But more information about the system dynamics can be found in the `Joystick` instances, since they are given the main params of the system dynamics as seen in the [`user_params.ini`](params/user_params.ini).

Other functionality of the robot includes:
  - Listening for joystick keywords such as `"sense"`, `"ready"`, or `"abandon"` to specify what action to take given the request. 
    - `"sense"` will send a `sim_state` (see below) to the running `Joystick`.
    - `"ready"` notifies the robot that the `"Joystick"` has fully received the episode metadata and is ready to begin the simulation.
    - `"algo: XYZ"` where `XYZ` is the name of an algorithm, will tell the robot (and thus `SocNavBench`) what planning algorithm is being used
    - `"abandon"` to instantly power off the robot and end its acting.
  - If the message sent to the robot is not a keyword then it is assumed to be commands (or multiple commands) from the `act()` phase.

## More about the `Episodes`
Our episodes consists of all the fields seen for each test in [`episode_params_val.ini`](params/episode_params_val.ini) such as:

- `map_name` which holds the title of the specific building (in `sd3dis`) to use for this episode 
- `pedestrian_datasets` which holds the names of the datasets to run in this episode (see [`params/dataset_params.ini`](params/dataset_params.ini))
- `datasets_start_t` which holds the starting times of the datasets from `pedestrian_datasets`
- `agents_start/end` which holds the custom start/goal positions of the autonomous agents
- `robot_start_goal` which holds the start position and goal position of the robot
  - Note for all our "pos3's" we use (x, y, theta)
- `max_time` which holds the time budget for this episode
- `write_episode_log` which flags the option to write a simulator log at the end of the simulation

To choose which tests to run, edit the `tests` under `[episode_params_val]` in `episode_params_val.ini` which holds the names of the tests to run.

## More about the `CentralSimulator`
The `CentralSimulator` progresses the state of the world in the main `simulate()` loop, which spawns update threads for all the agents in the scene, updates the robot (which blocks on the joystick input) and captures a "snapshot" of the current simulator status in the form of a `sim_state` that is stored for later use. 

Once the episode simulation has completed, the `CentralSimulator` will convert all its stored `sim_state` instances into image frames (`.png`'s) of the state of the world at that particular time. For non-3D-generated images these conversions can be done in parallel and merged afterwards to generate a `.gif` movie of the entire episode with the simulator time correlating to real-world viewing time.

## More about `sim_states`
Sim-states are what we use to keep track of the simulator without going deep into the internals of the moving parts. For example, a `sim_state` instance might contain information similar to what a robot would sense using a lidar sensor, that being the environment and the current pos3's (x, y, theta) of the agents (no velocity, acceleration, trajectory, starts/goals, etc.).

Also note that `sim_states` are designed to be READ-ONLY as well as easily serializable and deserializable, which is what allows for simple `JSON` transmissions between the `RobotAgent` and the `Joystick`. 

More information about the `sim_states` can be found in [`simulators/sim_state.py`](simulators/sim_state.py)


## Visualization
The default rendering mode is `Schematic`, which renders only the topview of the episode. The topdown view only uses matplotlib to render a "bird's-eye-view" perspective without needing the intensive OpenGL Swiftshader renderer. However, to visualize the Depth/RGB modes change the `render_mode` parameter in [`params/user_params.ini`](params/user_params.ini) to `full-render`. Note that currently the program does not support parallel image rendering when using the 3D renderer, making it very time consuming.
