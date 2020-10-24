### What is this for?

Consider a typical grid-based A* search algorithm on a simple 2D grid, agents at any point may choose between going up/down/left/right and to their respective diagonals. However, with dynamics based agents, such as the "humans" in `SocNavBench`, it is important to take into account the *subset* of available positions that an agent can take from their current config/state. For example, (considering a 2D grid) agents with a high 'upwards' velocity would likely be unable instantly negate their velocity and traverse through their bottom left/right diagonal grids in a single step (though they could in multiple), therefore to better simulate gradual smooth traejctory motions, velocity binning is an important mechanism to ensure agents respect their system dynamics during the planning and traversal of their trajectories.

### What does this folder do with the velocity bins?
This can be generalized to much more than the 8 grid spaces around a singular point, and thus can be very taxing to generate and recompute on every iteration of the agents, therefore we have serialized them during an initial run of the simulation and they will be saved to this folder for future ease of use. Since generating these files takes a while, they are saved as native python serialized files and can be reused with ease. 

### Example directory
An example of what this directory could look like (given the corresponding dynamics model parameters):
```
wayptnav_data
├── control_pipelines
│   └── control_pipeline_v0
│       ├── planning_horizon_200_dt_0.05
│       │   └── dubins_v2
│       │       ├── image_plane_projected_grid_n_21483_theta_bins_21_bound_min_-0.01_-0.01_-100000000.00_bound_max_0.01_0.01_0.00
│       │       │   └── 20_velocity_bins
│       │       └── image_plane_projected_grid_n_21483_theta_bins_21_bound_min_-0.01_-0.01_-3.14_bound_max_0.01_0.01_0.00
│       │           └── 20_velocity_bins
│       │               ├── incorrectly_binned.pkl
│       │               ├── velocity_0.000.pkl
│       │               ├── velocity_0.032.pkl
│       │               ├── velocity_0.063.pkl
│       │               ├── velocity_0.095.pkl
│       │               ├── velocity_0.126.pkl
│       │               ├── velocity_0.158.pkl
│       │               ├── velocity_0.189.pkl
│       │               ├── velocity_0.221.pkl
│       │               ├── velocity_0.253.pkl
│       │               ├── velocity_0.284.pkl
│       │               ├── velocity_0.316.pkl
│       │               ├── velocity_0.347.pkl
│       │               ├── velocity_0.379.pkl
│       │               ├── velocity_0.411.pkl
│       │               ├── velocity_0.442.pkl
│       │               ├── velocity_0.474.pkl
│       │               ├── velocity_0.505.pkl
│       │               ├── velocity_0.537.pkl
│       │               ├── velocity_0.568.pkl
│       │               └── velocity_0.600.pkl
│       └── planning_horizon_67_dt_0.15
│           └── dubins_v2
│               └── image_plane_projected_grid_n_21483_theta_bins_21_bound_min_-0.01_-0.01_-3.14_bound_max_0.01_0.01_0.00
│                   └── 20_velocity_bins
│                       ├── incorrectly_binned.pkl
│                       ├── velocity_0.000.pkl
│                       ├── velocity_0.032.pkl
│                       ├── velocity_0.063.pkl
│                       ├── velocity_0.095.pkl
│                       ├── velocity_0.126.pkl
│                       ├── velocity_0.158.pkl
│                       ├── velocity_0.189.pkl
│                       ├── velocity_0.221.pkl
│                       ├── velocity_0.253.pkl
│                       ├── velocity_0.284.pkl
│                       ├── velocity_0.316.pkl
│                       ├── velocity_0.347.pkl
│                       ├── velocity_0.379.pkl
│                       ├── velocity_0.411.pkl
│                       ├── velocity_0.442.pkl
│                       ├── velocity_0.474.pkl
│                       ├── velocity_0.505.pkl
│                       ├── velocity_0.537.pkl
│                       ├── velocity_0.568.pkl
│                       └── velocity_0.600.pkl
└── README.md
```
