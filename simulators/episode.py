

class Episode():
    def __init__(self, name: str, environment: dict, agents: dict,
                 t_budget: float, r_start: list, r_goal: list):
        self.name = name
        self.environment = environment
        self.agents = agents
        self.time_budget = t_budget
        self.robot_start = r_start    # starting position of the robot
        self.robot_goal = r_goal      # goal position of the robot

    def get_name(self):
        return self.name

    def get_environment(self):
        return self.environment

    def get_agents(self):
        return self.agents

    def get_time_budget(self):
        return self.time_budget

    def update(self, env, agents):
        if(not (not env)):  # only upate env if non-empty
            self.environment = env
        # update dict of agents
        self.agents = agents

    def get_robot_start(self):
        return self.robot_start

    def get_robot_goal(self):
        return self.robot_goal
