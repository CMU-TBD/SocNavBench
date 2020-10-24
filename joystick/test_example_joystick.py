from params.central_params import create_joystick_params
from joystick.example_joystick import JoystickRandom
from joystick.joystick_planner import JoystickWithPlanner
from joystick.joystick_planner import JoystickWithPlannerPosns


if __name__ == '__main__':
    joystick_params = create_joystick_params()
    if(joystick_params.use_random_planner):
        J = JoystickRandom()
    else:
        if joystick_params.use_system_dynamics:
            # uses the joystick that sends velocity commands instead of positional
            J = JoystickWithPlanner()
        else:
            # uses the joystick that sends positional commands instead of velocity
            J = JoystickWithPlannerPosns()

    """start the joystick process"""
    J.init_send_conn()
    J.init_recv_conn()
    # first listen() for the episode names
    assert(J.get_all_episode_names())
    episodes = J.get_episodes()

    # we want to run on at least one episode
    assert(len(episodes) > 0)
    for ep_title in episodes:
        print("Waiting for episode: {}".format(ep_title))

        # second listen() for the specific episode details
        J.get_episode_metadata()
        assert(J.current_ep and J.current_ep.get_name() == ep_title)

        J.init_control_pipeline()

        J.update_loop()
