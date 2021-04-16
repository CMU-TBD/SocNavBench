import os
import numpy as np
from utils.utils import touch, natural_sort
from utils.utils import color_red, color_green, color_reset
import glob
import imageio


def plot_image_observation(ax, img_mkd, size=None):
    """
    Plot an image observation (occupancy_grid, rgb, or depth).
    The image to be plotted is img_mkd an mxk image with d channels.
    """
    if img_mkd.shape[2] == 1:  # plot an occupancy grid image
        ax.imshow(img_mkd[:, :, 0], cmap='gray', extent=(
            0, size, -1 * size / 2.0, size / 2.0))
    elif img_mkd.shape[2] == 3:  # plot an rgb image
        ax.imshow(img_mkd.astype(np.int32))
        ax.grid(False)
    else:
        raise NotImplementedError


def gather_metadata(ppm: float, a, plot_start_goal: bool, start: list,
                    goal: list, traj_col: str = ''):
    # collision means either the agent collided with an obstacle (get_collided) or
    # the agent has recently been collided with and is on a "collision cooldown"
    collided = a.get_collided() or (a.get_collision_cooldown() > 0)
    markersize = a.get_radius() * ppm
    pos_3 = a.get_current_config().to_3D_numpy()
    if traj_col == "":
        traj_col = a.get_color()
    start_3 = None
    goal_3 = None
    if plot_start_goal:
        try:
            # set the start and goal if it exists in the agent, else use the provided
            start_3 = a.get_start_config().to_3D_numpy()
            goal_3 = a.get_goal_config().to_3D_numpy()
        except:
            # use the start_3 and goal_3 provided
            start_3 = start
            goal_3 = goal
        assert(start_3 is not None)
        assert(goal_3 is not None)

    return collided, markersize, pos_3, traj_col, start_3, goal_3


def gather_colors_and_labels(label: str, indx: int):
    start_col = 'yo'  # yellow circle
    goal_col = 'g'   # yellow (star)
    draw_label = None
    sl = None
    gl = None
    if indx == 0:
        # Only add label on the first humans
        draw_label = label
        sl = label + " start"
        gl = label + " goal"
    return start_col, goal_col, draw_label, sl, gl


def plot_agent_dict(ax, ppm: float, agents_dict: dict, label='Agent', normal_color='bo',
                    collided_color='ro', plot_trajectory=True, plot_quiver=False, alpha=1,
                    traj_color='', plot_start_goal=False, start_3=None, goal_3=None, traj_clip=0):
    # plot all the simulated prerecorded gen_agents
    for i, a in enumerate(agents_dict.values()):
        # gather important info regarding the values to plot
        collided, ms, pos_3, traj_col, start_3, goal_3 = \
            gather_metadata(ppm, a, plot_start_goal,
                            start_3, goal_3, traj_color)

        # render agent's trajectory
        if plot_trajectory and a.get_trajectory():
            a.get_trajectory().render(ax, freq=1, color=traj_col,
                                      alpha=alpha, plot_quiver=False,
                                      clip=traj_clip, linewidth=ppm / 8.2)

        # gather colors/labels for the agent plot
        start_col, goal_col, draw_label, sl, gl = \
            gather_colors_and_labels(label, i)

        # draw little dot in the middle of the collided agents if collision occurs
        if collided:
            ax.plot(pos_3[0], pos_3[1], collided_color, markersize=ms,
                    label=draw_label)
            ax.plot(pos_3[0], pos_3[1], normal_color, markersize=ms * 0.4,
                    label=None)
        else:
            ax.plot(pos_3[0], pos_3[1], normal_color, markersize=ms,
                    label=draw_label)

        # plot collision indicator
        # plot start + goal
        if plot_start_goal:
            ax.plot(start_3[0], start_3[1], start_col,
                    markersize=ms, label=sl, alpha=0.5)
            ax.plot(goal_3[0], goal_3[1], goal_col,
                    markersize=2 * ms, marker="*", label=gl, alpha=0.8)

        # plot a surrounding "force field" around the agent
        if plot_quiver:
            # Agent heading
            s = 0.5
            ax.quiver(pos_3[0], pos_3[1], s * np.cos(pos_3[2]), s * np.sin(pos_3[2]),
                      scale=1, scale_units='xy')
            if(plot_start_goal and (start_3 is not None and goal_3 is not None)):
                ax.quiver(start_3[0], start_3[1], s * np.cos(start_3[2]), s * np.sin(start_3[2]),
                          scale=1, scale_units='xy')
                ax.quiver(goal_3[0], goal_3[1], s * np.cos(goal_3[2]), s * np.sin(goal_3[2]),
                          scale=1, scale_units='xy')


def plot_topview(ax, extent, traversible, human_traversible, camera_pos_13,
                 pedestrians, robots, room_center, plot_quiver=False, plot_meter_tick=False):
    """Uses matplotlib to plot a birds-eye-view image of the world by plotting the environment
    and the gen_agents on every frame. The frame also includes the simulator time and wall clock time

    Args:
        ax (matplotlib.axes): the axes to plot on
        extent (np.array): the real_world extent (in meters) of the traversible
        traversible (np.array): the environment traversible (map bitmap)
        human_traversible (np.array/None): the human traversibles (or None for non-3D-plots)
        camera_pos_13 (np.array): the position of the camera (and robot)
        agents (AgentState dict): the agent states
        prerecs (HumanState dict): the prerecorded agent states
        robots (AgentState dict): the robots states
        room_center (np.array): the center of the "room" to focus the image plot off of
        plot_quiver (bool, optional): whether or not to plot the quiver (arrow). Defaults to False.
    """
    # get number of pixels-per-meter based off the ax plot space
    img_scale = \
        ax.transData.transform((0, 1)) - ax.transData.transform((0, 0))
    # scale the pixels-per-meter based off the image scale
    ppm = int(img_scale[1])
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-0.5, vmax=1.5, origin='lower')
    # Plot human traversible
    if human_traversible is not None:
        # NOTE: the human radius is only available given the openGL human modeling
        # and rendering, thus p.render_with_display must be True
        # Plot the 5x5 meter human radius grid atop the environment traversible
        alphas = np.empty(np.shape(human_traversible))
        for y in range(human_traversible.shape[1]):
            for x in range(human_traversible.shape[0]):
                alphas[x][y] = not(human_traversible[x][y])
        ax.imshow(human_traversible, extent=extent, cmap='autumn_r',
                  vmin=-.5, vmax=1.5, origin='lower', alpha=alphas)
        # alphas = np.all(np.logical_not(human_traversible))

    # TODO: make plot_quiver a simulator-wide param for pedestrians and robot
    # Plot the camera (robots)
    plot_agent_dict(ax, ppm, robots, label="Robot", normal_color="ro",
                    collided_color="ro", plot_quiver=plot_quiver, plot_start_goal=True,
                    alpha=0.8, traj_color="w")

    # plot all the simulated pedestrian agents
    plot_agent_dict(ax, ppm, pedestrians, label="Pedestrian", normal_color="co",
                    collided_color="ro", plot_quiver=plot_quiver, plot_start_goal=False,
                    alpha=0.3, traj_clip=50)

    if plot_meter_tick:
        # plot other useful informational visuals in the topview
        # such as the key to the length of a "meter" unit
        plot_line_loc = room_center[:2] * 0.65
        start = [0, 0] + plot_line_loc
        end = [1, 0] + plot_line_loc
        gather_xs = [start[0], end[0]]
        gather_ys = [start[1], end[1]]
        col = 'k-'
        h = 0.1  # height of the "ticks" of the key
        ax.plot(gather_xs, gather_ys, col)  # main line
        ax.plot([start[0], start[0]], [start[1] +
                                       h, start[1] - h], col)  # tick left
        ax.plot([end[0], end[0]], [end[1] + h,
                                   end[1] - h], col)  # tick right
        if plot_quiver:
            ax.text(0.5 * (start[0] + end[0]) - 0.2, start[1] +
                    0.5, "1m", fontsize=14, verticalalignment='top')


def render_scene(plt, p, rgb_image_1mk3, depth_image_1mk1, environment,
                 camera_pos_13, pedestrians, robots,
                 sim_t: float, wall_t: float, filename: str, with_zoom=False):
    """Plots a single frame from information provided about the world state

    Args:
        p (Map): Simulator params
        rgb_image_1mk3: the RGB image of the world_state to plot
        depth_image_1mk1: the depth-map image of the world_state to plot
        environment (dict): dictionary housing the obj map (bitmap) and more
        camera_pos_13 (np.array): the position of the camera (and robot)
        agents (AgentState dict): the agent states
        prerecs (HumanState dict): the prerecorded agent states
        robots (AgentState dict): the robots states
        sim_t (float): the simulator time in seconds
        wall_t (float): the wall clock time in seconds
        filename (str): the name of the file to save
    """
    map_scale = environment["map_scale"]
    room_center = environment["room_center"]
    # Obstacles/building traversible
    traversible = environment["map_traversible"]
    human_traversible = None
    if "human_traversible" in environment.keys():
        assert(p.render_3D)
        human_traversible = environment["human_traversible"]
    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent) * map_scale

    # count used to signify the number of images that will be generated in a single frame
    # default 1, for normal view (can add a second for zoomed view)
    plot_count = 1
    if with_zoom:
        plot_count += 1
        zoomed_img_plt_indx = plot_count  # 2
    if rgb_image_1mk3 is not None:
        plot_count = plot_count + 1
        rgb_img_plt_indx = plot_count  # 3
    if depth_image_1mk1 is not None:
        plot_count = plot_count + 1
        depth_img_plt_indx = plot_count  # 4

    img_size = 10 * p.img_scale
    fig = plt.figure(figsize=(plot_count * img_size, img_size))
    ax = fig.add_subplot(1, plot_count, 1)
    ax.set_aspect('equal')
    ax.set_xlim(0., traversible.shape[1] * map_scale)
    ax.set_ylim(0., traversible.shape[0] * map_scale)
    plot_topview(ax, extent, traversible, human_traversible,
                 camera_pos_13, pedestrians, robots, room_center, plot_quiver=True)
    if(len(robots) > 0 or len(pedestrians) > 0):
        ax.legend()
    time_string = "sim_t=%.3f" % sim_t + " wall_t=%.3f" % wall_t
    ax.set_title(time_string, fontsize=20)

    if with_zoom:
        # Plot the 5x5 meter occupancy grid centered around the camera
        zoom = 8.5  # zoom out in by a constant amount
        ax = fig.add_subplot(1, plot_count, zoomed_img_plt_indx)
        ax.set_xlim([room_center[0] - zoom, room_center[0] + zoom])
        ax.set_ylim([room_center[1] - zoom, room_center[1] + zoom])
        plot_topview(ax, extent, traversible, human_traversible,
                     camera_pos_13, pedestrians, robots, room_center, plot_quiver=True)
        if(len(robots) > 0 or len(pedestrians) > 0):
            ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(time_string, fontsize=20)

    if rgb_image_1mk3 is not None:
        # Plot the RGB Image
        ax = fig.add_subplot(1, plot_count, rgb_img_plt_indx)
        ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('RGB')

    if depth_image_1mk1 is not None:
        # Plot the Depth Image
        ax = fig.add_subplot(1, plot_count, depth_img_plt_indx)
        ax.imshow(depth_image_1mk1[0, :, :, 0].astype(
            np.uint8), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Depth')

    full_file_name = os.path.join(p.output_directory, filename)
    if not os.path.exists(full_file_name):
        if p.verbose_printing:
            print('\033[31m', "Failed to find:", full_file_name,
                  '\033[33m', "and therefore it will be created", '\033[0m')
        touch(full_file_name)  # Just as the bash command

    fig.savefig(full_file_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    if p.verbose_printing:
        print('\033[32m', "Successfully rendered:",
              full_file_name, '\033[0m')


def render_rgb_and_depth(r, camera_pos_13, dx_m: float, human_visible=True):
    """render the rgb and depth images from the openGL renderer

    Args:
        r: the openGL renderer object
        camera_pos_13: the 3D (x, y, theta) position of the camera
        dx_m (float): the delta_x in meters between real world and grid units
        human_visible (bool, optional): Whether or not the humans are drawn. Defaults to True.

    Returns:
        rgb_image_1mk3, depth_image_1mk1: the rgb and depth images respectively
    """
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2] / dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12,
                                      camera_pos_13[:, 2:3],
                                      human_visible=True)

    depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12,
                                                camera_pos_13[:, 2:3],
                                                xy_resolution=0.05,
                                                map_size=1500,
                                                pos_3=camera_pos_13[0, :3],
                                                human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1


def save_to_gif(IMAGES_DIR, duration=0.05, gif_filename="movie", clear_old_files=True, verbose=False):
    """Takes the image directory and naturally sorts the images into a singular movie.gif"""
    images = []
    if not os.path.exists(IMAGES_DIR):
        print('\033[31m', "ERROR: Failed to find image directory at",
              IMAGES_DIR, '\033[0m')
        os._exit(1)  # Failure condition
    files = natural_sort(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
    num_images = len(files)
    for i, filename in enumerate(files):
        if verbose:
            print("appending", filename)
        try:
            images.append(imageio.imread(filename))
        except:
            print("%sUnable to read file:" % color_red, filename,
                  "Try clearing the directory of old files and rerunning%s" % color_reset)
            exit(1)
        print("Movie progress: %d out of %d, %.3f%% \r" %
              (i + 1, num_images, 100. * ((i + 1) / num_images)), end="")
    print()
    output_location = os.path.join(IMAGES_DIR, gif_filename + ".gif")
    kargs = {'duration': duration}  # 1/fps
    imageio.mimsave(output_location, images, 'GIF', **kargs)
    print("%sRendered gif at" % color_green, output_location, '\033[0m')
    # Clearing remaining files to not affect next render
    if clear_old_files:
        for f in files:
            os.remove(f)
        print("%sCleaned directory" % color_green, '\033[0m')
