import glob
import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
from dotmap import DotMap
from matplotlib import pyplot
from socnav.socnav_renderer import SocNavRenderer

from utils.utils import color_text, mkdir_if_missing, natural_sort, touch

from enum import Enum, auto


class ImageType(Enum):
    SCHEMATIC = auto()
    RGB = auto()
    DEPTH = auto()

    def __str__(self) -> str:
        s = super().__str__()
        return s.replace("ImageType.", "")  # only the name


def plot_scene_images(
    p: DotMap,
    sim_state,  # SimState (circular dep)
    rgb_image_1mk3: np.ndarray,
    depth_image_1mk1: np.ndarray,
    filename: str,
) -> None:
    """Plots a single frame from information provided about the world state"""
    from simulators.sim_state import SimState

    assert isinstance(sim_state, SimState)

    # can customize the order of the 1 x n subplots here
    plots: List[str] = [ImageType.SCHEMATIC, ImageType.RGB, ImageType.DEPTH]

    if not p.render_3D:
        plots.remove(ImageType.RGB)
        plots.remove(ImageType.DEPTH)

    img_size: float = 10 * p.render_params.img_scale
    fig, axs = pyplot.subplots(1, len(plots), figsize=(len(plots) * img_size, img_size))
    title: str = "sim:{:.3f}s wall:{:.3f}s".format(sim_state.sim_t, sim_state.wall_t)
    # fig.suptitle(title, fontsize=20)
    if isinstance(axs, pyplot.Axes):
        axs = [axs]  # when plotting a single plot, make sure it is still a list
    for i, ax in enumerate(axs):
        plot_name: ImageType = plots[i]
        if plot_name == ImageType.SCHEMATIC:
            """PLOT TOPVIEW (SCHEMATIC)"""
            ax.set_title(str(plot_name) + " " + title, fontsize=14)
            ax.set_aspect("equal")
            sim_state.render(ax, p.render_params)
        elif plot_name == ImageType.RGB or plot_name == ImageType.DEPTH:
            """PLOT 3D RENDER"""
            assert rgb_image_1mk3 is not None and depth_image_1mk1 is not None
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(plot_name, fontsize=14)
            if plot_name == ImageType.RGB:
                ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
            else:
                ax.imshow(depth_image_1mk1[0, :, :, 0].astype(np.uint8), cmap="gray")
        else:
            raise Exception("Unknown render method: {}".format(plot_name))

    full_file_name = os.path.join(p.output_directory, filename)
    if not os.path.exists(full_file_name):
        if p.verbose_printing:
            print(
                color_text["red"],
                "Failed to find:",
                full_file_name,
                "and therefore it will be created",
                color_text["reset"],
            )
        touch(full_file_name)  # Just as the bash command
    # pyplot.subplots_adjust(top=0.85)  # for the suptitle
    pyplot.tight_layout()
    fig.savefig(full_file_name, bbox_inches="tight", pad_inches=0)
    pyplot.close()
    if p.verbose_printing:
        print(
            color_text["green"],
            "Successfully rendered:",
            full_file_name,
            color_text["reset"],
        )


def render_socnav(
    sim_state,  # SimState (circular dep)
    renderer: SocNavRenderer,
    params: DotMap,
    filename: str,
    camera_pos_13: Optional[np.ndarray] = None,
) -> None:
    from simulators.sim_state import SimState

    assert isinstance(sim_state, SimState)
    robot = None
    if len(sim_state.robots) > 0:
        robot = sim_state.get_robot()
        # overwrite camera to follow the robot
        camera_pos_13 = robot.get_current_config().position_and_heading_nk3(
            squeeze=True
        )
    else:
        # if not specified, place camera at center of room
        if camera_pos_13 is None:
            camera_pos_13 = sim_state.environment["room_center"]

    # NOTE: the rgb and depth images require the full-render
    rgb_image_1mk3: np.ndarray = None
    depth_image_1mk1: np.ndarray = None

    if params.render_3D:
        # TODO: Fix multiprocessing for properly deepcopied renderers
        # only when rendering with opengl
        assert "human_traversible" in sim_state.environment
        renderer.remove_all_humans()
        # update pedestrians humans
        renderer.update_bulk_humans(list(sim_state.pedestrians.values()))
        # renderer.load_bulk_humans
        # for a in sim_state.pedestrians.values():
        #     renderer.update_human(a)
        # Update human traversible
        # NOTE: this is technically not R-O since it modifies the human trav
        # TODO: use a separate variable to keep SimStates as R-O
        sim_state.environment["human_traversible"] = renderer.get_human_traversible()
        # compute the rgb and depth images
        rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(
            renderer,
            np.array([camera_pos_13]),
            sim_state.environment["map_scale"],
            human_visible=True,
        )

    # plot the rbg, depth, and topview images using matplotlib
    plot_scene_images(
        p=params,
        sim_state=sim_state,
        rgb_image_1mk3=rgb_image_1mk3,
        depth_image_1mk1=depth_image_1mk1,
        filename=filename,
    )


def render_rgb_and_depth(
    r: SocNavRenderer,
    camera_pos_13: np.ndarray,
    dx_m: float,
    human_visible: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """render the rgb and depth images from the openGL renderer"""
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2] / dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(
        camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=human_visible
    )

    depth_image_1mk1, _, _ = r._get_depth_image(
        camera_grid_world_pos_12,
        camera_pos_13[:, 2:3],
        xy_resolution=0.05,
        map_size=1500,
        pos_3=camera_pos_13[0, :3],
        human_visible=human_visible,
    )

    return rgb_image_1mk3, depth_image_1mk1


def save_to_gif_with_ffmpeg(
    filename: str,
    IMAGES_DIR: str,
    fps: int,
    clean_mp4: Optional[bool] = False,
    clean_pngs: Optional[bool] = False,
) -> bool:
    try:
        import subprocess

        mp4_filename: str = "{}.mp4".format(filename)
        gif_filename: str = "{}.gif".format(filename)
        all_png_files: str = os.path.join(IMAGES_DIR, "*.png")
        print("Rendering movie with ffmpeg -> mp4 -> gif")
        ffmpeg_pngs_to_mp4: str = "ffmpeg {ow} {logs} {fps} {inputs} {mp4} {q} {out}".format(
            ow="-y",  # force overwrite
            logs="-hide_banner -loglevel error",  # ignore logs
            fps="-framerate {}".format(fps),  # set framerate of resulting movie
            inputs="-pattern_type glob -i '{}'".format(all_png_files),
            mp4="-c:v mpeg4 -pix_fmt yuv420p",  # video codec
            q="-qscale 0",  # decrease the quality by 0
            out=mp4_filename,
        )
        og_dir: str = os.getcwd()  # make sure to return to this directory after rendering
        os.chdir(IMAGES_DIR)  # make sure to go to the output dir
        error_val = subprocess.call(ffmpeg_pngs_to_mp4, shell=True)
        if error_val:
            raise Exception("ffmpeg (png's -> mp4) error {}".format(error_val))
        assert os.path.exists(os.path.join(IMAGES_DIR, mp4_filename))
        print(
            "{}Rendered mp4 at {}{}".format(
                color_text["green"], mp4_filename, color_text["reset"]
            )
        )
        # successfully rendered the mp4, now converting that to gif
        # render gif
        ffmpeg_mp4_to_gif: str = "ffmpeg {ow} {logs} {conversion}".format(
            ow="-y",  # force overwrite
            logs="-hide_banner -loglevel error",  # ignore logs
            conversion="-i {} {}".format(mp4_filename, gif_filename),
        )
        error_val = subprocess.call(ffmpeg_mp4_to_gif, shell=True)
        if error_val:
            raise Exception("ffmpeg (mp4 -> gif) error {}".format(error_val))
        assert os.path.exists(os.path.join(IMAGES_DIR, gif_filename))
        print(
            "{}Rendered gif at {}{}".format(
                color_text["green"], gif_filename, color_text["reset"]
            )
        )
        if clean_mp4 and os.path.exists(os.path.join(IMAGES_DIR, mp4_filename)):
            os.remove(os.path.join(IMAGES_DIR, mp4_filename))
            print("Removed mp4 file")

        os.chdir(og_dir)
        # and we're done!
        return True
    except Exception as e:
        print(
            "{}Failed to use ffmpeg for gif creation. Defaulting to ImageIO. {}\nReason: {}".format(
                color_text["red"], color_text["reset"], e
            )
        )
        return save_to_gif_with_imageio(filename, IMAGES_DIR, fps, clean_pngs)


def save_to_gif_with_imageio(filename: str, IMAGES_DIR: str, fps: float) -> bool:
    gif_filename: str = os.path.join(IMAGES_DIR, "{}.gif".format(filename))
    files: List[str] = natural_sort(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    with imageio.get_writer(gif_filename, mode="I", fps=fps) as writer:
        for i, png_filename in enumerate(files):
            try:
                image: np.ndarray = imageio.imread(png_filename).astype(np.uint8)
                writer.append_data(image)
            except Exception as e:
                print(
                    "{}Unable to read file:{}{} Reason:{}".format(
                        color_text["red"], png_filename, color_text["reset"], e
                    )
                )
                print(
                    "{}Try clearing the directory of old files and rerunning{}".format(
                        color_text["yellow"], color_text["reset"]
                    )
                )
                raise e
            print(
                "Movie progress: {}/{} ({:.2f}%)\r".format(
                    i + 1, len(files), 100.0 * ((i + 1) / len(files))
                ),
                end="",
            )
        writer.close()
    print()  # newline for carriage return print
    print(
        "{}Rendered gif at {}{}".format(
            color_text["green"], gif_filename, color_text["reset"]
        )
    )
    return True


def save_to_gif(
    IMAGES_DIR: str,
    fps: Optional[float] = 20.0,
    filename: Optional[str] = "movie",
    clear_old_files: Optional[bool] = True,
    use_ffmpeg: Optional[bool] = True,
    clear_mp4: Optional[bool] = False,
) -> None:
    """Takes the image directory and naturally sorts the images into a singular movie.gif"""
    if not os.path.exists(IMAGES_DIR):
        mkdir_if_missing(IMAGES_DIR)
    # in the future we may consider using https://github.com/kkroening/ffmpeg-python
    # NOTE: ffmpeg produces a nice mp4, but worse gif, imageio produces a nicer gif
    if use_ffmpeg:
        save_to_gif_with_ffmpeg(filename, IMAGES_DIR, fps, clear_mp4)
    else:
        save_to_gif_with_imageio(filename, IMAGES_DIR, fps)

    files: List[str] = natural_sort(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if clear_old_files:
        for f in files:
            os.remove(f)
        print("%sCleaned directory" % color_text["green"], color_text["reset"])

