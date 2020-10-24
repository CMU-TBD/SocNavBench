import os

base_dir = 'human_meshes'
expected_meshes_filename = "txt/expected_human_meshes.txt"

def get_human_mesh_filenames():
    """
    Returns the relative path to all the meshes in
    base_dir
    """
    humanav_file_names = []

    velocity_dirs = os.listdir(base_dir)
    for velocity_dir in velocity_dirs:
        full_velocity_dir = os.path.join(base_dir, velocity_dir)
        if os.path.isdir(full_velocity_dir):
            pose_dirs = os.listdir(full_velocity_dir)
            for pose_dir in pose_dirs:
                full_pose_dir = os.path.join(full_velocity_dir, pose_dir)
                if os.path.isdir(full_pose_dir):
                    body_shape_dirs = os.listdir(full_pose_dir)
                    for body_shape_dir in body_shape_dirs:
                        full_body_shape_dir = os.path.join(full_pose_dir, body_shape_dir)
                        if os.path.isdir(full_body_shape_dir):
                            gender_dirs = os.listdir(full_body_shape_dir)
                            for gender_dir in gender_dirs:
                                full_gender_dir = os.path.join(full_body_shape_dir, gender_dir)
                                if os.path.isdir(full_gender_dir):
                                    human_files = os.listdir(full_gender_dir)
                                    human_mesh_files = [os.path.join(velocity_dir, pose_dir, body_shape_dir, gender_dir, x) for x in human_files if '.obj' in x]

                                    humanav_file_names.extend(human_mesh_files)

    return humanav_file_names


def verify_human_mesh_generation():
    """
    Check that the mesh generation process resulted in the
    expected dataset.
    """
    with open(expected_meshes_filename, 'r') as f:
        expected_meshes_str = f.read()

    expected_meshes = set(expected_meshes_str.split('\n'))

    actual_meshes = set(get_human_mesh_filenames())

    # Make sure we have all the expected meshes
    # Needed for the HumANav dataset
    assert expected_meshes.issubset(actual_meshes), 'Created dataset does not have all the expected files'

    # Make sure there are no extra files
    assert len(expected_meshes) == len(actual_meshes), 'Created dataset has extra files. Please delete them'


if __name__ == '__main__':
    verify_human_mesh_generation()
