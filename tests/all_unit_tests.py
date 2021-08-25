from unit_tests.test_coordinate_transform import main_test as test_coordinate_transform
from unit_tests.test_cost_function import main_test as test_cost_function
from unit_tests.test_costs import main_test as test_cost
from unit_tests.test_dynamics import main_test as test_dynamics
from unit_tests.test_fmm_map import main_test as test_fmm_map
from unit_tests.test_goal_angle_objective import main_test as test_goal_angle
from unit_tests.test_goal_distance_objective import main_test as test_goal_distance
from unit_tests.test_image_space_grid import main_test as test_image_space_grid
from unit_tests.test_lqr import main_test as test_lqr
from unit_tests.test_obstacle_map import main_test as test_obstacle_map
from unit_tests.test_obstacle_objective import main_test as test_obstacle_objective
from unit_tests.test_spline import main_test as test_spline
from unit_tests.test_voxel_interpolation import main_test as test_voxel_interpolation
from unit_tests.test_personal_cost import main_test as test_goal_psc
from utils.utils import color_text

if __name__ == "__main__":
    test_coordinate_transform()
    test_cost_function()
    test_cost()
    test_dynamics()
    test_fmm_map()
    test_goal_angle()
    test_goal_distance()
    test_goal_psc()
    test_image_space_grid()
    test_lqr()
    test_obstacle_map()
    test_obstacle_objective()
    test_spline()
    test_voxel_interpolation()
    print("%s\nAll tests passed!%s" % (color_text["green"], color_text["reset"]))
