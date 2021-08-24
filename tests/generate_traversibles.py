from utils.socnav_utils import load_building

from test_episodes import create_params


def regenerate_all_traversibles() -> None:
    # add custom maps here
    maps = ["DoubleHotel", "ETH", "Hotel", "Univ", "Zara"]
    p = create_params()
    for m in maps:
        p.building_params.building_name = m
        load_building(p, force_rebuild=True)


if __name__ == "__main__":
    regenerate_all_traversibles()
