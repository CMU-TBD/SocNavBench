#ifndef ENV_H
#define ENV_H

#include <string>
#include <vector>

using namespace std;

class Environment
{
private:
    Environment(vector<vector<int>> &building,
                vector<vector<int>> &human,
                vector<int> &center, float scale)
    {
        dx_scale = scale;
        building_traversible = building;
        human_traversible = building;
        room_center = center;
    }
    float dx_scale;
    vector<int> &room_center;
    vector<vector<int>> &building_traversible;
    vector<vector<int>> &human_traversible;

public:
    float get_dx_scale() const { return dx_scale; }
    vector<int> get_room_center() const { return room_center; }
    vector<vector<int>> get_building_traversible() const
    {
        return building_traversible;
    }
    vector<vector<int>> get_human_traversible() const
    {
        return human_traversible;
    }
};

#endif
