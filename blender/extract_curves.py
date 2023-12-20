# Extract Bezier curves from Blender scene and save it to a json file

import bpy
import os
import json
import argparse


def save_curves(output_path):
    # Check output path
    if os.path.isfile(output_path):
        raise ValueError(f"File '{output_path}' already exists")

    data = dict()
    for scene_obj in  bpy.context.scene.objects:
        if scene_obj.type == "CURVE":
            print(f"Getting curve '{scene_obj.name}'...")
            if scene_obj.name in data.keys():
                print("WARNING: multiple curves have the same name")
            world_matrix = scene_obj.matrix_world  # From local to global coordinates
            data[scene_obj.name] = []
            for waypoint in scene_obj.data.splines[0].bezier_points:  # Take first spline only
                co = world_matrix @ waypoint.co
                handle_left = world_matrix @ waypoint.handle_left
                handle_right = world_matrix @ waypoint.handle_right
                data[scene_obj.name].append({
                    "point": [round(co.x, 3), round(co.y, 3), round(co.z, 3)],
                    "prev_handle": [round(handle_left.x, 3), round(handle_left.y, 3), round(handle_left.z, 3)],
                    "next_handle": [round(handle_right.x, 3), round(handle_right.y, 3), round(handle_right.z, 3)],
                })

    # Save curve
    print(f"Saving curves...")
    with open(output_path, "w") as file:
        json.dump(data, file, cls=json.JSONEncoder)
    print("...Done !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract Bezier curves from Blender scene")
    parser.add_argument("--output-path", dest="output_path", type=str)
    args, _ = parser.parse_known_args()
    save_curves(args.output_path)
