import os
import json


def main():
    # gym env
    cfg_path = "./cfg/RL/"
    try:
        os.makedirs(cfg_path)
    except FileExistsError:
        print("Config path exists")
    # config 1
    config = {
        "map": "Town04",
        "num_cam": 4,
        "num_frame": 400,
        "map_expand": 40,
        "cam_x": 1920,
        "cam_y": 1080,
        "cam_fov": 60,
        "cam_pos_lst": [
            (208.530151, -237.021317, 2.802895),
            (192.174973, -256.515656, 2.724678),
            (210.715622, -253.989151, 2.371706),
            (191.716537, -238.320877, 2.265693),
        ],
        "cam_dir_lst": [
            (-5.742434, -119.260330, 0.0),
            (-3.791772, 45.109650, 0.0),
            (-3.856168, 146.016495, 0.0),
            (-9.010857, -41.768852, 0.0),
        ],
        "spawn_count": 25,
        "spawn_area": (191.5, 211.5, -256.85, -236.85, 2, 3),
        "env_action_space": "pitch-yaw",
    }
    with open(os.path.join(cfg_path, "1.cfg"), "w") as fp:
        json.dump(config, fp, indent=4)

    # test config
    config = {
        "map": "Town04",
        "num_cam": 4,
        "map_expand": 40,
        "cam_x": 1920,
        "cam_y": 1080,
        "cam_fov": 60,
        "cam_pos_lst": [
            (208.530151, -235.021317, 2.802895),
            (192.174973, -256.515656, 2.724678),
            (212.715622, -253.989151, 2.371706),
            (191.716537, -238.320877, 2.265693),
        ],
        "cam_dir_lst": [
            (-5.742434, -119.260330, 0.0),
            (-3.791772, 45.109650, 0.0),
            (-3.856168, 146.016495, 0.0),
            (-9.010857, -41.768852, 0.0),
        ],
        "spawn_count": 25,
        "spawn_area": (191.5, 211.5, -256.85, -236.85, 2, 3),
        "num_frame": 4,
        "env_action_space": "x-y-z-pitch-yaw-roll-fov",
    }
    with open(os.path.join(cfg_path, "test.cfg"), "w") as fp:
        json.dump(config, fp, indent=4)


if __name__ == "__main__":
    main()
