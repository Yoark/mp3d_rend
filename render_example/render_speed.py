# load multiple meshes and viewpoints and render at a time, profiling the speed of rendering

import os
import random

from render.mp3d_render_params import *
from render.render import get_device, load_viewpoints_dict

if __name__ == "__main__":
    # load scan_viewpoints dict
    connectivity_dir = "/mnt/sw/vln/data/matterport3d/connectivity"
    _, _, scan_to_vps_to_data = load_viewpoints_dict(connectivity_dir)
    # scan_to_vp_list = load_viewpoints(connectivity_dir)

    with open(os.path.join(connectivity_dir, "scans.txt")) as f:
        scans = [x.strip() for x in f]

    # select 10 random scans and select 5 random viewpoints from each to render
    NUM_SCAN = 10
    NUM_VIEWPOINTS = 5
    scans = scans[:NUM_SCAN]
    scan_viewpoint_pairs = []
    for scan in scans:
        # select 5 random scans
        random_scans = random.sample(scan_to_vps_to_data[scan].items(), NUM_VIEWPOINTS)
        scan_viewpoint_pairs.extend(
            [(scan, vp_id, vp_data) for vp_id, vp_data in random_scans]
        )

    # set device
    device = get_device()
    # render each scan_viewpoint pair and compute the time required Batched vs no batch?
