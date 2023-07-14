import json
import os

import h5py
import MatterSim
import numpy as np
import torch
import tqdm
from PIL import Image

from render.config import get_cfg_defaults
from render.matterport_utils import build_simulator
from render.utils import (
    build_feature_extractor,
    create_folder,
    get_device,
    read_gz_jsonlines,
)

# use MatterSim to initialize a view, make some actions (turn 360), render with pytorch3d
# TODO
configs = get_cfg_defaults()
configs.merge_from_file("render_example/configs/mp3d_render.yaml")

configs.freeze()
print(configs)
# load scan and viewpoint for val_unseen
connectivity_dir = configs.DATA.CONNECTIVITY_DIR
scan_dir = configs.DATA.MESH_DIR
image_dir = configs.DATA.MATTERPORT_IMAGE_DIR
val_unseen_dir = configs.DATA.RXR.VAL_UNSEEN

# load rxr_val_unseen_guide.jsonl.gz
rxr_val_unseen = read_gz_jsonlines(val_unseen_dir)

scan_vps = set()
for item in rxr_val_unseen:
    scan = item["scan"]
    viewpoints = item["path"]
    scan_vps.update([(scan, vp) for vp in viewpoints])

sim = build_simulator(connectivity_dir, image_dir)

device = get_device()

# TODO don't need this for now, just get images first
# model, transform = build_feature_extractor(
#     configs.RENDER.IMAGE_MODEL_NAME, configs.RENDER.IMAGE_MODEL_PATH, device=device
# )

torch.set_grad_enabled(False)

image_folder = configs.SAVE.IMAGE_DIR
save_folder = create_folder(image_folder)

if configs.TEST == True:
    scan_vps = list(scan_vps)


scan_vp_cam_poses = []

pbar = tqdm.tqdm(scan_vps, desc="Progress of scans:")

for scan, vp in pbar:
    for ix in range(configs.MP3D.VIEWPOINT_SIZE):
        if ix == 0:
            sim.newEpisode([scan], [vp], [0], [np.deg2rad(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])
        state = sim.getState()[0]
        assert state.viewIndex == ix

        # save the rgb as image here for checking
        image = np.array(state.rgb, copy=True)
        image = Image.fromarray(image[:, :, ::-1])

        scan_folder = create_folder(os.path.join(save_folder, scan))
        image_path = os.path.join(
            scan_folder, f"{scan}_{vp}_{state.heading}_{state.elevation}.png"
        )
        #! normalize and convert to tensor input for feature extractor

        image.save(image_path)

        scan_vp_cam_poses.append(
            (scan, vp, state.viewIndex, state.heading, state.elevation)
        )

# save scan_vp_cam_poses
#  TODO render here


data = {
    "VFOV": configs.CAMERA.VFOV,
    "resolution": [configs.CAMERA.WIDTH, configs.CAMERA.HEIGHT],
    "poses": [
        {
            "scan": scan,
            "vp": vp,
            "viewIndex": viewIndex,
            "heading": heading,
            "elevation": elevation,
        }
        for scan, vp, viewIndex, heading, elevation in scan_vp_cam_poses
    ],
}

# save to json
_ = create_folder(configs.SAVE.VAL_UNSEEN_POSE)

with open(os.path.join(configs.SAVE.VAL_UNSEEN_POSE, "val_unseen_pose.json"), "w") as f:
    json.dump(data, f)
    print(f"Save poses to {configs.SAVE.VAL_UNSEEN_POSE}/val_unseen_pose.json")
