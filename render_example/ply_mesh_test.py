import json
import os
import pathlib
from ast import Str
from collections import namedtuple
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from iopath.common.file_io import PathManager
from pytorch3d.io.ply_io import _load_ply_raw
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes

from render.config import get_cfg_defaults
from render.render import get_camera, get_mesh_renderer
from render.utils import (
    get_device,
    get_viewpoint_info,
    load_viewpoints_dict,
    read_gz_jsonlines,
    read_image,
    save_images_to_one,
)

# prepare params
scan = "r1Q1Z4BcV1o"
split = "val_seen"
configs = get_cfg_defaults()
configs.merge_from_file("render_example/configs/ply_mesh_test.yaml")
data_path = pathlib.Path("/mnt/sw/data/vln")
val_seen = read_gz_jsonlines(data_path / "traj/data/RxR/rxr_val_seen_guide.jsonl.gz")
image_folder = os.path.join(configs.SAVE.IMAGE_DIR, split)

with open("./render_example/save/poses/val_seen_scan2vpspose.json") as f:
    scan2vpspose = json.load(f)
_, _, scan_to_vps_to_data = load_viewpoints_dict(configs.DATA.CONNECTIVITY_DIR)
# load ply mesh
path_manager = PathManager()
f = "/mnt/sw/data/vln/test/r1Q1Z4BcV1o/house_segmentations/r1Q1Z4BcV1o.ply"
header, elements = _load_ply_raw(f, path_manager=path_manager)
# face = elements.get("face", None)
# z = [[x[0], x[2]] for x in face]
# len(z)

# create ply mesh
vertex = elements["vertex"][0][:, :3]
vertex_torch = torch.tensor(vertex.copy())
del vertex
face = np.vstack([item[0] for item in elements["face"]])
face = torch.tensor(face.copy())
texture = torch.tensor(elements["vertex"][1]).unsqueeze(0) / 255.0
textureV = TexturesVertex(texture)
device = get_device()
mesh = Meshes(
    verts=[vertex_torch.to(device)],
    faces=[face.to(device)],
    textures=textureV.to(device),
)


# get obj mesh
renderer, obj_mesh = get_mesh_renderer(configs, scan)

test_render_set = scan2vpspose[scan]
vp = test_render_set[40]["vp"]
heading = test_render_set[40]["heading"]
elevation = test_render_set[40]["elevation"]
ori_img = read_image(
    scan, vp, heading, elevation, display=False, image_folder=image_folder
)
heading = np.array(heading).reshape(-1, 1)
elevation = np.array(elevation).reshape(-1, 1)
viewpoint_info = get_viewpoint_info(scan, vp, scan_to_vps_to_data)


location = namedtuple("location", ["x", "y", "z"])
loc = location(*viewpoint_info["location"])
device = get_device()
camera = get_camera(configs, loc, heading, elevation, device=device)


ply_img = renderer(meshes_world=mesh, cameras=camera)
obj_img = renderer(meshes_world=obj_mesh, cameras=camera)

with torch.no_grad():
    ply_img = ply_img[..., :3].cpu().detach().squeeze().numpy()
    obj_img = obj_img[..., :3].cpu().detach().squeeze().numpy()
    ori_img = np.array(ori_img) / 255.0

file_name = os.path.splitext(os.path.basename(__file__))[0]
save_dir = os.path.join("render_example/save", file_name)

save_images_to_one(
    [ori_img, ply_img, obj_img],
    title=["ori", "ply", f"obj{configs.MESH.TEXTURE_ATLAS_SIZE}"],
    filename=f"{scan}_{vp}_{heading}_{elevation}.png",
    save_dir=save_dir,
)
# save obj rendered image, ply image, and origianal to the same folder
