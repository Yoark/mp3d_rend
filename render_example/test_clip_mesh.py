# imports
# load a pose
import json
import os
import resource

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    AmbientLights,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.clip import ClipFrustum, clip_faces
from pytorch3d.renderer.mesh.textures import TexturesAtlas
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from render.config import cfg
from render.render import get_render_params, init_episode, load_meshes
from render.utils import get_device, get_viewpoint_info, load_viewpoints_dict


def compare_two_mesh(
    mesh1, mesh2, renderer, save_dir="render_example/save/tmp", save=False
):
    plt.figure(figsize=(20, 10))
    img1 = renderer(mesh1)
    # img2 = renderer(mesh2)
    ori_img = img1.clone()
    ori_img = ori_img[0, ..., :3].cpu().detach().numpy()
    plt.subplot(1, 2, 1)
    plt.title("ori")
    plt.imshow(ori_img)

    # plt.subplot(1, 2, 2)
    # clip_img = img2.clone()
    # clip_img = clip_img[0, ..., :3].cpu().detach().numpy()
    # plt.imshow(clip_img)
    # plt.title("clip")
    # plt.axis("off")
    plt.savefig(f"{save_dir}/compare.png")


def read_image(scan, vp, heading, elevation):
    # find image and display for compare
    img_path = os.path.join(
        cfg.SAVE.IMAGE_DIR, scan, f"{scan}_{vp}_{heading}_{elevation}.png"
    )
    img = Image.open(img_path)
    print("Image size:", img.size)
    img.show()
    return img


# load related data.
cfg.merge_from_file("render_example/configs/mp3d_render.yaml")

scan_ids = [
    "2azQ1b91cZZ",
    "8194nk5LbLH",
    "EU6Fwq7SyZv",
    "oLBMNvg9in8",
    "pLe4wQe7qrG",
    "QUCTc6BB5sX",
    "TbHJrupSAjP",
    "X7HyMhZNoso",
    "x8F5xyUWy9e",
    "Z6MFQCViBuw",
    "zsNo4HB9uLZ",
]

# with open("./render_example/save/poses/val_unseen_scan2vpspose.json") as f:
with open(cfg.DATA.POSE) as f:
    scan2vpspose = json.load(f)

_, _, scan_to_vps_to_data = load_viewpoints_dict(cfg.DATA.CONNECTIVITY_DIR)

scan = scan_ids[-3]
test_render_set = scan2vpspose[scan]
vp = test_render_set[0]["vp"]
heading = test_render_set[0]["heading"]
elevation = test_render_set[0]["elevation"]

device = get_device()
mesh_data = load_meshes(
    [scan],
    mesh_dir=cfg.DATA.MESH_DIR,
    device=device,
    texture_atlas_size=cfg.MESH.TEXTURE_ATLAS_SIZE,
)
viewpoint_info = get_viewpoint_info(scan, vp, scan_to_vps_to_data)

# get original mesh
# -------------------------------------------
verts, faces, textures = mesh_data[scan]

textures = textures.to(device)
verts = verts.to(device)
faces = faces.to(device)

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures,
)
# -------------------------------------------
raster_settings = RasterizationSettings(
    image_size=((cfg.CAMERA.HEIGHT, cfg.CAMERA.WIDTH)),
    blur_radius=0.0,
    faces_per_pixel=1,
    # cull_to_frustum=True,
)

# set lights
ambient_color = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
light = AmbientLights(device=device, ambient_color=ambient_color[None, :])
# set up renderer and intial view
pose = viewpoint_info["pose"]
render_params = get_render_params(
    pose,
    heading,
    elevation,
    raster_settings=raster_settings,
    cfg=cfg,
    device=device,
)
# get renderer
camera = render_params["camera"]
raster_settings = render_params["raster_settings"]
device = render_params["device"]
light = render_params["light"]

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=camera, lights=light),
)
# -------------------------------------------
# set up frustum
# fov_y = cfg.CAMERA.HFOV
# fov_x = cfg.CAMERA.VFOV
# near = 0
# far = 100
# left = -1
# right = 1
# top = -1
# bottom = 1
# z_clip_value = 0.1
# cull = True
# perspective_correct = True

# clip_frustum = ClipFrustum(
#     left=left,
#     right=right,
#     top=top,
#     bottom=bottom,
#     znear=near,
#     zfar=far,
#     cull=cull,
#     perspective_correct=perspective_correct,
#     z_clip_value=z_clip_value,
# )
# # -------------------------------------------

# _, _, texture = mesh_data[scan]  # (V, 3), (F, 3)

# face_packed = mesh.faces_packed()
# verts_packed = mesh.verts_packed()
# face_verts = verts_packed[face_packed]


# num_faces_per_mesh = mesh.num_faces_per_mesh()
# mesh_to_face_first_idx = mesh.mesh_to_faces_packed_first_idx()

# clipped_faces = clip_faces(
#     face_verts, mesh_to_face_first_idx, num_faces_per_mesh, clip_frustum
# )


# clip_face_to_verts = torch.index_select(face_packed, 0, clipped_faces.faces_clipped_to_unclipped_idx)  # type: ignore
# print(f"original texture atlas shape: {texture.atlas_packed().shape}")
# clipped_texture = torch.index_select(texture.atlas_packed(), 0, clipped_faces.faces_clipped_to_unclipped_idx)  # type: ignore

# clipped_atlas = TexturesAtlas(atlas=clipped_texture.unsqueeze(0))

# print(
#     f"verts packed shape{str(verts_packed.shape):>40}\nclipped face to verts:{str(clip_face_to_verts.shape):>40}"
# )

# clipped_mesh = Meshes(
#     verts=verts_packed.unsqueeze(0),
#     faces=clip_face_to_verts.unsqueeze(0),
#     textures=clipped_atlas,
# )


compare_two_mesh(mesh, mesh, renderer, save=True)
