# load multiple meshes and viewpoints and render at a time, profiling the speed of rendering

import os
import random

import numpy as np
import torch
from ipdb import launch_ipdb_on_exception
from matplotlib import pyplot as plt
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_batch

from render.config import cfg
from render.render import get_device, load_meshes, load_viewpoints_dict


def main():
    # load scan_viewpoints dict
    connectivity_dir = cfg.DATA.CONNECTIVITY_DIR
    _, _, scan_to_vps_to_data = load_viewpoints_dict(connectivity_dir)
    # scan_to_vp_list = load_viewpoints(connectivity_dir)

    with open(os.path.join(connectivity_dir, "scans.txt")) as f:
        scans = [x.strip() for x in f]

    # select 10 random scans and select 5 random viewpoints from each to render
    NUM_SCAN = 1
    NUM_VIEWPOINTS = 10
    scans = scans[:NUM_SCAN]
    scan_viewpoint_pairs = []
    for scan in scans:
        # select 5 random scans
        random_scans = random.sample(
            list(scan_to_vps_to_data[scan].items()), NUM_VIEWPOINTS
        )
        scan_viewpoint_pairs.extend(
            [(scan, vp_id, vp_data) for vp_id, vp_data in random_scans]
        )

    # set device
    device = get_device()
    # render each scan_viewpoint pair and compute the time required Batched vs no batch?

    # TODO compute the time loading meshes and viewpoints
    # TODO can we batch atlas textures?
    mesh_dict = load_meshes(scans, mesh_dir=cfg.DATA.MESH_DIR, device=device)

    viewpoints_infos = []
    # n_verts, n_face_verts_index, n_textures = [], [], []
    meshes = []
    locations = []

    for scan, viewpoint, viewpoint_data in scan_viewpoint_pairs:
        # viewpoints_infos.append(
        #     get_viewpoint_info(scan, viewpoint, scan_to_vps_to_data)
        # )
        verts, face_vert_idx, texture = mesh_dict[scan]
        # n_verts.append(verts)
        # n_face_verts_index.append(face_vert_idx)
        # n_textures.append(texture)
        mesh = Meshes(verts=[verts], faces=[face_vert_idx], textures=texture).to(device)
        meshes.append(mesh)

        pose = viewpoint_data["pose"]
        locations.append([pose[3], pose[7], pose[11]])

    # TODO check if we need to set include_textures to True
    meshes = join_meshes_as_batch(meshes)

    # meshes = {k: Meshes(*v) for k, v in mesh_dict.items()}

    raster_settings = RasterizationSettings(
        image_size=((cfg.CAMERA.HEIGHT, cfg.CAMERA.WIDTH)),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # init_heading = heading
    # init_heading = np.deg2rad(0)
    # init_elevation = np.deg2rad(0)

    # test rendering
    # if args.render_type == "image":
    # TODO update code to only update camera pose
    # plt.figure(figsize=(4, 3))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.savefig(args.save_dir / "image.png", bbox_inches="tight")
    # create a temp folder to store the rendered images

    # -----

    # pose = viewpoint_data["pose"]
    # get_list_of poses
    locations = torch.tensor(
        locations,
        dtype=torch.float32,
        device=device,
    ).view((-1, 3))
    at = locations.clone()
    at[:, 1] += 1
    up = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(
        (len(scan_viewpoint_pairs), 1)
    )

    # eye_r, at_r, up_r = rotate_heading_elevation(eye, at, up, heading, elevation)
    R, T = look_at_view_transform(eye=locations, up=up, at=at)
    # TODO set znear and zfar to be the same as mp3d
    # init camera
    camera = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        aspect_ratio=cfg.RENDER.PIXEL_ASPECT_RATIO,
        fov=cfg.CAMERA.HFOV,
    )
    # use ambient lighting
    # ambient = AmbientLights(device=device, ambient_color=((1, 1, 1),))
    # point_light = PointLights(location=eye_r, device=device)
    # init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
        ),
    )

    # TODO render multiple meshes at a time
    import time

    render_start = time.time()
    # for i , (scan, _, _) in enumerate(scan_viewpoint_pairs):
    # mesh = meshes[scan]
    images = renderer(meshes_world=meshes, cameras=camera)
    render_end = time.time()

    print(
        f"Time to render {len(scan_viewpoint_pairs)} viewpoints: {render_end-render_start}, avg is {(render_end-render_start)/len(scan_viewpoint_pairs)}"
    )

    if not os.path.exists("temp"):
        os.makedirs("temp")

    for i, image in enumerate(images):
        plt.figure(figsize=(4, 3))
        plt.imshow(image[..., :3].cpu().numpy())
        plt.axis("off")
        plt.savefig(f"temp/image_{i}.png", bbox_inches="tight")

    plt.figure(figsize=(4, 3))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
