import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    look_at_view_transform,
)

from render.config import cfg, get_cfg_defaults
from render.render import (
    CameraData,
    init_episode,
    load_meshes,
    rotate_heading_elevation,
)
from render.utils import get_viewpoint_info, load_viewpoints_dict, read_gz_jsonlines


# DONE use config file to set up the parameters
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_type", type=str, default="image", choices=["panorama", "image"]
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path(
            "/home/zijiao/research/pytorch_rend/render_example/rendered_sample"
        ),
    )
    args = parser.parse_args()
    print(args)

    # set up config
    # cfg = get_cfg_defaults()
    cfg.merge_from_file("render_example/configs/test.yaml")
    cfg.freeze()
    # setting up the correct meta parameters
    # MESH_DIR = "/mnt/sw/vln/data/matterport3d/mp3d_mesh/v1/scans/{}/matterport_mesh"
    # VFOV = 60
    # WIDTH = 640
    # HEIGHT = 480
    # FOV = VFOV * WIDTH / HEIGHT
    # HEADINGS = [np.deg2rad(30.0 * h) for h in range(12)]
    # ELEVATIONS = [np.deg2rad(e) for e in [-30.0, 0, 30]]
    # pixel_aspect_ratio = 1  # pixel aspect
    # image_aspect_ratio = 4 / 3  # image aspect
    # image_size = (WIDTH, HEIGHT)

    # set devic
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # use one data sample
    # rxr_data_dir = "/mnt/sw/vln/data/RxR/rxr_train_guide.jsonl.gz"
    train_data = read_gz_jsonlines(cfg.DATA.RXR_DIR)
    # load connectivity info
    _, _, scan_to_vps_to_data = load_viewpoints_dict(cfg.DATA.CONNECTIVITY_DIR)
    scan = train_data[0]["scan"]
    viewpoint = train_data[0]["path"][0]
    heading = train_data[0]["heading"]
    # scan = "SN83YJsR3w2"
    # viewpointid = "4471fcf26b3847ed88ce41eca5ecb13d"
    # load mesh and camera info
    mesh_data = load_meshes([scan], mesh_dir=cfg.DATA.MESH_DIR, device=device)
    viewpoint_info = get_viewpoint_info(scan, viewpoint, scan_to_vps_to_data)
    # cam_info = CameraData(viewpoint_info["pose"])

    # set up meta data
    # render settings
    raster_settings = RasterizationSettings(
        image_size=((cfg.CAMERA.HEIGHT, cfg.CAMERA.WIDTH)),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # set up renderer and intial view
    # init_heading = 0
    init_heading = heading
    init_elevation = np.deg2rad(0)
    renderer_dict = init_episode(
        viewpoint_info,
        init_heading,
        init_elevation,
        raster_settings=raster_settings,
        cfg=cfg,
    )
    # test rendering
    if args.render_type == "image":
        # TODO update code to only update camera pose
        images = renderer_dict["renderer"](
            mesh_data[scan], cameras=renderer_dict["camera"]
        )
        plt.figure(figsize=(4, 3))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.savefig(args.save_dir / "image.png", bbox_inches="tight")
        # init_heading = train_data[0]["heading"]

    elif args.render_type == "panorama":
        rows = 3
        cols = 12
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

        eye_init = renderer_dict["eye"]
        up_init = renderer_dict["up"]
        at_init = renderer_dict["at"]

        for e, elevation in enumerate(cfg.MP3D.ELEVATIONS):
            for h, heading in enumerate(cfg.MP3D.HEADINGS):
                eye, at, up = rotate_heading_elevation(
                    eye_init, at_init, up_init, heading, elevation
                )

                R, T = look_at_view_transform(eye=[eye], up=[up], at=[at])
                cameras = FoVPerspectiveCameras(
                    device=device, R=R, T=T, aspect_ratio=1, fov=80.0
                )
                images = renderer_dict["renderer"](mesh_data[scan], cameras=cameras)

                axes[e, h].imshow(images[0, ..., :3].cpu().numpy())
                axes[e, h].axis("off")
        plt.savefig(args.save_dir / f"panorama.png", bbox_inches="tight")


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
