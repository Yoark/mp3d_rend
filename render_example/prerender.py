import json
import os

import h5py
import ipdb
import MatterSim
import numpy as np
import torch
import tqdm
from PIL import Image
from pytorch3d.renderer import AmbientLights, RasterizationSettings
from requests import get

from render.config import get_cfg_defaults
from render.matterport_utils import build_simulator
from render.render import (
    create_and_expand_mesh,
    get_camera,
    get_mesh_data,
    get_mesh_renderer,
    get_render,
    get_render_params,
)
from render.utils import (
    build_feature_extractor,
    create_folder,
    get_device,
    read_gz_jsonlines,
)


def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, "scans.txt")) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, "%s_connectivity.json" % scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x["image_id"]) for x in data if x["included"]])
    print("Loaded %d viewpoints" % len(viewpoint_ids))
    return viewpoint_ids


def main():
    # use MatterSim to initialize a view, make some actions (turn 360), render with pytorch3d
    configs = get_cfg_defaults()
    configs.merge_from_file("render_example/configs/mp3d_pre.yaml")
    configs.freeze()
    print(configs)
    # load scan and viewpoint for val_unseen
    connectivity_dir = configs.DATA.CONNECTIVITY_DIR
    scan_dir = configs.DATA.MESH_DIR
    image_dir = configs.DATA.MATTERPORT_IMAGE_DIR
    # val_unseen_dir = configs.DATA.RXR.VAL_UNSEEN

    # load rxr_val_unseen_guide.jsonl.gz
    # rxr_val_unseen = read_gz_jsonlines(val_unseen_dir)

    # scan_vps = set()
    # for item in rxr_val_unseen:
    #     scan = item["scan"]
    #     viewpoints = item["path"]
    #     scan_vps.update([(scan, vp) for vp in viewpoints])
    scan_vps = load_viewpoint_ids(connectivity_dir)

    sim = build_simulator(connectivity_dir, image_dir)
    device = get_device()
    # TODO don't need this for now, just get images first
    # model, transform = build_feature_extractor(
    #     configs.RENDER.IMAGE_MODEL_NAME, configs.RENDER.IMAGE_MODEL_PATH, device=device
    # )

    torch.set_grad_enabled(False)

    # set save folder
    image_folder = configs.SAVE.IMAGE_DIR
    save_folder = create_folder(image_folder)

    # if configs.TEST == True:
    #     scan_vps = list(scan_vps)
    raster_settings = RasterizationSettings(
        image_size=((configs.CAMERA.HEIGHT, configs.CAMERA.WIDTH)),
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_to_frustum=True
        # cull_backfaces=True,
    )
    ambient_color = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)[None, :]
    light = AmbientLights(device=device, ambient_color=ambient_color)

    # scan_vps = [("1pXnuDYAj8r", "49b4f59afc74417d846ad8cf1634e3d8")]
    scan_vp_cam_poses = []

    # get_render
    renderer = get_render(configs, raster_settings, device, light)

    pbar = tqdm.tqdm(scan_vps, desc="Progress of scans:")

    # the render number per round
    round_num = 4

    for scan, vp in pbar:
        headings = []
        elevations = []
        locations = []
        ixs = []
        scanvp_path = create_folder(os.path.join(save_folder, scan, vp))
        faces, verts, atlas = get_mesh_data(configs, scan, device)

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
            # ipdb.set_trace()

            headings.append(state.heading)
            elevations.append(state.elevation)
            ixs.append(ix)
            # import ipdb

            # ipdb.set_trace()
            if (ix + 1) % round_num == 0:
                print(ix)
                pose = state.location.x, state.location.y, state.location.z
                # renderer, mesh = get_mesh_renderer(configs, scan, expand=round_num)

                mesh = create_and_expand_mesh(
                    faces, verts, atlas, device=device, expand_num=round_num
                )

                # ipdb.set_trace()
                render_params = get_render_params(
                    pose,
                    headings,
                    elevations,
                    cfg=configs,
                    device=device,
                    raster_settings=raster_settings,
                )

                cameras = render_params["cameras"]
                # ipdb.set_trace()
                images = renderer(mesh, cameras=cameras)
                # save image by scan_vp_ix

                images = images[..., :3].cpu().detach().numpy()

                for i, ix in enumerate(ixs):
                    image_path = os.path.join(scanvp_path, f"{ix}.png")
                    image = images[i]
                    # Convert numpy array to PIL Image
                    image_np = image.cpu().numpy()
                    image_clipped = np.clip(image_np, 0, 1)
                    image_uint8 = (image_clipped * 255).round().astype(np.uint8)
                    # Save the image
                    im = Image.fromarray(image_uint8)
                    im.save(image_path)
                headings = []
                elevations = []
                ixs = []
            # feature extractor?


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
# save scan_vp_cam_poses
#  TODO render here


# data = {
#     "VFOV": configs.CAMERA.VFOV,
#     "resolution": [configs.CAMERA.WIDTH, configs.CAMERA.HEIGHT],
#     "poses": [
#         {
#             "scan": scan,
#             "vp": vp,
#             "viewIndex": viewIndex,
#             "heading": heading,
#             "elevation": elevation,
#         }
#         for scan, vp, viewIndex, heading, elevation in scan_vp_cam_poses
#     ],
# }

# # save to json
# _ = create_folder(configs.SAVE.VAL_UNSEEN_POSE)

# with open(os.path.join(configs.SAVE.VAL_UNSEEN_POSE, "val_unseen_pose.json"), "w") as f:
#     json.dump(data, f)
#     print(f"Save poses to {configs.SAVE.VAL_UNSEEN_POSE}/val_unseen_pose.json")
