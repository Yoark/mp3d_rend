# load a pose

import json

import matplotlib.pyplot as plt
import torch
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    AmbientLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesAtlas,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from torch.nn import Parameter
from torch.nn.functional import mse_loss
from tqdm import trange

from render.config import cfg
from render.render import get_render_params, init_episode, load_meshes
from render.utils import get_device, get_viewpoint_info, load_viewpoints_dict

# %load_ext autoreload
# %autoreload 2


# define render function with default stuff.
def get_mesh_renderer(cfg, scan):
    cfg.merge_from_file("render_example/configs/mp3d_render.yaml")
    device = get_device()

    mesh_data = load_meshes(
        [scan],
        mesh_dir=cfg.DATA.MESH_DIR,
        device=device,
        texture_atlas_size=50,
        with_atlas=False,
    )
    verts, faces, aux = mesh_data[scan]
    # create mesh
    atlas = aux.texture_atlas
    if atlas.ndim == 4:
        atlas = atlas.unsqueeze(0)
    atlas = atlas.to(device)
    verts = verts.to(device)
    faces = faces.to(device)
    # atlas = Parameter(atlas)
    # atlas.requires_grad = True
    textures = TexturesAtlas(atlas=atlas)
    # verts.requires_grad = True
    # verts = Parameter(verts)
    # textures = Parameter(textures)
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures,
    )

    # replace this with scan_id
    with open("./render_example/save/poses/val_unseen_scan2vpspose.json") as f:
        scan2vpspose = json.load(f)

    # replace these with the vp, heading elevation we get from mattport3d sim
    test_render_set = scan2vpspose[scan]
    vp = test_render_set[0]["vp"]
    heading = test_render_set[0]["heading"]
    elevation = test_render_set[0]["elevation"]
    _, _, scan_to_vps_to_data = load_viewpoints_dict(cfg.DATA.CONNECTIVITY_DIR)
    viewpoint_info = get_viewpoint_info(scan, vp, scan_to_vps_to_data)

    # set rasterazation setting
    raster_settings = RasterizationSettings(
        image_size=((cfg.CAMERA.HEIGHT, cfg.CAMERA.WIDTH)),
        blur_radius=0.0,
        faces_per_pixel=5,
    )

    # set lights
    ambient_color = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)[None, :]
    light = AmbientLights(device=device, ambient_color=ambient_color)
    # set up renderer params
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
        shader=SoftPhongShader(cameras=camera, device=device, lights=light),
    )

    return renderer, mesh


if __name__ == "__main__":
    scan = "x8F5xyUWy9e"
    renderer, mesh = get_mesh_renderer(cfg, scan)
    image = renderer(mesh)
    plt.imshow(image[..., :3].squeeze().cpu().numpy())
    plt.axis("off")
    plt.savefig("./test.png")
# plt.imshow(image[..., :3].squeeze().cpu().numpy())
