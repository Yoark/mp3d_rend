# load a pose

import json
import os
import resource

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from torch.nn import Parameter
from torch.nn.functional import mse_loss
from tqdm import trange

from plot_image_grid import image_grid
from render.config import cfg
from render.render import get_render_params, init_episode, load_meshes
from render.utils import get_device, get_viewpoint_info, load_viewpoints_dict


# mesh
def visualize_prediction(
    predicted_mesh, renderer=None, target_image=None, title="", save_dir=None
):
    inds = 3
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., :inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{save_dir}/{title}.png")


# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l["values"], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")


def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

    # l1 norm


def print_memory_usage():
    # Note: `ru_maxrss` is in KB on Linux, and B on macOS
    factor = 1024
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / factor
    print(f"Memory usage: {mem_usage:.4f} MB")


def read_image(scan, vp, heading, elevation, display=False):
    # find image and display for compare
    img_path = os.path.join(
        cfg.SAVE.IMAGE_DIR, scan, f"{scan}_{vp}_{heading}_{elevation}.png"
    )
    img = Image.open(img_path)
    print("Image size:", img.size)
    if display:
        img.show()
    return img


cfg.merge_from_file("render_example/configs/mp3d_render.yaml")
#! here it takes a lot of time to load the meshes
# with open("./render_example/save/poses/val_unseen_pose.json") as f:
# scan_vp_poses = json.load(f)

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

with open("./render_example/save/poses/val_unseen_scan2vpspose.json") as f:
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
    texture_atlas_size=50,
    with_atlas=False,
)
viewpoint_info = get_viewpoint_info(scan, vp, scan_to_vps_to_data)
############

# maybe scale the mesh to 0,

verts, faces, aux = mesh_data[scan]
deform_verts = torch.full(verts.shape, 0.0, device=device, requires_grad=True)

atlas = aux.texture_atlas
if atlas.ndim == 4:
    atlas = atlas.unsqueeze(0)
atlas = atlas.to(device)
verts = verts.to(device)
faces = faces.to(device)
# atlas = Parameter(atlas)
atlas.requires_grad = True
textures = TexturesAtlas(atlas=atlas)
# verts.requires_grad = True
# verts = Parameter(verts)
# textures = Parameter(textures)
mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures,
)
# N = verts.shape[0]
# center = verts.mean(0)
# scale = max((verts - center).abs().max()[0])
# mesh.offset_verts(-center)

# current render heading and elev
print(scan, vp, heading, elevation)
# display target image.
# read_image(scan, vp, heading, elevation)

# set rasterization settings
raster_settings = RasterizationSettings(
    image_size=((cfg.CAMERA.HEIGHT, cfg.CAMERA.WIDTH)),
    blur_radius=0.0,
    faces_per_pixel=10,
)

# set lights
ambient_color = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)[None, :]
light = AmbientLights(device=device, ambient_color=ambient_color)
# set up renderer and intial view
render_params = get_render_params(
    viewpoint_info,
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
    shader=SoftPhongShader(device=device, cameras=camera, lights=light),
)

target_image = read_image(scan, vp, heading, elevation)

target_image = torch.from_numpy(np.array(target_image)).to(device) / 255.0


os.makedirs("render_example/save/fitting", exist_ok=True)
savedir = "render_example/save/fitting"

plot_period = 100
iter = trange(1000)
losses = {
    "image": {"weight": 1.0, "values": []},
    "edge": {"weight": 0.1, "values": []},
    "normal": {"weight": 0.1, "values": []},
    "laplacian": {"weight": 0.1, "values": []},
    "sparsity": {"weight": 0.1, "values": []},
}

optimizer = torch.optim.AdamW([deform_verts, atlas], lr=1e-3, weight_decay=1e-4)
# optimizer = torch.optim.SGD([deform_verts, atlas], lr=0.01, momentum=0.9)
for i in iter:
    optimizer.zero_grad()
    new_mesh = mesh.offset_verts(deform_verts)
    image = renderer(new_mesh)
    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    update_mesh_shape_prior_losses(new_mesh, loss)
    loss["sparsity"] = torch.norm(deform_verts, p=1)
    # compute mse
    loss["image"] = mse_loss(image.squeeze()[..., :3], target_image, reduction="mean")
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        # TODO here!
        losses[k]["values"].append(l.item())

    iter.set_description(f"loss: {sum_loss.item()}")

    if i % plot_period == 0:
        visualize_prediction(
            new_mesh,
            title="iter: %d" % i,
            renderer=renderer,
            target_image=target_image,
            save_dir=savedir,
        )
    sum_loss.backward()
    optimizer.step()
    # if i % 10 == 0:

    #     print(f"loss: {loss.item()}")
    #     plt.imshow(image[0, ..., :3].cpu().detach().numpy())
    #     plt.axis("on")
    #     plt.title(f"iter: {i}")
    # plt.show()
    # plt.show()
    # fig = plot_scene({
    #     "subplot1": {
    #         "cow_mesh": clipped_mesh
    #     }
    # })
