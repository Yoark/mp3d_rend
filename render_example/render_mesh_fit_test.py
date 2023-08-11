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
    predicted_image=None,
    # predicted_mesh=None,
    # renderer=None,
    target_image=None,
    title="",
    save_dir=None,
):
    inds = 3

    with torch.no_grad():
        # if not predicted_image:
        # predicted_images = renderer(predicted_mesh)
        # else:
        predicted_images = predicted_image.clone()
        predicted_images = predicted_images[..., :inds].cpu().detach().numpy()
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images)

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created {save_dir}")
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

# scan = scan_ids[-3]EU6Fwq7SyZv
scan = "EU6Fwq7SyZv"
test_render_set = scan2vpspose[scan]
vp = test_render_set[0]["vp"]
heading = test_render_set[0]["heading"]
elevation = test_render_set[0]["elevation"]

# load second goal image
vp2 = test_render_set[1]["vp"]
heading2 = test_render_set[1]["heading"]
elevation2 = test_render_set[1]["elevation"]

headings = [heading, heading2]
elevations = [elevation, elevation2]

device = get_device()
mesh_data = load_meshes(
    [scan],
    mesh_dir=cfg.DATA.MESH_DIR,
    device=device,
    texture_atlas_size=5,
    with_atlas=False,
)
viewpoint_info = get_viewpoint_info(scan, vp, scan_to_vps_to_data)
############

# maybe scale the mesh to 0,

verts, faces, aux = mesh_data[scan]
# ! deform verts is not useful for currrnt case
# deform_verts = torch.full(verts.shape, 0.0, device=device, requires_grad=True)

atlas = aux.texture_atlas
if atlas.ndim == 4:
    atlas = atlas.unsqueeze(0)
atlas = atlas.to(device)
verts = verts.to(device)
faces = faces.to(device)
# atlas = Parameter(atlas)
# atlas.requires_grad = True
# atlas2 = atlas.clone()
# atlas = torch.nn.Parameter(atlas, requires_grad=True)
# textures = TexturesAtlas(atlas=atlas)
# textures2 = TexturesAtlas(atlas=atlas2)
# verts.requires_grad = True
# verts = Parameter(verts)
# textures = Parameter(textures)
# mesh = Meshes(
#     verts=[verts],
#     faces=[faces],
#     textures=textures,
# )

# meshes = mesh.extend(2)

# atlas = meshes[0].textures.atlas_packed().clone().detach()
# atlas_packed.requires_grad = True
# meshes.textures = TexturesAtlas(atlas=atlas_packed)
# meshes = [mesh] * 2
# * testing different methods for combining a mesh
verts = [verts, verts.clone().detach()]
faces = [faces, faces.clone().detach()]
# textures = TexturesAtlas(atlas=torch.cat([atlas, atlas2], dim=0))
comb_atlas = torch.cat([atlas, atlas], dim=0)
comb_atlas = torch.nn.Parameter(comb_atlas, requires_grad=True)
textures = TexturesAtlas(atlas=comb_atlas)
meshes = Meshes(
    verts=verts,
    faces=faces,
    textures=textures,
)
# N = verts.shape[0]
# center = verts.mean(0)
# scale = max((verts - center).abs().max()[0])
# mesh.offset_verts(-center)

# current render heading and elev
print(f"first goal {scan}, {vp}, {heading}, {elevation}")
print(f"second goal {scan}, {vp2}, {heading2}, {elevation2}")
# display target image.
# read_image(scan, vp, heading, elevation)

# set rasterization settings
raster_settings = RasterizationSettings(
    image_size=((cfg.CAMERA.HEIGHT, cfg.CAMERA.WIDTH)),
    blur_radius=0.0,
    faces_per_pixel=1,
    cull_to_frustum=False,
    cull_backfaces=True,
)

# set lights
ambient_color = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
light = AmbientLights(device=device, ambient_color=ambient_color[None, :])
# set up renderer and intial view
pose = viewpoint_info["pose"]
# maybe need to batchfy this function
render_params = get_render_params(
    pose,
    headings,
    elevations,
    raster_settings=raster_settings,
    cfg=cfg,
    device=device,
)

# get render params for second goal
# render_params2 = get_render_params(
#     pose,
#     heading2,
#     elevation2,
#     raster_settings=raster_settings,
#     cfg=cfg,
#     device=device,
# )
# get renderer
cameras = render_params["cameras"]
raster_settings = render_params["raster_settings"]
device = render_params["device"]
light = render_params["light"]

# get camera2
# camera2 = render_params2["camera"]

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=light),
)
# set up renderer2 this is temporary
# renderer2 = MeshRenderer(
#     rasterizer=MeshRasterizer(cameras=camera2, raster_settings=raster_settings),
#     shader=SoftPhongShader(device=device, cameras=camera2, lights=light),
# )


def get_target_image(scan, vp, heading, elevation, normalize=True):
    target_image = read_image(scan, vp, heading, elevation)
    target_image = torch.from_numpy(np.array(target_image)).to(device) / 255.0
    return target_image


target_image1 = get_target_image(scan, vp, heading, elevation)
target_image2 = get_target_image(scan, vp2, heading2, elevation2)


os.makedirs("render_example/save/fitting", exist_ok=True)
savedir = f"render_example/save/fitting/{scan}_{vp}_{heading}_{elevation}"

plot_period = 100
iter = trange(1000)
losses = {
    "image1": {"weight": 1.0, "values": []},
    "image2": {"weight": 1.0, "values": []},
    "edge": {"weight": 0.1, "values": []},
    "normal": {"weight": 0.1, "values": []},
    "laplacian": {"weight": 0.1, "values": []},
    "sparsity": {"weight": 0.1, "values": []},
}

optimizer = torch.optim.AdamW(
    [comb_atlas],
    lr=1e-3,
    weight_decay=1e-4,
)
# optimizer = torch.optim.SGD([deform_verts, atlas], lr=0.01, momentum=0.9)
for i in iter:
    # check if the mesh's texture is modified
    optimizer.zero_grad()
    # new_mesh = mesh.offset_verts(deform_verts)
    # TODO how to remove the mem of mesh 1 but still keep the grad
    # meshes = meshes[0].extend(2)
    images = renderer(meshes, cameras=cameras)
    # image2 = renderer2(new_mesh)
    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    # update_mesh_shape_prior_losses(new_mesh, loss)
    # loss["sparsity"] = torch.norm(deform_verts, p=1)
    # compute mse

    loss["image1"] = mse_loss(images[0, ..., :3], target_image1, reduction="mean")
    loss["image2"] = mse_loss(images[1, ..., :3], target_image2, reduction="mean")

    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        # TODO here!
        losses[k]["values"].append(l.item())

    iter.set_description(f"loss: {sum_loss.item()}")

    if i % plot_period == 0:
        visualize_prediction(
            images[0],
            # new_mesh,
            title="iter: %d image1" % i,
            # renderer=renderer,
            target_image=target_image1,
            save_dir=savedir,
        )

        visualize_prediction(
            images[1],
            # new_mesh,
            title="iter: %d image2" % i,
            # renderer=renderer,
            target_image=target_image2,
            save_dir=savedir,
        )
    sum_loss.backward()
    # TODO combine the gradient of two atlas to a single one
    # meshes[0].textures.atlas_packed().grad += meshes[1].textures.atlas_packed().grad
    # meshes[1].textures.atlas_packed().grad = meshes[0].textures.atlas_packed().grad

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
