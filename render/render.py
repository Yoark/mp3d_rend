from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Util function for loading meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    AmbientLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.textures import TexturesAtlas
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation
from sympy import Union
from yacs.config import CfgNode

from render.utils import get_obj_paths

# DONE (zijiao): use config file rather than hard-coded parameters
# setting up the correct meta parameters
# MESH_DIR = "/mnt/sw/vln/data/matterport3d/mp3d_mesh/v1/scans/{}/matterport_mesh"
# VFOV = 60
# WIDTH = 640
# HEIGHT = 480
# FOV = VFOV * WIDTH / HEIGHT
# HEADINGS = [np.deg2rad(30.0 * h) for h in range(12)]
# ELEVATIONS = [np.deg2rad(e) for e in [-30.0, 0, 30]]

# aspect_ratio = 1  # pixel aspect
# image_aspect_ratio = 4 / 3  # image aspect
# image_size = (WIDTH, HEIGHT)


# * rotation operations
def normalize(v):
    return v / np.linalg.norm(v)


def rotate_vector(vec, axis, angle_radians):
    rotation = Rotation.from_rotvec(axis * angle_radians, degrees=False)
    return rotation.apply(vec)


# Camera parameters
def rotate_heading(eye, at, up, angle_radians):
    eye = np.array(eye)
    at = np.array(at)
    up = np.array(up)

    look = at - eye
    look_normalized = normalize(look)

    # add negative sign such that turn right is positive
    look_rotated = rotate_vector(look_normalized, up, -angle_radians)

    # Compute the new target position
    at_rotated = eye + look_rotated * np.linalg.norm(look)
    return eye, at_rotated, up


def rotate_heading_elevation(eye, at, up, heading, elevation):
    eye = np.array(eye)
    at = np.array(at)
    up = np.array(up)

    # Calculate the view direction and right vector
    view_direction = at - eye
    right = np.cross(view_direction, up)
    # right = (1, 0, 0)

    # Normalize view direction, up, and right vectors
    view_direction = view_direction / np.linalg.norm(view_direction)
    up = up / np.linalg.norm(up)
    right = right / np.linalg.norm(right)

    # Create rotations for heading and elevation
    heading_rotation = Rotation.from_rotvec(-heading * up)
    elevation_rotation = Rotation.from_rotvec(elevation * right)
    combined_rotation = heading_rotation * elevation_rotation

    # Rotate the view direction and up vector
    rotated_view_direction = combined_rotation.apply(view_direction)
    rotated_up_vector = combined_rotation.apply(up)

    # Calculate new at-point
    rotated_at = eye + rotated_view_direction

    return eye.tolist(), rotated_at.tolist(), rotated_up_vector.tolist()


def load_meshes(
    scan_ids: List[str], mesh_dir: str, with_atlas: bool = True, **kwargs: Any
) -> Dict[str, Tuple]:
    """load meshes from scan ids

    Args:
        scan_ids (List of str): list of scan ids

    Returns:
        Meshes: Meshes object (batched meshes)
    """
    device = kwargs.get("device", "cpu")
    texture_atlas_size = kwargs.get("texture_atlas_size", 30)

    scan_paths = get_obj_paths(mesh_dir, scan_ids)
    mesh_dict = {}
    for scan, scan_path in scan_paths.items():
        verts, faces, aux = load_obj(
            scan_path,
            load_textures=True,
            create_texture_atlas=True,
            texture_wrap="clamp",
            texture_atlas_size=texture_atlas_size,
        )
        if with_atlas:
            atlas = aux.texture_atlas
            if atlas.ndim == 4:
                atlas = atlas.unsqueeze(0)
            textures = TexturesAtlas(atlas=atlas)
            mesh_dict[scan] = (
                verts.to(device),
                faces.verts_idx.to(device),
                textures.to(device),
            )
        else:
            mesh_dict[scan] = (verts.to(device), faces.verts_idx.to(device), aux)

        # TODO explore padding to potentially speed up
        # meshes = Meshes(verts=n_verts, faces=n_face_verts_idx, textures=n_textures)
        # meshes = meshes.to(device)

    return mesh_dict


# function prepare for camera and viewpoint info


# TODO join function into classes
# visualize mesh texture atlas
def visualize_texture_atlas(atlas: Dict[str, Any], save_dir: str) -> None:
    # Convert the atlas to a NumPy array and remove the batch dimension
    atlas_np = atlas.cpu().numpy().squeeze(0)

    # Calculate the number of rows and columns for the grid
    num_faces, height, width, num_channels = atlas_np.shape
    num_rows = int(np.ceil(np.sqrt(num_faces)))
    num_cols = int(np.ceil(np.sqrt(num_faces)))

    # Create an empty grid for the texture atlas image
    atlas_image = np.zeros(
        (num_rows * height, num_cols * width, num_channels), dtype=np.float32
    )

    # Fill the grid with the face textures from the atlas
    for face_idx, face_texture in enumerate(atlas_np):
        row = face_idx // num_cols
        col = face_idx % num_cols
        atlas_image[
            row * height : (row + 1) * height, col * width : (col + 1) * width
        ] = face_texture

    # Visualize the texture atlas image
    plt.figure(figsize=(10, 10))
    plt.imshow(atlas_image)
    plt.axis("off")
    if save_dir:
        plt.imsave(save_dir + "/atlas.png", atlas_image)
    else:
        plt.show()


# create a rendering at given viewpoint within a mesh, and heading elevation
def init_episode(
    viewpoint_data: dict,
    heading: float,
    elevation: float,
    cfg: CfgNode,
    mesh: Optional[Any],
    K=None,
    **kwargs: Any,
) -> Dict[str, Any]:
    # load mesh
    # initialize the camera
    raster_settings = kwargs.get(
        "raster_settings",  # render settings
        RasterizationSettings(
            image_size=(cfg.RENDER.IMAGE_SIZE),
            blur_radius=0.0,
            faces_per_pixel=1,
        ),
    )
    device = kwargs.get("device", "cpu")
    pose = viewpoint_data["pose"]
    eye = [pose[3], pose[7], pose[11]]  # location
    at = [eye[0], eye[1] + 1, eye[2]]
    up = [0, 0, 1]
    eye_r, at_r, up_r = rotate_heading_elevation(eye, at, up, heading, elevation)
    # init camera
    R, T = look_at_view_transform(eye=[eye_r], up=[up_r], at=[at_r])
    camera = FoVPerspectiveCameras(
        device=device,
        # zfar=500,
        # znear=20,
        R=R,
        T=T,
        # K=K,
        aspect_ratio=cfg.RENDER.PIXEL_ASPECT_RATIO,
        fov=60,  # cfg.CAMERA.HFOV,
    )

    # use ambient lighting
    # ambient = AmbientLights(device=device, ambient_color=((1, 1, 1),))
    light = kwargs.get(
        "light", AmbientLights(device=device, ambient_color=((1, 1, 1),))
    )
    # point_light = PointLights(location=eye_r, device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=camera, lights=light),
    )
    if mesh:
        if isinstance(mesh, tuple):
            verts, faces, textures = mesh
            mesh = Meshes(
                verts=[verts.to(device)],
                faces=[faces.to(device)],
                textures=textures.to(device),
            )
        images = renderer(mesh, cameras=camera, lights=light)
        #! This does not work images = renderer(mesh, cameras=cameras, lights=point_light)
        plt.figure(figsize=(8, 6))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
    return {
        "renderer": renderer,
        "eye": eye,
        "at": at,
        "up": up,
        "camera": camera,
        "light": light,
        "mesh": mesh,
    }


def get_render_params(
    pose,
    heading: float,
    elevation: float,
    cfg: CfgNode,
    K=None,
    require_grad=False,
    **kwargs: Any,
) -> Dict[str, Any]:
    # load mesh
    # initialize the camera
    raster_settings = kwargs.get(
        "raster_settings",  # render settings
        RasterizationSettings(
            image_size=(cfg.RENDER.IMAGE_SIZE),
            blur_radius=0.0,
            faces_per_pixel=1,
        ),
    )
    device = kwargs.get("device", "cpu")
    eye = [pose[3], pose[7], pose[11]]  # location
    at = [eye[0], eye[1] + 1, eye[2]]
    up = [0, 0, 1]
    eye_r, at_r, up_r = rotate_heading_elevation(eye, at, up, heading, elevation)
    # init camera
    R, T = look_at_view_transform(eye=[eye_r], up=[up_r], at=[at_r])
    R = R.requires_grad_(require_grad)
    T = T.requires_grad_(require_grad)
    camera = FoVPerspectiveCameras(
        device=device,
        # zfar=500,
        # znear=20,
        R=R,
        T=T,
        # K=K,
        aspect_ratio=cfg.RENDER.PIXEL_ASPECT_RATIO,
        fov=60,  # cfg.CAMERA.HFOV,
    )

    # use ambient lighting
    # ambient = AmbientLights(device=device, ambient_color=((1, 1, 1),))
    light = kwargs.get(
        "light", AmbientLights(device=device, ambient_color=((1, 1, 1),))
    )
    # point_light = PointLights(location=eye_r, device=device)
    return {
        "camera": camera,
        "light": light,
        "eye": eye,
        "at": at,
        "up": up,
        "device": device,
        "raster_settings": raster_settings,
    }


class CameraData:
    """This class is a data wrapper for camera pose."""

    def __init__(self, pose):
        self.eye = [pose[3], pose[7], pose[11]]
        self.at = [self.eye[0], self.eye[1] + 1, self.eye[2]]
        self.up = [0, 0, 1]
