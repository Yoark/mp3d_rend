import gzip
import json
import os
import pathlib
from collections import defaultdict
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
from yacs.config import CfgNode

# setting up the correct meta parameters
MESH_DIR = "/mnt/sw/vln/data/matterport3d/mp3d_mesh/v1/scans/{}/matterport_mesh"
VFOV = 60
WIDTH = 640
HEIGHT = 480
FOV = VFOV * WIDTH / HEIGHT
# HEADINGS = [np.deg2rad(30.0 * h) for h in range(12)]
# ELEVATIONS = [np.deg2rad(e) for e in [-30.0, 0, 30]]

# aspect_ratio = 1  # pixel aspect
# image_aspect_ratio = 4 / 3  # image aspect
image_size = (WIDTH, HEIGHT)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def read_gz_jsonlines(filename):
    data = []
    with open(filename, "rb") as f:
        for args in map(json.loads, gzip.open(f)):
            data.append(args)
    return data


# load viewpoints -> scan : {viewpointid : pose, height}
def load_viewpoints_dict(conn: str) -> Tuple[Dict[str, List], int, Dict[str, Dict]]:
    """Loads viewpoints into a dictionary of sceneId -> viewpoints -> pose, weight
    where each viewpoint has keys viewpointId, and the value has key of pose, height}.
    """
    viewpoints = []
    with open(os.path.join(conn, "scans.txt")) as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(os.path.join(conn, f"{scan}_connectivity.json")) as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpoint_data = {
                            "viewpointId": item["image_id"],
                            "pose": item["pose"],
                            "height": item["height"],
                        }
                        viewpoints.append((scan, viewpoint_data))

    scans_to_vps = defaultdict(list)
    for scene_id, viewpoint in viewpoints:
        scans_to_vps[scene_id].append(viewpoint)

    scan_to_vp_to_meta = defaultdict(dict)
    for scene_id, viewpoint in viewpoints:
        scan_to_vp_to_meta[scene_id][viewpoint["viewpointId"]] = viewpoint

    return scans_to_vps, len(viewpoints), scan_to_vp_to_meta


def get_obj_paths(base_dir, scan_ids):
    # Format base_dir with scan_id and create a pathlib.Path object
    obj_files = {}
    for scan_id in scan_ids:
        scan_path = pathlib.Path(base_dir.format(scan_id))
        # Get the first directory inside the scan_path
        obj_file_dir = list(scan_path.iterdir())[0]
        # Find .obj file inside obj_file_dir
        obj_file = [d for d in obj_file_dir.iterdir() if d.suffix == ".obj"][0]
        obj_files[scan_id] = obj_file
    return obj_files


# * rotation operations
def normalize(v):
    return v / np.linalg.norm(v)


def rotate_vector(vec, axis, angle_radians):
    rotation = Rotation.from_rotvec(axis * angle_radians, degrees=False)
    return rotation.apply(vec)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


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


def load_meshes(scan_ids: List[str], mesh_dir: str, **kwargs: Any) -> Dict[str, Meshes]:
    """load meshes from scan ids

    Args:
        scan_ids (List of str): list of scan ids

    Returns:
        Dict[str, Meshes]: dictionary of scan id to Meshes
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
            texture_wrap="repeat",
            texture_atlas_size=texture_atlas_size,
        )
        atlas = aux.texture_atlas
        if atlas.ndim == 4:
            atlas = atlas.unsqueeze(0)
        textures = TexturesAtlas(atlas=atlas)

        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)
        mesh = mesh.to(device)
        mesh_dict[scan] = mesh

    return mesh_dict


# function prepare for camera and viewpoint info
def get_viewpoint_info(
    scan_id: str, viewpointid: str, scan_to_vps_to_data: Dict[str, Any]
) -> Dict[str, Any]:
    scan_viewpoints = scan_to_vps_to_data[scan_id]
    viewpoint_data = scan_viewpoints[viewpointid]
    pose = viewpoint_data["pose"]
    # height = viewpoint_data['height']
    location = [pose[3], pose[7], pose[11]]
    return {
        "location": location,
        "viewpointId": viewpointid,
        "pose": pose,
    }


# visualize mesh texture atlas
def visualize_texture_atlas(atlas: Dict[str, Any]) -> None:
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
    plt.show()


# create a rendering at given viewpoint within a mesh, and heading elevation
def init_episode(
    viewpoint_data: dict,
    heading: float,
    elevation: float,
    cfg: CfgNode,
    mesh: Optional[Meshes] = None,
    **kwargs: Dict[str, Any],
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
    pose = viewpoint_data["pose"]
    eye = [pose[3], pose[7], pose[11]]  # location
    at = [eye[0], eye[1] + 1, eye[2]]
    up = [0, 0, 1]
    eye_r, at_r, up_r = rotate_heading_elevation(eye, at, up, heading, elevation)
    # init camera
    R, T = look_at_view_transform(eye=[eye_r], up=[up_r], at=[at_r])
    camera = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        aspect_ratio=cfg.RENDER.PIXEL_ASPECT_RATIO,
        fov=cfg.CAMERA.HFOV,
    )

    # use ambient lighting
    ambient = AmbientLights(device=device, ambient_color=((1, 1, 1),))
    # point_light = PointLights(location=eye_r, device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=camera,
            # lights=lights
        ),
    )
    if mesh:
        images = renderer(mesh, cameras=camera, lights=ambient)
        #! This does not work images = renderer(mesh, cameras=cameras, lights=point_light)
        plt.figure(figsize=(4, 3))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
    return {
        "renderer": renderer,
        "eye": eye,
        "at": at,
        "up": up,
        "camera": camera,
        "light": ambient,
    }


class CameraData:
    """This class is a data wrapper for camera pose."""

    def __init__(self, pose):
        self.eye = [pose[3], pose[7], pose[11]]
        self.at = [self.eye[0], self.eye[1] + 1, self.eye[2]]
        self.up = [0, 0, 1]
