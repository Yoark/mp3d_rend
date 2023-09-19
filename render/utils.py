import gzip
import json
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# create folder if not exist
def create_folder(folder_path):
    abs_path = pathlib.Path(folder_path).resolve()
    abs_path.mkdir(parents=True, exist_ok=True)
    # print(abs_path, "created.")
    return abs_path


def read_gz_jsonlines(filename):
    data = []
    with open(filename, "rb") as f:
        for args in map(json.loads, gzip.open(f)):
            data.append(args)
    return data


def get_device(get_cpu=False) -> torch.device:
    if get_cpu:
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        return device


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


def get_viewpoint_info(
    scan_id: str, viewpointid: str, scan_to_vps_to_data: Dict[str, Any]
) -> Dict[str, Any]:
    viewpoint_data = scan_to_vps_to_data[scan_id][viewpointid]
    pose = viewpoint_data["pose"]
    # height = viewpoint_data['height']
    location = [pose[3], pose[7], pose[11]]
    return {
        "location": location,
        "viewpointId": viewpointid,
        "pose": pose,
    }


def build_feature_extractor(model_name, checkpoint_file=None, device="cpu"):
    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(
        device
    )
    if checkpoint_file is not None:
        state_dict = torch.load(
            checkpoint_file, map_location=lambda storage, loc: storage
        )["state_dict"]
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms


def read_image(scan, vp, heading, elevation, display=False, image_folder=""):
    # find image and display for compare
    if not image_folder:
        image_folder = (
            "/home/zijiao/research/pytorch_rend/render_example/save/images/val_seen"
        )
    img_path = os.path.join(
        image_folder,
        scan,
        f"{scan}_{vp}_{heading}_{elevation}.png",
    )
    img = Image.open(img_path)
    print("Image size:", img.size)
    if display:
        img.show()
    return img


def save_images_to_one(
    images: List[torch.tensor],
    # predicted_mesh=None,
    # renderer=None,
    title: List[str] = [],
    filename: str = "",
    save_dir: str = None,
):
    """generate a figure with image list and title list

    Args:
        images (List[torch.tensor]): images to be saved
        title (List[str], optional): . Defaults to [].
        save_dir (str, optional): save dir. Defaults to None.
    """
    # inds = 3

    plt.figure(figsize=(20, 10))
    plt.axis("off")

    for i, image in enumerate(images):
        # image = image[..., :inds].cpu().detach().numpy()
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.title(title[i])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created {save_dir}")
    # title = "_".join(title)
    plt.savefig(f"{save_dir}/{filename}.png")
