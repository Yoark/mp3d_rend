# This script provide an example to compute image feature,
# it is modified from "https://raw.githubusercontent.com/cshizhe/VLN-HAMT/main/preprocess/precompute_img_features_vit.py"

#!/usr/bin/env python3
# TODO convert this to pytorch3d compatible
""" Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. """

import argparse
import copy
import json
import math
import os
import sys
import time

import h5py
import MatterSim
import numpy as np
import timm
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from progressbar import ProgressBar
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from render.render import build_simulator
from render.utils import build_feature_extractor


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


TSV_FIELDNAMES = [
    "scanId",
    "viewpointId",
    "image_w",
    "image_h",
    "vfov",
    "features",
    "logits",
]
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60


def process_features(proc_id, out_queue, scanvp_list, args):
    print("start proc_id: %d" % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(
        args.model_name, args.checkpoint_file
    )

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True)  # in BGR channel
            image = Image.fromarray(
                image[:, :, ::-1]
            )  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts, logits = [], []
        for k in range(0, len(images), args.batch_size):
            b_fts = model.forward_features(images[k : k + args.batch_size])
            b_logits = model.head(b_fts)
            b_fts = b_fts.data.cpu().numpy()
            b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            logits.append(b_logits)
        fts = np.concatenate(fts, 0)
        logits = np.concatenate(logits, 0)

        out_queue.put((scan_id, viewpoint_id, fts, logits))

    out_queue.put(None)


def build_feature_file(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx:eidx], args),
        )
        process.start()
        processes.append(process)

    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, "w") as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts, logits = res
                key = "%s_%s" % (scan_id, viewpoint_id)
                if args.out_image_logits:
                    data = np.hstack([fts, logits])
                else:
                    data = fts
                outf.create_dataset(key, data.shape, dtype="float", compression="gzip")
                outf[key][...] = data
                outf[key].attrs["scanId"] = scan_id
                outf[key].attrs["viewpointId"] = viewpoint_id
                outf[key].attrs["image_w"] = WIDTH
                outf[key].attrs["image_h"] = HEIGHT
                outf[key].attrs["vfov"] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument("--connectivity_dir", default="../connectivity")
    parser.add_argument("--scan_dir", default="../data/v1/scans")
    parser.add_argument("--out_image_logits", action="store_true", default=False)
    parser.add_argument("--output_file")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    build_feature_file(args)
