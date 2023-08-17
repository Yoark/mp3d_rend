import math

import numpy as np
from yacs.config import CfgNode as CN

_C = CN()
_C.TEST = False

_C.DATA = CN()
_C.DATA.MESH_DIR = "/mnt/sw/vln/data/matterport3d/mp3d_mesh/v1/scans/{}/matterport_mesh"
_C.DATA.MATTERPORT_IMAGE_DIR = "/mnt/sw/vln/data/matterport3d/v1/scans"
_C.DATA.CONNECTIVITY_DIR = "/mnt/sw/vln/data/matterport3d/connectivity"
_C.DATA.RXR_DIR = "/mnt/sw/vln/data/RxR/rxr_train_guide.jsonl.gz"

_C.DATA.RXR = CN()
_C.DATA.RXR.TRAIN = "/mnt/sw/vln/data/RxR/rxr_train_guide.jsonl.gz"
_C.DATA.RXR.VAL_UNSEEN = "/mnt/sw/vln/data/RxR/rxr_val_unseen_guide.jsonl.gz"
_C.DATA.RXR.VAL_SEEN = "/mnt/sw/vln/data/RxR/rxr_val_seen_guide.jsonl.gz"

_C.DATA.POSE = ""


_C.CAMERA = CN()
_C.CAMERA.VFOV = 60
_C.CAMERA.WIDTH = 640
_C.CAMERA.HEIGHT = 480
# _C.CAMERA.HFOV = _C.CAMERA.VFOV * _C.CAMERA.WIDTH / _C.CAMERA.HEIGHT
_C.CAMERA.HFOV = math.degrees(
    2
    * math.atan(
        math.tan(math.radians(_C.CAMERA.VFOV) / 2)
        * (_C.CAMERA.WIDTH / _C.CAMERA.HEIGHT)
    )
)

_C.MESH = CN()
_C.MESH.TEXTURE_ATLAS_SIZE = 50
_C.MESH.TEXTURE_WRAP = "repeat"

_C.RENDER = CN()
_C.RENDER.PIXEL_ASPECT_RATIO = 1  # pixel aspect
_C.RENDER.IMAGE_ASPECT_RATIO = 4 / 3  # image aspect
_C.RENDER.IMAGE_SIZE = (_C.CAMERA.HEIGHT, _C.CAMERA.WIDTH)  # pytorch3d uses (H, W)
_C.RENDER.IMAGE_MODEL_NAME = None
_C.RENDER.IMAGE_MODEL_PATH = None

_C.MP3D = CN()
_C.MP3D.HEADINGS = [np.deg2rad(30.0 * h) for h in range(12)]
_C.MP3D.ELEVATIONS = [np.deg2rad(e) for e in [-30.0, 0, 30]]
_C.MP3D.VIEWPOINT_SIZE = 36

_C.SAVE = CN()
_C.SAVE.IMAGE_DIR = ""
_C.SAVE.VAL_UNSEEN_POSE = ""
_C.SAVE.VAL_SEEN_POSE = ""


def get_cfg_defaults():
    return _C.clone()


cfg = _C

# Note on coordinate systems in pytorch3d
# woorld and camera view, ndc system: x left, y up, z inward
# screen: top left is (0, 0), bottom right is (W, H)
