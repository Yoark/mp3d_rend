import numpy as np
from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.MESH_DIR = "/mnt/sw/vln/data/matterport3d/mp3d_mesh/v1/scans/{}/matterport_mesh"
_C.DATA.CONNECTIVITY_DIR = "/mnt/sw/vln/data/matterport3d/connectivity"
_C.DATA.RXR_DIR = "/mnt/sw/vln/data/RxR/rxr_train_guide.jsonl.gz"

_C.CAMERA = CN()
_C.CAMERA.VFOV = 60
_C.CAMERA.WIDTH = 640
_C.CAMERA.HEIGHT = 480
_C.CAMERA.HFOV = _C.CAMERA.VFOV * _C.CAMERA.WIDTH / _C.CAMERA.HEIGHT

_C.MESH = CN()
_C.MESH.TEXTURE_ATLAS_SIZE = 30
_C.MESH.TEXTURE_WRAP = "repeat"
_C.RENDER = CN()
_C.RENDER.PIXEL_ASPECT_RATIO = 1  # pixel aspect
_C.RENDER.IMAGE_ASPECT_RATIO = 4 / 3  # image aspect
_C.RENDER.IMAGE_SIZE = (_C.CAMERA.WIDTH, _C.CAMERA.HEIGHT)

_C.MP3D = CN()
_C.MP3D.HEADINGS = [np.deg2rad(30.0 * h) for h in range(12)]
_C.MP3D.ELEVATIONS = [np.deg2rad(e) for e in [-30.0, 0, 30]]


def get_cfg_defaults():
    return _C.clone()


cfg = _C
