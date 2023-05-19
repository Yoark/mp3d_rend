import os

import numpy as np
from ipdb import launch_ipdb_on_exception
from matplotlib import pyplot as plt
from pytorch3d.io import load_obj

from render.config import cfg
from render.render import (
    get_device,
    get_obj_paths,
    load_viewpoints_dict,
    visualize_texture_atlas,
)


def main():
    _, _, scan_to_vps = load_viewpoints_dict(cfg.DATA.CONNECTIVITY_DIR)
    scan_name = "17DRP5sb8fy"
    device = get_device()

    scan_paths = get_obj_paths(cfg.DATA.MESH_DIR, [scan_name])

    verts, faces, aux = load_obj(
        scan_paths[scan_name],
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=cfg.MESH.TEXTURE_ATLAS_SIZE,
        device=device,
    )
    atlas = aux.texture_atlas
    if atlas.ndim == 4:
        atlas = atlas.unsqueeze(0)

    print(atlas.shape)
    # the second dim controls how many faces you want to see
    visualize_texture_atlas(atlas[:, 10000:10100, ...], save_dir="temp")
    # visualize mesh texture atlas
    # plt.show()


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
