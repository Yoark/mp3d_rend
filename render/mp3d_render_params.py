import numpy as np

MESH_DIR = "/mnt/sw/vln/data/matterport3d/mp3d_mesh/v1/scans/{}/matterport_mesh"
VFOV = 60
WIDTH = 640
HEIGHT = 480
FOV = VFOV * WIDTH / HEIGHT
HEADINGS = [np.deg2rad(30.0 * h) for h in range(12)]
ELEVATIONS = [np.deg2rad(e) for e in [-30.0, 0, 30]]
CONNECTIVITY_DIR = "/mnt/sw/vln/data/matterport3d/connectivity"
pixel_aspect_ratio = 1  # pixel aspect
image_aspect_ratio = 4 / 3  # image aspect
image_size = (WIDTH, HEIGHT)
