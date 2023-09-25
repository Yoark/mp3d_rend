import json
import os

import h5py
import MatterSim
import numpy as np
import torch
import tqdm
from PIL import Image

from mp3d_rend.render.config import get_cfg_defaults
from mp3d_rend.render.matterport_utils import build_simulator
from mp3d_rend.render.render import get_camera, get_mesh_renderer
from mp3d_rend.render.utils import (
    build_feature_extractor,
    create_folder,
    get_device,
    read_gz_jsonlines,
)
from pytorch3d.io.ply_io import _load_ply_raw
from iopath.common.file_io import PathManager
import ipdb
import pytorch3d 
from tqdm import tqdm
import torch
import pandas
# use MatterSim to initialize a view, make some actions (turn 360), render with pytorch3d
# TODO

def render_init(scan, mask=None):
    configs = get_cfg_defaults()
    configs.merge_from_file("/nfs/hpc/sw/xiangxi/VLN/VLN-HAMT/mp3d_rend/render_example/configs/mp3d_render.yaml")

    configs.freeze()
    print(configs)

    connectivity_dir = configs.DATA.CONNECTIVITY_DIR
    scan_dir = configs.DATA.MESH_DIR
    image_dir = configs.DATA.MATTERPORT_IMAGE_DIR

    renderer, mesh, atlas, verts, faces, textures = get_mesh_renderer(configs, scan, mask)
    sim = build_simulator(connectivity_dir, image_dir)
    #device = get_device()
    return renderer, mesh, sim, atlas, verts, faces, textures, configs

# TODO don't need this for now, just get images first
# model, transform = build_feature_extractor(
#     configs.RENDER.IMAGE_MODEL_NAME, configs.RENDER.IMAGE_MODEL_PATH, device=device
# )

#torch.set_grad_enabled(False)

def capture_observation(scan, vp, sim, renderer, mesh, configs, keyframe):
    #image_folder = configs.SAVE.IMAGE_DIR
    #save_folder = create_folder(image_folder)

    # if configs.TEST == True:
    #     scan_vps = list(scan_vps)


    scan_vp_images = []
    z=[]
    for ix in tqdm(range(36)):
        if ix == 0:
            sim.newEpisode([scan], [vp], [0], [np.deg2rad(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])
        state = sim.getState()[0]
        assert state.viewIndex == ix
        # save the rgb as image here for checking
        if ix in keyframe:
            camera = get_camera(configs,  (state.location.x,  state.location.y,  state.location.z), state.heading, state.elevation)
            image = renderer(mesh.cuda(), cameras=camera.cuda())
            #print(image.shape, 'fdasfdhafjkhfuikdshakjfhkjfhajkhfkjhdfgskjhgds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            image = image[0, ..., :3]
            scan_vp_images.append(image)
    return scan_vp_images
        # feature extractor?

def get_ins2vertidx(ply_f, obj_f, house_f, obj_num):
    """
    meta_house = open(house_f, 'r').readlines()
    meta_house = [x.replace('  ', ' ').split(' ') for x in meta_house]
    seg2ins = {}
    for i in meta_house:
      if i[0] == 'E':
          seg2ins[int(i[3])] = int(i[2])
    """
    meta_house = json.load(open(house_f, 'r'))['segGroups']
    meta_objs = []
    for i in meta_house:
        if i['id'] == obj_num:
            meta_objs.append(i)
    path_manager = PathManager()
    header, elements = _load_ply_raw(ply_f, path_manager=path_manager)
    obj = pytorch3d.io.load_obj(obj_f)
    obj_verts = []
    for i in range(len(obj[1].verts_idx)):
        tmp = []
        for j in range(3):
            tmp.append(obj[0][obj[1].verts_idx[i][j]])
        obj_verts.append(np.stack(tmp))
    obj_verts = np.stack(obj_verts)
    face = elements.get("face", None)
    face_head = next(head for head in header.elements if head.name == "face")
    z = [[x[0], x[1], x[3]] for x in face if not x[2] == -1]
    for i in z:
        if -1 == i[1]:
            print(i)
    vertex_ply = elements['vertex'][0][:, :3]
    tmp = {}
    obj_cate = {}
    for i in z:
        if not i[1] in tmp:
            tmp[i[1]] = []
            obj_cate[i[1]] = i[2]
            #print(i[1], i[2])
        assert obj_cate[i[1]] == i[2]
        tmp[i[1]].append(i[0][0])
        tmp[i[1]].append(i[0][1])
        tmp[i[1]].append(i[0][2])
    for i in tqdm(tmp):
        tmp[i] = list(set(tmp[i]))
        for j in range(len(tmp[i])):
            tmp[i][j] = vertex_ply[tmp[i][j]]
        tmp[i] = np.stack(tmp[i])
        tmp[i] = torch.from_numpy(tmp[i][np.newaxis, np.newaxis, :, :]).cuda()

    print('!!!!!!!!!!!!!!!!!!!!!' + 'number of instance: ' + str(len(tmp)) + '???????????????????????????')
   
    aligment_ind = []
    for i in range(len(obj_verts) // 2000 + (len(obj_verts) % 2000 > 0)):
        obj_dises = []
        with torch.no_grad():
            obj_vert = torch.from_numpy(obj_verts[i*2000:(i+1)*2000, :, np.newaxis, :]).cuda()
            for j in tqdm(tmp):
                dis = ((obj_vert - tmp[j])**2).sum(-1)**0.5
                obj_dises.append(dis.min(-1)[0].sum(-1))
            obj_dises = torch.stack(obj_dises)
            dis, ind = obj_dises.min(0)
            ind = ind.cpu().detach().numpy()
            zz = list(tmp.keys())
            for j in ind:
                #aligment_ind.append(seg2ins[zz[j]])
                aligment_ind.append(zz[j])
    aligment_ind = np.stack(aligment_ind)
    masks = []
    for meta_obj in meta_objs:
        masks.append(np.isin(aligment_ind, meta_obj['segments']))
    return aligment_ind, obj_cate, masks

def mask_check_face_in_ins(faces, vert_in_ins, idx):
    vert_idx = faces.verts_idx 
    mask = []
    for idxs in vert_idx:
        e = []
        for idx in idxs:
            e.append(idx in vert_in_ins[idx])
        e = np.array(e).sum()
        if e >= 2:
            mask.append(1)
        else:
            mask.append(0)
    return mask
if __name__ == '__main__':
    
    ply_f = '/nfs/hpc/sw/temp_shared_mp3d/matterport/images/v1/scans/Uxmj2M2itWa/Uxmj2M2itWa/house_segmentations/Uxmj2M2itWa.ply'
    obj_f = '/nfs/hpc/sw_data/virl-common-datasets/mp3d-data/v1/scans/Uxmj2M2itWa/matterport_mesh/b9116f2d4e0a44178d14fe804de4e518/b9116f2d4e0a44178d14fe804de4e518.obj'
    house_f = '/nfs/hpc/sw_data/virl-common-datasets/mp3d-data/v1/scans/Uxmj2M2itWa/house_segmentations/Uxmj2M2itWa.semseg.json'
    z, obj_cate, masks = get_ins2vertidx(ply_f, obj_f, house_f, [292])
    #print('111111111111111111111object category :' + str(obj_cate[292]) + '22222222222222222222')
    renderer, mesh, sim, atlas, verts, faces, textures, configs = render_init('Uxmj2M2itWa', masks[0])
    images =  capture_observation('Uxmj2M2itWa', '6028b7e2036c448999b5bd48d16a4175', sim, renderer, mesh, configs, list(range(36)))
    os.makedirs('mesh_test', exist_ok=True)
    count = 0
    for idx,i in enumerate(images):
        size = (i.sum(-1) < 2.99).sum() / i.shape[0] /i.shape[1]
        print(size)
        if size > count:
            count = size
        i = (i * 255).int().cpu().detach().numpy()
        i = Image.fromarray(i.astype(np.uint8)) 
        i.save('./mesh_test/%d.jpg'%(idx))
    print(count)

