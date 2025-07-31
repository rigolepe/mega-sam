import sys

sys.path.append("UniDepth")


import cv2
import numpy as np
import torch
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid

def uni_depth(images, LONG_DIM=640):

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    depths = []
    fov = []
    
    for i, rgb in enumerate(images):
        print(i, end='\r')
        if rgb.shape[1] > rgb.shape[0]:
          final_w, final_h = LONG_DIM, int(round(LONG_DIM * rgb.shape[0] / rgb.shape[1]))
        else:
          final_w, final_h = (int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])), LONG_DIM)
        rgb = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
        predictions = model.infer(rgb_torch)
        fov_ = np.rad2deg(2 * np.arctan(predictions["depth"].shape[-1] / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())))
        depth = predictions["depth"][0, 0].cpu().numpy()
        
        depths.append(depth)
        fov.append(fov_)
        
    depths = np.array(depths)
    fov = np.array(fov)
    
    return depths, fov