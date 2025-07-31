import sys

sys.path.append("Depth-Anything")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from depth_anything.dpt import DPT_DINOv2


def depth_anything(images, load_from='Depth-Anything/checkpoints/depth_anything_vitl14.pth'):
    depth_anything = DPT_DINOv2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            localhub=False,
        ).cuda()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    
    depth_anything.load_state_dict(
      torch.load(load_from, map_location='cpu'), strict=True
    )
    depth_anything.eval()
    transform = Compose([
      Resize(
          width=768,
          height=768,
          resize_target=False,
          keep_aspect_ratio=True,
          ensure_multiple_of=14,
          resize_method='upper_bound',
          image_interpolation_method=cv2.INTER_CUBIC,
      ),
      NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      PrepareForNet(),
    ])
    
    depths = []
    for i, image in enumerate(images):
        print(i, end='\r')
        h, w = image.shape[:2]
        image = transform({'image': image / 255.0})['image']
        image = torch.from_numpy(image).unsqueeze(0).cuda()
        with torch.no_grad():
          depth = depth_anything(image)
        depth = F.interpolate(
            depth[None], (h, w), mode='bilinear', align_corners=False
        )[0, 0]
        depth_npy = np.float32(depth.cpu().numpy())
        depths.append(depth_npy)
    depths = np.array(depths)
    return depths