import sys

sys.path.append("base/droid_slam")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from droid import Droid
from tqdm import tqdm
from lietorch import SE3



def image_stream(images, mono_disp_list, use_depth=False, aligns=None, K=None):
  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )

  for t, image in enumerate(images):
      
    mono_disp = mono_disp_list[t]
    depth = np.clip(1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])), 1e-4, 1e4,)
    depth[depth < 1e-2] = 0.0


    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]
    image = torch.as_tensor(image).permute(2, 0, 1)

    depth = torch.as_tensor(depth)
    depth = F.interpolate(depth[None, None], (h1, w1), mode="nearest-exact").squeeze()
    depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

    mask = torch.ones_like(depth)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= w1 / w0
    intrinsics[1::2] *= h1 / h0

    if use_depth:
      yield t, image[None], depth, intrinsics, mask
    else:
      yield t, image[None], intrinsics, mask


class DroidConfig:
    def __init__(self, weights="droid.pth", disable_vis=True, image_size=[240, 320], buffer=1024,
                 stereo=False, depth=False, filter_thresh=2.0, upsample=False, warmup=8, beta=0.3, 
                 frontend_nms=1, backend_thresh=16.0, backend_radius=2, backend_nms=3,
                 keyframe_thresh=2.0, frontend_window=25, frontend_thresh=12.0, frontend_radius=2):
        
        self.weights = weights
        self.disable_vis = disable_vis
        self.image_size = image_size
        self.buffer = buffer
        self.stereo = stereo
        self.filter_thresh = filter_thresh
        self.upsample = upsample
        self.warmup = warmup
        self.beta = beta
        self.frontend_nms = frontend_nms
        self.keyframe_thresh = keyframe_thresh
        self.frontend_window = frontend_window
        self.frontend_thresh = frontend_thresh
        self.frontend_radius = frontend_radius
        self.backend_thresh = backend_thresh
        self.backend_radius = backend_radius
        self.backend_nms = backend_nms
        self.depth = depth


def megasam(images, metric_depths, mono_depths,fovs, weights='checkpoints/megasam_final.pth'):

    config = DroidConfig(weights)
    
    img_0 = images[0]
    
    tstamps = []
    rgb_list = []
    senor_depth_list = []
    scales = []
    shifts = []
    mono_disp_list = []
    
    for i in range(len(images)):
        da_disp = mono_depths[i]
        metric_depth = metric_depths[i]
        
        da_disp = cv2.resize(da_disp, (metric_depth.shape[1], metric_depth.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        
        mono_disp_list.append(da_disp)
        
        gt_disp = 1.0 / (metric_depth + 1e-8)
        
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2
        
        sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
        
        if sky_ratio > 0.5:
          non_sky_mask = da_disp > 0.01
          gt_disp_ms = (gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8)
          da_disp_ms = (da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8)
          scale = np.median(gt_disp_ms / da_disp_ms)
          shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
        else:
          gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
          da_disp_ms = da_disp - np.median(da_disp) + 1e-8
          scale = np.median(gt_disp_ms / da_disp_ms)
          shift = np.median(gt_disp - scale * da_disp)
        
        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = da_disp - np.median(da_disp) + 1e-8
        
        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp - scale * da_disp)
        
        scales.append(scale)
        shifts.append(shift)
    
    print("************** UNIDEPTH FOV ", np.median(fovs))
    
    ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
    K = np.eye(3)
    K[0, 0] = (ff * 1.0)  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
    K[1, 1] = (ff * 1.0)  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
    K[0, 2] = (img_0.shape[1] / 2.0)  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
    K[1, 2] = (img_0.shape[0] / 2.0)  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))
    
    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))
    
    align_scale = scales[med_idx]  # np.median(np.array(scales))
    align_shift = shifts[med_idx]  # np.median(np.array(shifts))
    normalize_scale = (np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98)  / 2.0)
    
    aligns = (align_scale, align_shift, normalize_scale)
    
    for t, image, depth, intrinsics, mask in tqdm(image_stream(images, mono_disp_list, use_depth=True, aligns=aligns, K=K,)):
    
        rgb_list.append(image[0])
        senor_depth_list.append(depth)
    
        if t == 0:
          config.image_size = [image.shape[2], image.shape[3]]
          droid = Droid(config)
    
        droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)
    
    traj_est, depth_est, motion_prob = droid.terminate(image_stream(images, mono_disp_list, use_depth=True,aligns=aligns,K=K,), _opt_intr=True, full_ba=True, scene_name='')
    
    t = traj_est.shape[0]
    disps = 1.0 / (np.array([np.array(a) for a in depth_est[:t]]) + 1e-6)
    poses = traj_est
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()[0] * 8.0
    poses_th = torch.as_tensor(poses, device="cpu")
    cam_c2w = SE3(poses_th).inv().matrix().numpy()
    
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    
    max_frames = min(1000, images.shape[0])
    
    depths = np.float32(1.0 / disps[:max_frames, ...])
    intrinsic = K
    cam_c2w = cam_c2w[:max_frames]
    
    return depths, disps, intrinsic, cam_c2w, motion_prob, poses, K