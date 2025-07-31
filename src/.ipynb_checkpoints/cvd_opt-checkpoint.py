import sys
sys.path.append('cvd_opt')
sys.path.append('cvd_opt/core')

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.raft import RAFT
from core.utils.utils import InputPadder
from cvd_opt import consistency_loss, sobel_fg_alpha, si_loss, gradient_loss
from geometry_utils import NormalGenerator
from tqdm import tqdm
from lietorch import SE3

ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5

def warp_flow(img, flow):
  h, w = flow.shape[:2]
  flow_new = flow.copy()
  flow_new[:, :, 0] += np.arange(w)
  flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

  res = cv2.remap(
      img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
  )
  return res


def resize_flow(flow, img_h, img_w):
  # flow = np.load(flow_path)
  flow_h, flow_w = flow.shape[0], flow.shape[1]
  flow[:, :, 0] *= float(img_w) / float(flow_w)
  flow[:, :, 1] *= float(img_h) / float(flow_h)
  flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

  return flow


class CVDconfig:
    def __init__(self, model='raft-things.pth',num_heads=1, small=False, mixed_precision=True, position_and_content=False,
                 position_only=True):
        self.model = model
        self.small = small
        self.mixed_precision = mixed_precision
        self.position_and_content = position_and_content
        self.position_only =position_only 
        self.num_heads = num_heads
        
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

def preprocess_cvd(images, cvd_conf):
    
    model = torch.nn.DataParallel(RAFT(cvd_conf))
    model.load_state_dict(torch.load(cvd_conf.model))
    
    
    flow_model = model.module
    flow_model.cuda()
    flow_model.eval()
    
    img_data = []
    for t, (image) in enumerate(images):
        
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        _image = cv2.resize(image, (w1, h1))
        _image = image[: h1 - h1 % 8, : w1 - w1 % 8].transpose(2, 0, 1)
        img_data.append(_image)
    
    img_data = np.array(img_data)
    
    flows_low = []
    
    flows_high = []
    flow_masks_high = []
    
    flow_init = None
    flows_arr_low_bwd = {}
    flows_arr_low_fwd = {}
    
    ii = []
    jj = []
    flows_arr_up = []
    masks_arr_up = []
    
    for step in [1, 2, 4, 8, 15]:
        flows_arr_low = []
        for i in tqdm(range(max(0, -step), img_data.shape[0] - max(0, step))):
          image1 = torch.as_tensor(np.ascontiguousarray(img_data[i : i + 1])).float().cuda()
          image2 = torch.as_tensor(np.ascontiguousarray(img_data[i + step : i + step + 1])).float().cuda()
        
          ii.append(i)
          jj.append(i + step)
        
          with torch.no_grad():
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            if np.abs(step) > 1:
              flow_init = np.stack([flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0)
              flow_init = (torch.as_tensor(np.ascontiguousarray(flow_init)).float().cuda().permute(0, 3, 1, 2))
            else:
              flow_init = None
        
            flow_low, flow_up, _ = flow_model(torch.cat([image1, image2], dim=0), torch.cat([image2, image1], dim=0),iters=22,
                                              test_mode=True, flow_init=flow_init)
        
            flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
            flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)
        
            flow_up_fwd = resize_flow(flow_up[0].cpu().numpy().transpose(1, 2, 0), flow_up.shape[-2] // 2, flow_up.shape[-1] // 2)
            flow_up_bwd = resize_flow(flow_up[1].cpu().numpy().transpose(1, 2, 0), flow_up.shape[-2] // 2, flow_up.shape[-1] // 2)
        
            bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
            fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
            fwd_mask_up = fwd_lr_error < 1.0
        
            # flows_arr_low.append(flow_low_fwd)
            flows_arr_low_bwd[i + step] = flow_low_bwd
            flows_arr_low_fwd[i] = flow_low_fwd
        
            # masks_arr_low.append(fwd_mask_low)
            flows_arr_up.append(flow_up_fwd)
            masks_arr_up.append(fwd_mask_up)
    
    
    iijj = np.stack((ii, jj), axis=0)
    flows_high = np.float16(np.array(flows_arr_up).transpose(0, 3, 1, 2))
    flow_masks_high = np.array(masks_arr_up)[:, None, ...]
    
    return flow_masks_high, flows_high, iijj



def opt_cvd(images, disp_data, poses, flow_masks, flows, iijj, K, mot_prob):
    poses_th = torch.as_tensor(poses, device="cpu").float().cuda()
    flow_masks = np.float32(flow_masks)
    img_data_pt = torch.from_numpy(np.ascontiguousarray(images)).float().cuda() / 255.0
    flow_masks = (torch.from_numpy(np.ascontiguousarray(flow_masks)).float().cuda())
    flows = torch.from_numpy(np.ascontiguousarray(flows)).float().cuda()
    iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().cuda()
    ii = iijj[0, ...].long()
    jj = iijj[1, ...].long()
    K = torch.from_numpy(K).float().cuda()
    init_disp = torch.from_numpy(disp_data).float().cuda()
    disp_data = torch.from_numpy(disp_data).float().cuda()
    assert init_disp.shape == disp_data.shape
    init_disp = torch.nn.functional.interpolate(init_disp.unsqueeze(1), scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
                                                mode="bilinear").squeeze(1)
    disp_data = torch.nn.functional.interpolate(disp_data.unsqueeze(1), scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
                                                mode="bilinear").squeeze(1)
    
    
    fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
    fg_alpha = fg_alpha.squeeze(1).float() + 0.2
    
    cvd_prob = torch.nn.functional.interpolate(torch.from_numpy(mot_prob).unsqueeze(1).cuda(), scale_factor=(4, 4), mode="bilinear")
    cvd_prob[cvd_prob > 0.5] = 0.5
    cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)
    
    # rescale intrinsic matrix to small resolution
    K_o = K.clone()
    K[0:2, ...] *= RESIZE_FACTOR
    K_inv = torch.linalg.inv(K)
    
    disp_data.requires_grad = False
    poses_th.requires_grad = False
    
    uncertainty = cvd_prob
    
    # First optimize scale and shift to align them
    log_scale_ = torch.log(torch.ones(init_disp.shape[0]).to(disp_data.device))
    shift_ = torch.zeros(init_disp.shape[0]).to(disp_data.device)
    log_scale_.requires_grad = True
    shift_.requires_grad = True
    uncertainty.requires_grad = True
    
    optim = torch.optim.Adam([{"params": log_scale_, "lr": 1e-2}, {"params": shift_, "lr": 1e-2}, {"params": uncertainty, "lr": 1e-2}])
    
    compute_normals = []
    compute_normals.append(NormalGenerator(disp_data.shape[-2], disp_data.shape[-1]))
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)
    
    for i in range(100):
        optim.zero_grad()
        scale_ = torch.exp(log_scale_)
        cam_c2w = SE3(poses_th).inv().matrix()
        loss = consistency_loss(cam_c2w, K, K_inv, torch.clamp(disp_data * scale_[..., None, None] + shift_[..., None, None], 1e-3, 1e3),
                                init_disp, torch.clamp(uncertainty, 1e-4, 1e3), flows, flow_masks, ii, jj, compute_normals, fg_alpha)
        
        loss.backward()
        uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
        log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
        shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)
        
        optim.step()
        print("step ", i, loss.item())
    # Then optimize depth and uncertainty
    disp_data = (disp_data * torch.exp(log_scale_)[..., None, None].detach() + shift_[..., None, None].detach())
    init_disp = (init_disp * torch.exp(log_scale_)[..., None, None].detach() + shift_[..., None, None].detach())
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)
    disp_data.requires_grad = True
    uncertainty.requires_grad = True
    poses_th.requires_grad = False  # True

    optim = torch.optim.Adam([{"params": disp_data, "lr": 5e-3}, {"params": uncertainty, "lr": 5e-3}])

    losses = []
    for i in range(400):
        optim.zero_grad()
        cam_c2w = SE3(poses_th).inv().matrix()
        loss = consistency_loss(cam_c2w, K,  K_inv, torch.clamp(disp_data, 1e-3, 1e3), init_disp, torch.clamp(uncertainty, 1e-4, 1e3), flows,
                                flow_masks, ii, jj, compute_normals, fg_alpha, w_ratio=1.0, w_flow=0.2, w_si=1, w_grad=2, w_normal=5)
        
        loss.backward()
        disp_data.grad = torch.nan_to_num(disp_data.grad, nan=0.0)
        uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
        
        optim.step()
        print("step ", i, loss.item())
        losses.append(loss)
    disp_data_opt = torch.nn.functional.interpolate(disp_data.unsqueeze(1), scale_factor=(2, 2), mode="bilinear").squeeze(1).detach().cpu().numpy()
    depths = np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2)
    intrinsic = K_o.detach().cpu().numpy()
    cam_c2w = cam_c2w.detach().cpu().numpy()

    return disp_data_opt, depths, intrinsic, cam_c2w