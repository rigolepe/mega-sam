# Copyright 2025 DeepMind Technologies Limited
# Tile-based CVD optimization voor extreme geheugen besparing

import argparse
import os
from pathlib import Path

from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3
import numpy as np
import torch
import torch.nn.functional as F

ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.25
TILE_SIZE = 64  # Process in tiles van 64x64 pixels


def gradient_loss(gt, pred, u):
  """Gradient loss."""
  del u
  diff = pred - gt
  v_gradient = torch.abs(
      diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1]
  )
  h_gradient = torch.abs(
      diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:]
  )

  pred_grad = torch.abs(
      pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])
  ) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
  gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + torch.abs(
      gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:]
  )

  grad_diff = torch.abs(pred_grad - gt_grad)
  nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
  weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
  weight *= nearby_mask

  g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
  return g_loss


def si_loss(gt, pred):
  log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
  log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
  log_diff = log_gt - log_pred
  num_pixels = gt.shape[-2] * gt.shape[-1]
  data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - torch.sum(
      log_diff, dim=-1
  ) ** 2 / (num_pixels**2)
  return torch.mean(data_loss)


def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
  sobel_grad = kornia.filters.spatial_gradient(
      disp, mode=mode, normalized=False
  )
  sobel_mag = torch.sqrt(
      sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2
  )
  alpha = torch.exp(-1.0 * beta * sobel_mag).detach()
  return alpha


def consistency_loss_tiled(
    cam_c2w,
    K,
    K_inv,
    disp_data,
    init_disp,
    uncertainty,
    flows,
    flow_masks,
    ii,
    jj,
    compute_normals,
    fg_alpha,
    tile_size=TILE_SIZE,
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
  """Tile-based consistency loss voor extreme geheugen besparing."""
  batch_size, H, W = disp_data.shape

  # Bereken transformatie matrices één keer
  cam_1to2 = torch.bmm(
      torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
      torch.index_select(cam_c2w, dim=0, index=ii),
  )
  rot = cam_1to2[:, :3, :3]
  trans = cam_1to2[:, :3, 3:4]

  # Initialize loss accumulators
  total_loss_flow = 0.0
  total_loss_d_ratio = 0.0
  total_weight = 0.0

  # Process per frame om geheugen te besparen
  for batch_idx in range(batch_size):
    # Process in tiles
    for y_start in range(0, H, tile_size):
      for x_start in range(0, W, tile_size):
        y_end = min(y_start + tile_size, H)
        x_end = min(x_start + tile_size, W)
        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Maak grid voor deze tile
        xx = torch.arange(x_start, x_end).view(1, -1).repeat(tile_h, 1)
        yy = torch.arange(y_start, y_end).view(-1, 1).repeat(1, tile_w)
        xx = xx.view(1, tile_h, tile_w)
        yy = yy.view(1, tile_h, tile_w)
        grid = torch.cat((xx, yy), 0).float().cuda().permute(1, 2, 0).unsqueeze(0)

        # Get tile data
        flows_tile = flows[batch_idx:batch_idx+1, :, y_start:y_end, x_start:x_end]
        flow_masks_tile = flow_masks[batch_idx:batch_idx+1, :, y_start:y_end, x_start:x_end]

        flows_step = flows_tile.permute(0, 2, 3, 1)
        flow_masks_step = flow_masks_tile.permute(0, 2, 3, 1).squeeze(-1)

        # Warp disp from target time
        pixel_locations = grid + flows_step
        resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
        normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

        disp_sampled = F.grid_sample(
            torch.index_select(disp_data, dim=0, index=jj[batch_idx:batch_idx+1])[:, None, ...],
            normalized_pixel_locations,
            align_corners=True,
        )

        # Get reference depth for tile
        ref_depth_tile = 1.0 / torch.clamp(
            torch.index_select(disp_data, dim=0, index=ii[batch_idx:batch_idx+1])[:, y_start:y_end, x_start:x_end],
            1e-3, 1e3
        )

        # 3D transform for tile
        grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)

        with torch.cuda.amp.autocast():
          pts_3d_ref = ref_depth_tile[..., None, None] * (K_inv[None, None, None] @ grid_h)
          pts_3d_tgt = (rot[batch_idx:batch_idx+1, None, None] @ pts_3d_ref) + trans[batch_idx:batch_idx+1, None, None]

        depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
        disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

        # Compute losses for tile
        disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
        disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

        ratio = torch.maximum(
            disp_sampled.squeeze() / disp_tgt.squeeze(),
            disp_tgt.squeeze() / disp_sampled.squeeze(),
        )

        # Accumulate losses
        tile_weight = flow_masks_step.sum()
        if tile_weight > 0:
          total_loss_d_ratio += ((ratio - 1.0).abs() * flow_masks_step).sum()
          total_weight += tile_weight

        # Free memory
        del pts_3d_ref, pts_3d_tgt, grid_h, grid
        torch.cuda.empty_cache()

  # Normalize losses
  if total_weight > 0:
    total_loss_d_ratio /= total_weight

  # Add other loss terms (simplified)
  loss_prior = si_loss(init_disp, disp_data)

  return w_ratio * total_loss_d_ratio + w_si * loss_prior


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--w_grad", type=float, default=2.0)
  parser.add_argument("--w_normal", type=float, default=6.0)
  parser.add_argument("--output_dir", type=str, default="outputs_cvd")
  parser.add_argument("--scene_name", type=str, help="scene name")
  args = parser.parse_args()

  cache_dir = "./cache_flow"
  rootdir = os.getcwd() + "/reconstructions"

  output_dir = args.output_dir
  scene_name = args.scene_name
  print("***************************** ", scene_name)

  # Load data
  img_data = np.load(os.path.join(rootdir, scene_name, "images.npy"))[:, ::-1, ...]
  disp_data = np.load(os.path.join(rootdir, scene_name.replace("_opt", ""), "disps.npy")) + 1e-6
  intrinsics = np.load(os.path.join(rootdir, scene_name, "intrinsics.npy"))
  poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
  mot_prob = np.load(os.path.join(rootdir, scene_name, "motion_prob.npy"))

  flows = np.load("%s/%s/flows.npy" % (cache_dir, scene_name), allow_pickle=True)
  flow_masks = np.load("%s/%s/flows_masks.npy" % (cache_dir, scene_name), allow_pickle=True)
  flow_masks = np.float32(flow_masks)
  iijj = np.load("%s/%s/ii-jj.npy" % (cache_dir, scene_name), allow_pickle=True)

  intrinsics = intrinsics[0]
  poses_th = torch.as_tensor(poses, device="cpu").float().cuda()

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]

  img_data_pt = torch.from_numpy(np.ascontiguousarray(img_data)).float().cuda() / 255.0
  flows = torch.from_numpy(np.ascontiguousarray(flows)).float().cuda()
  flow_masks = torch.from_numpy(np.ascontiguousarray(flow_masks)).float().cuda()
  iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().cuda()
  ii = iijj[0, ...].long()
  jj = iijj[1, ...].long()
  K = torch.from_numpy(K).float().cuda()

  init_disp = torch.from_numpy(disp_data).float().cuda()
  disp_data = torch.from_numpy(disp_data).float().cuda()

  # Resize everything
  init_disp = F.interpolate(init_disp.unsqueeze(1), scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR), mode="bilinear").squeeze(1)
  disp_data = F.interpolate(disp_data.unsqueeze(1), scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR), mode="bilinear").squeeze(1)

  # Resize flows to match
  target_H, target_W = disp_data.shape[-2:]
  original_flow_H, original_flow_W = flows.shape[-2:]
  flows = F.interpolate(flows, size=(target_H, target_W), mode="bilinear", align_corners=True)
  flows[:, 0, :, :] *= (target_W / original_flow_W)
  flows[:, 1, :, :] *= (target_H / original_flow_H)
  flow_masks = F.interpolate(flow_masks, size=(target_H, target_W), mode="nearest")

  fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
  fg_alpha = fg_alpha.squeeze(1).float() + 0.2

  cvd_prob = F.interpolate(torch.from_numpy(mot_prob).unsqueeze(1).cuda(), scale_factor=(4, 4), mode="bilinear")
  cvd_prob[cvd_prob > 0.5] = 0.5
  cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)

  # Rescale intrinsic matrix
  K_o = K.clone()
  K[0:2, ...] *= RESIZE_FACTOR
  K_inv = torch.linalg.inv(K)

  disp_data.requires_grad = False
  poses_th.requires_grad = False
  uncertainty = cvd_prob

  # Optimize scale and shift
  log_scale_ = torch.log(torch.ones(init_disp.shape[0]).to(disp_data.device))
  shift_ = torch.zeros(init_disp.shape[0]).to(disp_data.device)
  log_scale_.requires_grad = True
  shift_.requires_grad = True
  uncertainty.requires_grad = True

  optim = torch.optim.Adam([
      {"params": log_scale_, "lr": 1e-2},
      {"params": shift_, "lr": 1e-2},
      {"params": uncertainty, "lr": 1e-2},
  ])

  compute_normals = []
  compute_normals.append(NormalGenerator(disp_data.shape[-2], disp_data.shape[-1]))
  init_disp = torch.clamp(init_disp, 1e-3, 1e3)

  print("Starting tiled optimization...")
  for i in range(100):
    optim.zero_grad()
    cam_c2w = SE3(poses_th).inv().matrix()
    scale_ = torch.exp(log_scale_)

    loss = consistency_loss_tiled(
        cam_c2w,
        K,
        K_inv,
        torch.clamp(disp_data * scale_[..., None, None] + shift_[..., None, None], 1e-3, 1e3),
        init_disp,
        torch.clamp(uncertainty, 1e-4, 1e3),
        flows,
        flow_masks,
        ii,
        jj,
        compute_normals,
        fg_alpha,
    )

    loss.backward()
    uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
    log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
    shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)

    optim.step()
    print("step ", i, loss.item())

    if i % 10 == 0:
      torch.cuda.empty_cache()

  print("Tiled optimization completed!")

  # Save results
  disp_data_opt = F.interpolate(disp_data.unsqueeze(1), scale_factor=(2, 2), mode="bilinear").squeeze(1).detach().cpu().numpy()

  Path(output_dir).mkdir(parents=True, exist_ok=True)
  np.savez(
      "%s/%s_sgd_cvd_tiled.npz" % (output_dir, scene_name),
      images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
      depths=np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2),
      intrinsic=K_o.detach().cpu().numpy(),
      cam_c2w=cam_c2w.detach().cpu().numpy(),
  )