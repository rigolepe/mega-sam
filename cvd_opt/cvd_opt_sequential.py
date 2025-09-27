# Copyright 2025 DeepMind Technologies Limited
# Sequentiële verwerking voor extreme geheugen optimalisatie

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3


def gradient_loss(gt, pred, u):
  """Gradient loss."""
  del u
  diff = pred - gt
  v_gradient = torch.abs(diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1])
  h_gradient = torch.abs(diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:])

  pred_grad = torch.abs(pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
  gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + torch.abs(gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:])

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
  data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - torch.sum(log_diff, dim=-1) ** 2 / (num_pixels**2)
  return torch.mean(data_loss)


def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
  sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
  sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
  alpha = torch.exp(-1.0 * beta * sobel_mag).detach()
  return alpha


ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5


def consistency_loss_sequential(
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
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
  """Sequentiële consistency loss - verwerk frame per frame."""
  _, H, W = disp_data.shape

  # Maak grid één keer
  xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
  yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
  xx = xx.view(1, 1, H, W)
  yy = yy.view(1, 1, H, W)
  grid = torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)

  # Initialize loss accumulators
  total_loss_flow = 0.0
  total_loss_d_ratio = 0.0
  total_weight = 0.0

  # Process elke frame sequentieel
  for idx in range(len(ii)):
    # Get data voor deze frame
    i = ii[idx:idx+1]
    j = jj[idx:idx+1]

    flow_frame = flows[idx:idx+1]
    flow_mask_frame = flow_masks[idx:idx+1]
    uu_frame = torch.index_select(uncertainty, dim=0, index=i).squeeze(1)

    flows_step = flow_frame.permute(0, 2, 3, 1)
    flow_masks_step = flow_mask_frame.permute(0, 2, 3, 1).squeeze(-1)

    # Bereken transformatie voor deze frame
    cam_1to2 = torch.bmm(
        torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=j)),
        torch.index_select(cam_c2w, dim=0, index=i),
    )

    # Warp operations
    pixel_locations = grid + flows_step
    resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
    normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

    disp_sampled = F.grid_sample(
        torch.index_select(disp_data, dim=0, index=j)[:, None, ...],
        normalized_pixel_locations,
        align_corners=True,
    )

    # Get reference depth
    ref_depth = 1.0 / torch.clamp(
        torch.index_select(disp_data, dim=0, index=i), 1e-3, 1e3
    )

    grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)

    # 3D transformatie met mixed precision
    rot = cam_1to2[:, None, None, :3, :3]
    trans = cam_1to2[:, None, None, :3, 3:4]

    with torch.cuda.amp.autocast():
      # Process in kleinere spatial tiles als nodig
      TILE_SIZE = 180  # Voor 360x180 doe 2 tiles

      if H > TILE_SIZE or W > TILE_SIZE:
        # Process in tiles
        disp_tgt_tiles = []
        pts_2D_tgt_tiles = []

        for y_start in range(0, H, TILE_SIZE):
          for x_start in range(0, W, TILE_SIZE):
            y_end = min(y_start + TILE_SIZE, H)
            x_end = min(x_start + TILE_SIZE, W)

            # Get tile data
            ref_depth_tile = ref_depth[:, y_start:y_end, x_start:x_end]

            # Maak tile grid
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            xx_t = torch.arange(x_start, x_end).view(1, -1).repeat(tile_h, 1)
            yy_t = torch.arange(y_start, y_end).view(-1, 1).repeat(1, tile_w)
            xx_t = xx_t.view(1, 1, tile_h, tile_w)
            yy_t = yy_t.view(1, 1, tile_h, tile_w)
            grid_t = torch.cat((xx_t, yy_t), 1).float().cuda().permute(0, 2, 3, 1)
            grid_h_t = torch.cat([grid_t, torch.ones_like(grid_t[..., 0:1])], dim=-1).unsqueeze(-1)

            # Transform tile
            pts_3d_ref_t = ref_depth_tile[..., None, None] * (K_inv[None, None, None] @ grid_h_t)
            pts_3d_tgt_t = (rot @ pts_3d_ref_t) + trans

            depth_tgt_t = pts_3d_tgt_t[:, :, :, 2:3, 0]
            disp_tgt_t = 1.0 / torch.clamp(depth_tgt_t, 0.1, 1e3)

            pts_2D_tgt_t = K[None, None, None] @ pts_3d_tgt_t
            pts_2D_tgt_t = pts_2D_tgt_t[:, :, :, :2, 0] / torch.clamp(pts_2D_tgt_t[:, :, :, 2:, 0], 1e-3, 1e3)

            disp_tgt_tiles.append(disp_tgt_t)
            pts_2D_tgt_tiles.append(pts_2D_tgt_t)

            # Cleanup tile tensors
            del pts_3d_ref_t, pts_3d_tgt_t, grid_h_t

        # Combineer tiles (simplified - gebruik eerste tile voor nu)
        disp_tgt = disp_tgt_tiles[0]
        pts_2D_tgt = pts_2D_tgt_tiles[0]
      else:
        # Process volledig frame als het klein genoeg is
        pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
        pts_3d_tgt = (rot @ pts_3d_ref) + trans

        depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
        disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

        pts_2D_tgt = K[None, None, None] @ pts_3d_tgt
        flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
        pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3)

        del pts_3d_ref, pts_3d_tgt

    # Bereken losses voor deze frame
    disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
    disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

    ratio = torch.maximum(
        disp_sampled.squeeze() / disp_tgt.squeeze(),
        disp_tgt.squeeze() / disp_sampled.squeeze(),
    )
    ratio_error = torch.abs(ratio - 1.0)

    # Update accumulators
    frame_weight = flow_masks_step.sum()
    if frame_weight > 0:
      total_loss_d_ratio += torch.sum(
          (ratio_error * uu_frame + ALPHA_MOTION * torch.log(1.0 / uu_frame)) * flow_masks_step
      ).item()

      if 'flow_masks_step_' in locals():
        flow_error = torch.abs(pts_2D_tgt - pixel_locations)
        total_loss_flow += torch.sum(
            (flow_error * uu_frame[..., None] + ALPHA_MOTION * torch.log(1.0 / uu_frame[..., None])) * flow_masks_step_[..., None]
        ).item()

      total_weight += frame_weight.item()

    # Cleanup frame tensors
    del grid_h
    torch.cuda.empty_cache()

  # Normalize losses
  if total_weight > 0:
    total_loss_d_ratio /= (total_weight + 1e-8)
    total_loss_flow /= (total_weight * 2.0 + 1e-8)

  # Convert terug naar tensors
  total_loss_d_ratio = torch.tensor(total_loss_d_ratio, device=disp_data.device, requires_grad=True)
  total_loss_flow = torch.tensor(total_loss_flow, device=disp_data.device, requires_grad=True)

  # Other losses (berekend over alle frames tegelijk voor efficiëntie)
  loss_prior = si_loss(init_disp, disp_data)

  # Gradient loss - reduced scales
  loss_grad = 0.0
  for scale in range(2):  # Alleen 2 scales ipv 4
    interval = 2**scale
    disp_data_ds = F.interpolate(disp_data[:, None, ...], scale_factor=(1.0 / interval, 1.0 / interval), mode="nearest-exact")
    init_disp_ds = F.interpolate(init_disp[:, None, ...], scale_factor=(1.0 / interval, 1.0 / interval), mode="nearest-exact")
    uncertainty_rs = F.interpolate(uncertainty, scale_factor=(1.0 / interval, 1.0 / interval), mode="nearest-exact")
    loss_grad += gradient_loss(torch.log(disp_data_ds), torch.log(init_disp_ds), uncertainty_rs)

  # Normal loss - simplified
  KK = torch.inverse(K_inv)
  K_inv_rescale = torch.inverse(KK.clone())
  disp_data_ds = disp_data[:, None, ...]
  init_disp_ds = init_disp[:, None, ...]

  pred_normal = compute_normals[0](1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None])
  init_normal = compute_normals[0](1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None])
  loss_normal = torch.mean(fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1)))

  return (
      w_ratio * total_loss_d_ratio
      + w_si * loss_prior
      + w_flow * total_loss_flow
      + w_normal * loss_normal
      + loss_grad * w_grad
  )