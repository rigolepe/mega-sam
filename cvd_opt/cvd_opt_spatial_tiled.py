# Copyright 2025 DeepMind Technologies Limited
# Spatial tiling voor GPU memory optimalisatie met RESIZE_FACTOR=0.5

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


def consistency_loss_spatial_tiled(
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
    spatial_tile_size=128,  # Process ruimtelijk in tiles van 128x128
    batch_tile_size=2,  # Process max 2 frames tegelijk
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
  """Spatial tiling consistency loss voor geheugen optimalisatie."""
  batch_size, H, W = disp_data.shape

  # Initialize totale losses
  total_loss_d_ratio = 0.0
  total_loss_flow = 0.0
  total_weight = 0.0

  # Process in batch tiles
  for batch_start in range(0, batch_size, batch_tile_size):
    batch_end = min(batch_start + batch_tile_size, batch_size)
    batch_indices = torch.arange(batch_start, batch_end).long().cuda()

    # Get batch tile indices
    batch_ii = ii[batch_indices]
    batch_jj = jj[batch_indices]

    # Bereken transformatie voor deze batch
    cam_1to2_batch = torch.bmm(
        torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=batch_jj)),
        torch.index_select(cam_c2w, dim=0, index=batch_ii),
    )
    rot_batch = cam_1to2_batch[:, None, None, :3, :3]
    trans_batch = cam_1to2_batch[:, None, None, :3, 3:4]

    # Process spatial tiles
    for y_start in range(0, H, spatial_tile_size):
      y_end = min(y_start + spatial_tile_size, H)
      for x_start in range(0, W, spatial_tile_size):
        x_end = min(x_start + spatial_tile_size, W)
        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Maak grid voor deze spatial tile
        xx = torch.arange(x_start, x_end).view(1, -1).repeat(tile_h, 1)
        yy = torch.arange(y_start, y_end).view(-1, 1).repeat(1, tile_w)
        xx = xx.view(1, 1, tile_h, tile_w)
        yy = yy.view(1, 1, tile_h, tile_w)
        grid = torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)

        # Get tile data voor deze batch
        flows_tile = flows[batch_indices, :, y_start:y_end, x_start:x_end]
        flow_masks_tile = flow_masks[batch_indices, :, y_start:y_end, x_start:x_end]
        uu_tile = torch.index_select(uncertainty, dim=0, index=batch_ii)[:, :, y_start:y_end, x_start:x_end].squeeze(1)

        flows_step = flows_tile.permute(0, 2, 3, 1)
        flow_masks_step = flow_masks_tile.permute(0, 2, 3, 1).squeeze(-1)

        # Warp operations
        pixel_locations = grid + flows_step
        resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
        normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

        disp_sampled = F.grid_sample(
            torch.index_select(disp_data, dim=0, index=batch_jj)[:, None, ...],
            normalized_pixel_locations,
            align_corners=True,
        )

        # 3D transformatie voor tile
        ref_depth_tile = 1.0 / torch.clamp(
            torch.index_select(disp_data, dim=0, index=batch_ii)[:, y_start:y_end, x_start:x_end],
            1e-3, 1e3
        )

        grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)

        # Process elke frame in batch apart voor 3D transformatie
        pts_3d_tgt_list = []
        for i in range(batch_end - batch_start):
          with torch.cuda.amp.autocast():
            pts_3d_ref_i = ref_depth_tile[i:i+1, ..., None, None] * (K_inv[None, None, None] @ grid_h)
            pts_3d_tgt_i = (rot_batch[i:i+1] @ pts_3d_ref_i) + trans_batch[i:i+1]
            pts_3d_tgt_list.append(pts_3d_tgt_i.float())
            del pts_3d_ref_i

        pts_3d_tgt = torch.cat(pts_3d_tgt_list, dim=0)
        del pts_3d_tgt_list

        depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
        disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

        # Flow consistency loss
        pts_2D_tgt = K[None, None, None] @ pts_3d_tgt
        flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
        pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3)

        disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
        disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

        ratio = torch.maximum(
            disp_sampled.squeeze() / disp_tgt.squeeze(),
            disp_tgt.squeeze() / disp_sampled.squeeze(),
        )
        ratio_error = torch.abs(ratio - 1.0)

        # Accumulate tile losses
        tile_weight = flow_masks_step_.sum()
        if tile_weight > 0:
          total_loss_d_ratio += torch.sum(
              (ratio_error * uu_tile + ALPHA_MOTION * torch.log(1.0 / uu_tile)) * flow_masks_step_
          )
          flow_error = torch.abs(pts_2D_tgt - pixel_locations)
          total_loss_flow += torch.sum(
              (flow_error * uu_tile[..., None] + ALPHA_MOTION * torch.log(1.0 / uu_tile[..., None])) * flow_masks_step_[..., None]
          )
          total_weight += tile_weight

        # Free memory
        del pts_3d_tgt, grid_h, grid
        torch.cuda.empty_cache()

  # Normalize losses
  if total_weight > 0:
    total_loss_d_ratio /= (total_weight + 1e-8)
    total_loss_flow /= (total_weight * 2.0 + 1e-8)

  # Other loss terms
  loss_prior = si_loss(init_disp, disp_data)

  # Simplified gradient and normal losses
  loss_grad = 0.0
  loss_normal = 0.0

  KK = torch.inverse(K_inv)
  K_rescale = KK.clone()
  K_inv_rescale = torch.inverse(K_rescale)

  # Process gradient loss in smaller chunks
  for scale in range(2):  # Reduced from 4 to 2 for memory
    interval = 2**scale
    disp_data_ds = F.interpolate(disp_data[:, None, ...], scale_factor=(1.0 / interval, 1.0 / interval), mode="nearest-exact")
    init_disp_ds = F.interpolate(init_disp[:, None, ...], scale_factor=(1.0 / interval, 1.0 / interval), mode="nearest-exact")
    uncertainty_rs = F.interpolate(uncertainty, scale_factor=(1.0 / interval, 1.0 / interval), mode="nearest-exact")
    loss_grad += gradient_loss(torch.log(disp_data_ds), torch.log(init_disp_ds), uncertainty_rs)

  # Normal loss (simplified)
  disp_data_ds = disp_data[:, None, ...]
  init_disp_ds = init_disp[:, None, ...]

  # Process normals in smaller batches
  batch_normal_size = 2
  for i in range(0, batch_size, batch_normal_size):
    end = min(i + batch_normal_size, batch_size)
    pred_normal = compute_normals[0](
        1.0 / torch.clamp(disp_data_ds[i:end], 1e-3, 1e3), K_inv_rescale[None]
    )
    init_normal = compute_normals[0](
        1.0 / torch.clamp(init_disp_ds[i:end], 1e-3, 1e3), K_inv_rescale[None]
    )
    loss_normal += torch.mean(fg_alpha[i:end] * (1.0 - torch.sum(pred_normal * init_normal, dim=1)))

  loss_normal /= (batch_size / batch_normal_size)

  return (
      w_ratio * total_loss_d_ratio
      + w_si * loss_prior
      + w_flow * total_loss_flow
      + w_normal * loss_normal
      + loss_grad * w_grad
  )