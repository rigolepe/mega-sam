# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Consistent video depth optimization - Geheugen geoptimaliseerde versie."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=redefined-outer-name

import argparse
import os
from pathlib import Path

from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3
import numpy as np
import torch


def gradient_loss(gt, pred, u):
  """Gradient loss."""
  del u
  diff = pred - gt
  v_gradient = torch.abs(
      diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1]
  )  # * mask_v
  h_gradient = torch.abs(
      diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:]
  )  # * mask_h

  pred_grad = torch.abs(
      pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])
  ) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
  gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + torch.abs(
      gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:]
  )

  grad_diff = torch.abs(pred_grad - gt_grad)
  nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
  # weight = (1. - torch.exp(-(grad_diff * 5.)).detach())
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


ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5  # Minimum resolutie voor kwaliteit


def plan_overlapping_batches(total_flows, batch_size=100, overlap_ratio=0.25):
  """Plan batches met overlap voor smooth transitions.

  Args:
    total_flows: Totaal aantal flows om te verwerken
    batch_size: Grootte van elke batch
    overlap_ratio: Fractie overlap tussen batches (0.25 = 25%)

  Returns:
    List van tuples (start_idx, end_idx) voor elke batch
  """
  overlap = int(batch_size * overlap_ratio)
  stride = batch_size - overlap  # 75 bij 25% overlap

  batches = []
  for start in range(0, total_flows, stride):
    end = min(start + batch_size, total_flows)
    batches.append((start, end))
    if end >= total_flows:
      break

  return batches


def process_flow_batch(flows_batch, flow_masks_batch, ii, jj, cam_c2w, K, K_inv,
                       disp_data, uncertainty, grid, H, W, weights):
  """Process een batch van flows met gewichten.

  Returns:
    tuple: (loss_flow, loss_d_ratio, total_weight)
  """
  batch_size = len(ii)

  # Permute flows voor processing
  flows_step = flows_batch.permute(0, 2, 3, 1)
  flow_masks_step = flow_masks_batch.permute(0, 2, 3, 1).squeeze(-1)

  # Bereken camera transformaties
  cam_1to2 = torch.bmm(
      torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
      torch.index_select(cam_c2w, dim=0, index=ii),
  )

  # Warp disp from target time
  pixel_locations = grid + flows_step
  resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
  normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

  disp_sampled = torch.nn.functional.grid_sample(
      torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
      normalized_pixel_locations,
      align_corners=True,
  )

  uu = torch.index_select(uncertainty, dim=0, index=ii).squeeze(1)

  # Depth of reference view
  ref_depth = 1.0 / torch.clamp(
      torch.index_select(disp_data, dim=0, index=ii), 1e-3, 1e3
  )

  grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)

  # 3D transformatie met frame-by-frame processing
  rot = cam_1to2[:, None, None, :3, :3]
  trans = cam_1to2[:, None, None, :3, 3:4]

  pts_3d_tgt_list = []
  for i in range(batch_size):
    with torch.cuda.amp.autocast():
      ref_depth_i = ref_depth[i:i+1]
      rot_i = rot[i:i+1]
      trans_i = trans[i:i+1]

      pts_3d_ref_i = ref_depth_i[..., None, None] * (K_inv[None, None, None] @ grid_h)
      pts_3d_tgt_i = (rot_i @ pts_3d_ref_i) + trans_i

      pts_3d_tgt_list.append(pts_3d_tgt_i.float())
      del pts_3d_ref_i

  pts_3d_tgt = torch.cat(pts_3d_tgt_list, dim=0)
  del pts_3d_tgt_list

  depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
  disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

  # Flow consistency loss
  pts_2D_tgt = K[None, None, None] @ pts_3d_tgt
  flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
  pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(
      pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3
  )

  disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
  disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

  # Bereken losses met gewichten
  ratio = torch.maximum(
      disp_sampled.squeeze() / disp_tgt.squeeze(),
      disp_tgt.squeeze() / disp_sampled.squeeze(),
  )
  ratio_error = torch.abs(ratio - 1.0)

  # Apply weights voor overlap regions
  weighted_masks = flow_masks_step_ * weights[:, None, None]

  loss_d_ratio = torch.sum(
      (ratio_error * uu + ALPHA_MOTION * torch.log(1.0 / uu)) * weighted_masks
  )

  flow_error = torch.abs(pts_2D_tgt - pixel_locations)
  loss_flow = torch.sum(
      (flow_error * uu[..., None] + ALPHA_MOTION * torch.log(1.0 / uu[..., None]))
      * weighted_masks[..., None]
  )

  total_weight = torch.sum(weighted_masks)

  # Cleanup
  del pts_3d_tgt, grid_h
  torch.cuda.empty_cache()

  return loss_flow, loss_d_ratio, total_weight


def consistency_loss(
    cam_c2w,
    K,
    K_inv,
    disp_data,
    init_disp,
    uncertainty,
    flows,  # Dit is nu flows_cpu!
    flow_masks,  # Dit is nu flow_masks_cpu!
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
  """Consistency loss - met overlappende batches voor alle flows."""
  _, H, W = disp_data.shape

  # Check of flows op CPU zijn
  flows_on_cpu = not flows.is_cuda

  # mesh grid
  xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
  yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
  xx = xx.view(1, 1, H, W)
  yy = yy.view(1, 1, H, W)
  grid = (
      torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)
  )

  # Gebruik alleen step=1 flows (eerste 970)
  batch_size = len(ii)
  print(f"Total ii/jj pairs: {batch_size}")

  if batch_size > 1000:
    print(f"Using only step=1 flows (first 970 pairs)")
    actual_batch_size = min(970, batch_size)
    ii = ii[:actual_batch_size]
    jj = jj[:actual_batch_size]
    batch_size = actual_batch_size

  # Plan overlappende batches
  MAX_CHUNK = 100
  OVERLAP_RATIO = 0.25

  if batch_size > MAX_CHUNK and flows_on_cpu:
    batches = plan_overlapping_batches(batch_size, MAX_CHUNK, OVERLAP_RATIO)
    print(f"Processing {batch_size} flows in {len(batches)} overlapping batches")

    # Accumulators voor gewogen losses
    total_loss_flow = 0.0
    total_loss_d_ratio = 0.0
    total_weight = 0.0

    for batch_idx, (batch_start, batch_end) in enumerate(batches):
      batch_ii = ii[batch_start:batch_end]
      batch_jj = jj[batch_start:batch_end]
      batch_size_local = batch_end - batch_start

      # Weight scheduling voor overlap regions
      weights = torch.ones(batch_size_local).cuda()
      if batch_idx > 0 and batch_start < batches[batch_idx-1][1]:
        # Dit is een overlap region met de vorige batch
        overlap_size = batches[batch_idx-1][1] - batch_start
        weights[:overlap_size] *= 0.5  # Verminder gewicht voor overlap

      # Laad flows voor deze batch van CPU
      flows_batch = flows[batch_ii.cpu()].cuda()
      flow_masks_batch = flow_masks[batch_ii.cpu()].cuda()

      # Process deze batch
      loss_flow_batch, loss_d_ratio_batch, batch_weight = process_flow_batch(
          flows_batch, flow_masks_batch, batch_ii, batch_jj,
          cam_c2w, K, K_inv, disp_data, uncertainty,
          grid, H, W, weights
      )

      # Accumuleer gewogen losses
      total_loss_flow += loss_flow_batch
      total_loss_d_ratio += loss_d_ratio_batch
      total_weight += batch_weight

      # Cleanup batch tensors
      del flows_batch, flow_masks_batch
      torch.cuda.empty_cache()

      print(f"  Batch {batch_idx+1}/{len(batches)}: frames {batch_ii[0]}-{batch_ii[-1]}")

    # Normaliseer losses
    loss_flow = total_loss_flow / (total_weight * 2.0 + 1e-8)
    loss_d_ratio = total_loss_d_ratio / (total_weight + 1e-8)

  else:
    # Enkele batch processing (voor kleine datasets of GPU flows)
    if flows_on_cpu:
      flows_batch = flows[ii.cpu()].cuda()
      flow_masks_batch = flow_masks[ii.cpu()].cuda()
    else:
      flows_batch = flows
      flow_masks_batch = flow_masks

    weights = torch.ones(len(ii)).cuda()
    loss_flow, loss_d_ratio, _ = process_flow_batch(
        flows_batch, flow_masks_batch, ii, jj,
        cam_c2w, K, K_inv, disp_data, uncertainty,
        grid, H, W, weights
    )

    if flows_on_cpu:
      del flows_batch, flow_masks_batch
      torch.cuda.empty_cache()

  # prior mono-depth reg loss
  loss_prior = si_loss(init_disp, disp_data)
  KK = torch.inverse(K_inv)

  # multi gradient consistency
  disp_data_ds = disp_data[:, None, ...]
  init_disp_ds = init_disp[:, None, ...]
  K_rescale = KK.clone()
  K_inv_rescale = torch.inverse(K_rescale)
  pred_normal = compute_normals[0](
      1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None]
  )
  init_normal = compute_normals[0](
      1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None]
  )

  loss_normal = torch.mean(
      fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1))
  )  # / (1e-8 + torch.sum(fg_alpha))

  loss_grad = 0.0
  for scale in range(4):
    interval = 2**scale
    disp_data_ds = torch.nn.functional.interpolate(
        disp_data[:, None, ...],
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    init_disp_ds = torch.nn.functional.interpolate(
        init_disp[:, None, ...],
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    uncertainty_rs = torch.nn.functional.interpolate(
        uncertainty,
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    loss_grad += gradient_loss(
        torch.log(disp_data_ds), torch.log(init_disp_ds), uncertainty_rs
    )

  return (
      w_ratio * loss_d_ratio
      + w_si * loss_prior
      + w_flow * loss_flow
      + w_normal * loss_normal
      + loss_grad * w_grad
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
  parser.add_argument("--w_normal", type=float, default=6.0, help="w_normal")
  parser.add_argument(
      "--output_dir", type=str, default="outputs_cvd", help="outputs direcotry"
  )
  parser.add_argument("--scene_name", type=str, help="scene name")

  args = parser.parse_args()

  cache_dir = "./cache_flow"
  rootdir = os.getcwd() + "/reconstructions"

  output_dir = args.output_dir
  scene_name = args.scene_name
  print("***************************** ", scene_name)
  img_data = np.load(os.path.join(rootdir, scene_name, "images.npy"))[
      :, ::-1, ...
  ]
  disp_data = (
      np.load(
          os.path.join(rootdir, scene_name.replace("_opt", ""), "disps.npy")
      )
      + 1e-6
  )
  intrinsics = np.load(os.path.join(rootdir, scene_name, "intrinsics.npy"))
  poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
  mot_prob = np.load(os.path.join(rootdir, scene_name, "motion_prob.npy"))

  flows = np.load(
      "%s/%s/flows.npy" % (cache_dir, scene_name), allow_pickle=True
  )
  flow_masks = np.load(
      "%s/%s/flows_masks.npy" % (cache_dir, scene_name), allow_pickle=True
  )
  flow_masks = np.float32(flow_masks)
  iijj = np.load("%s/%s/ii-jj.npy" % (cache_dir, scene_name), allow_pickle=True)

  intrinsics = intrinsics[0]
  poses_th = torch.as_tensor(poses, device="cpu").float().cuda()

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]

  img_data_pt = (
      torch.from_numpy(np.ascontiguousarray(img_data)).float().cuda() / 255.0
  )

  # BELANGRIJK: Houd flows op CPU om geheugen te besparen!
  print(f"Keeping flows on CPU to save memory...")
  flows_cpu = torch.from_numpy(np.ascontiguousarray(flows)).float()  # CPU!
  flow_masks_cpu = torch.from_numpy(np.ascontiguousarray(flow_masks)).float()  # CPU!

  # Deze kunnen wel naar GPU (klein)
  iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().cuda()
  ii = iijj[0, ...].long()
  jj = iijj[1, ...].long()
  K = torch.from_numpy(K).float().cuda()

  init_disp = torch.from_numpy(disp_data).float().cuda()
  disp_data = torch.from_numpy(disp_data).float().cuda()

  assert init_disp.shape == disp_data.shape

  init_disp = torch.nn.functional.interpolate(
      init_disp.unsqueeze(1),
      scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
      mode="bilinear",
  ).squeeze(1)
  disp_data = torch.nn.functional.interpolate(
      disp_data.unsqueeze(1),
      scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
      mode="bilinear",
  ).squeeze(1)

  # Debug: print shapes voor resize
  print(f"Original disp_data shape after resize: {disp_data.shape}")
  print(f"Original flow shape (CPU): {flows_cpu.shape}")

  # BELANGRIJK: Resize flows op CPU naar EXACT dezelfde dimensie als disp_data
  target_H, target_W = disp_data.shape[-2:]
  original_flow_H, original_flow_W = flows_cpu.shape[-2:]

  # Resize flows op CPU
  print(f"Resizing flows on CPU to save GPU memory...")
  flows_cpu = torch.nn.functional.interpolate(
      flows_cpu,
      size=(target_H, target_W),
      mode="bilinear",
      align_corners=True
  )

  # Schaal de flow waarden proportioneel aan de resize
  flows_cpu[:, 0, :, :] *= (target_W / original_flow_W)  # Schaal x-component
  flows_cpu[:, 1, :, :] *= (target_H / original_flow_H)  # Schaal y-component

  print(f"Resized flow shape (CPU): {flows_cpu.shape}")

  flow_masks_cpu = torch.nn.functional.interpolate(
      flow_masks_cpu,
      size=(target_H, target_W),
      mode="nearest"
  )
  print(f"Resized flow_masks shape (CPU): {flow_masks_cpu.shape}")
  print(f"Target disp_data shape: {disp_data.shape}")

  fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
  fg_alpha = fg_alpha.squeeze(1).float() + 0.2

  cvd_prob = torch.nn.functional.interpolate(
      torch.from_numpy(mot_prob).unsqueeze(1).cuda(),
      scale_factor=(4, 4),
      mode="bilinear",
  )
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

  optim = torch.optim.Adam([
      {"params": log_scale_, "lr": 1e-2},
      {"params": shift_, "lr": 1e-2},
      {"params": uncertainty, "lr": 1e-2},
  ])

  compute_normals = []
  compute_normals.append(
      NormalGenerator(disp_data.shape[-2], disp_data.shape[-1])
  )
  init_disp = torch.clamp(init_disp, 1e-3, 1e3)

  for i in range(100):
    optim.zero_grad()
    cam_c2w = SE3(poses_th).inv().matrix()
    scale_ = torch.exp(log_scale_)

    loss = consistency_loss(
        cam_c2w,
        K,
        K_inv,
        torch.clamp(
            disp_data * scale_[..., None, None] + shift_[..., None, None],
            1e-3,
            1e3,
        ),
        init_disp,
        torch.clamp(uncertainty, 1e-4, 1e3),
        flows_cpu,  # Pass CPU flows!
        flow_masks_cpu,  # Pass CPU flow_masks!
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

    # Vrij geheugen na elke 10 iteraties
    if i % 10 == 0:
      torch.cuda.empty_cache()

  # Then optimize depth and uncertainty
  disp_data = (
      disp_data * torch.exp(log_scale_)[..., None, None].detach()
      + shift_[..., None, None].detach()
  )
  init_disp = (
      init_disp * torch.exp(log_scale_)[..., None, None].detach()
      + shift_[..., None, None].detach()
  )
  init_disp = torch.clamp(init_disp, 1e-3, 1e3)

  disp_data.requires_grad = True
  uncertainty.requires_grad = True
  poses_th.requires_grad = False  # True

  optim = torch.optim.Adam([
      {"params": disp_data, "lr": 5e-3},
      {"params": uncertainty, "lr": 5e-3},
  ])

  losses = []
  for i in range(400):
    optim.zero_grad()
    cam_c2w = SE3(poses_th).inv().matrix()
    loss = consistency_loss(
        cam_c2w,
        K,
        K_inv,
        torch.clamp(disp_data, 1e-3, 1e3),
        init_disp,
        torch.clamp(uncertainty, 1e-4, 1e3),
        flows_cpu,  # Pass CPU flows!
        flow_masks_cpu,  # Pass CPU flow_masks!
        ii,
        jj,
        compute_normals,
        fg_alpha,
        w_ratio=1.0,
        w_flow=0.2,
        w_si=1,
        w_grad=args.w_grad,
        w_normal=args.w_normal,
    )

    loss.backward()
    disp_data.grad = torch.nan_to_num(disp_data.grad, nan=0.0)
    uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)

    optim.step()
    print("step ", i, loss.item())
    losses.append(loss)

    # Vrij geheugen na elke 20 iteraties
    if i % 20 == 0:
      torch.cuda.empty_cache()

  disp_data_opt = (
      torch.nn.functional.interpolate(
          disp_data.unsqueeze(1), scale_factor=(2, 2), mode="bilinear"
      )
      .squeeze(1)
      .detach()
      .cpu()
      .numpy()
  )

  # poses_ = poses_th.detach().cpu().numpy()

  Path(output_dir).mkdir(parents=True, exist_ok=True)
  # Bereken depths en clip eerst, dan pas converteren naar float16
  depths = 1.0 / np.clip(disp_data_opt, 0.01, 1000.0)  # Clip disp eerst
  depths = np.clip(depths, 1e-3, 1e2)  # Clip depths
  depths_float16 = np.float16(depths)  # Converteer naar float16

  np.savez(
      "%s/%s_sgd_cvd_hr.npz" % (output_dir, scene_name),
      images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
      depths=depths_float16,
      intrinsic=K_o.detach().cpu().numpy(),
      cam_c2w=cam_c2w.detach().cpu().numpy(),
  )
  print(f"Saved results to {output_dir}/{scene_name}_sgd_cvd_hr.npz")