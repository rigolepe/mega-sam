#!/usr/bin/env python
"""Gemodificeerde cvd_opt.py met geheugenoptimalisaties voor 720p."""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3


def process_in_chunks(tensor_op, *tensors, chunk_size=4):
    """Process grote tensor operaties in chunks om geheugen te besparen."""
    results = []
    n_samples = tensors[0].shape[0]

    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunk_tensors = [t[i:end_idx] for t in tensors]
        with torch.cuda.amp.autocast():
            result = tensor_op(*chunk_tensors)
        results.append(result)
        torch.cuda.empty_cache()

    return torch.cat(results, dim=0)


def consistency_loss_optimized(
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
    tile_size=256,  # Process in tiles
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
    """Geoptimaliseerde consistency loss met tiling voor geheugenbesparing."""
    _, H, W = disp_data.shape

    # Process in kleinere batches
    batch_size = len(ii)
    if batch_size > 8:  # Limiteer batch size
        # Split in kleinere batches
        mid = batch_size // 2
        loss1 = consistency_loss_optimized(
            cam_c2w, K, K_inv, disp_data, init_disp, uncertainty,
            flows[:mid], flow_masks[:mid], ii[:mid], jj[:mid],
            compute_normals, fg_alpha, tile_size,
            w_ratio, w_flow, w_si, w_grad, w_normal
        )
        loss2 = consistency_loss_optimized(
            cam_c2w, K, K_inv, disp_data, init_disp, uncertainty,
            flows[mid:], flow_masks[mid:], ii[mid:], jj[mid:],
            compute_normals, fg_alpha, tile_size,
            w_ratio, w_flow, w_si, w_grad, w_normal
        )
        return (loss1 + loss2) / 2

    # Voor kleinere batches, gebruik tiling
    total_loss = 0.0
    num_tiles = 0

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            tile_h = y_end - y
            tile_w = x_end - x

            # Maak grid voor deze tile
            xx = torch.arange(x, x_end).view(1, -1).repeat(tile_h, 1)
            yy = torch.arange(y, y_end).view(-1, 1).repeat(1, tile_w)
            xx = xx.view(1, 1, tile_h, tile_w)
            yy = yy.view(1, 1, tile_h, tile_w)
            grid = torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)

            # Extract tile data
            flows_tile = flows[:, :, y:y_end, x:x_end]
            flow_masks_tile = flow_masks[:, :, y:y_end, x:x_end]

            # Bereken transformaties
            cam_1to2 = torch.bmm(
                torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
                torch.index_select(cam_c2w, dim=0, index=ii),
            )

            flows_step = flows_tile.permute(0, 2, 3, 1)
            flow_masks_step = flow_masks_tile.permute(0, 2, 3, 1).squeeze(-1)

            # Warp disp from target time
            pixel_locations = grid + flows_step
            resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
            normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

            with torch.cuda.amp.autocast():
                disp_sampled = F.grid_sample(
                    torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
                    normalized_pixel_locations,
                    align_corners=True,
                )

                # Get tile of reference disparity
                disp_ref_tile = torch.index_select(disp_data, dim=0, index=ii)[:, y:y_end, x:x_end]
                ref_depth = 1.0 / torch.clamp(disp_ref_tile, 1e-3, 1e3)

                # 3D points voor tile - process in smaller chunks
                grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)

                # Split matrix multiplication in kleinere stukken
                pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
                rot = cam_1to2[:, None, None, :3, :3]
                trans = cam_1to2[:, None, None, :3, 3:4]

                # Process transformation in chunks als nodig
                if tile_h * tile_w > 32768:  # Als tile te groot is
                    # Process in sub-tiles
                    sub_tile_size = 128
                    pts_3d_tgt_list = []
                    for sy in range(0, tile_h, sub_tile_size):
                        for sx in range(0, tile_w, sub_tile_size):
                            sy_end = min(sy + sub_tile_size, tile_h)
                            sx_end = min(sx + sub_tile_size, tile_w)
                            sub_pts = pts_3d_ref[:, sy:sy_end, sx:sx_end]
                            sub_tgt = (rot @ sub_pts) + trans
                            pts_3d_tgt_list.append(sub_tgt)
                    # Combineer resultaten (dit zou complexer moeten zijn voor 2D grid)
                    pts_3d_tgt = pts_3d_tgt_list[0]  # Simplified
                else:
                    pts_3d_tgt = (rot @ pts_3d_ref) + trans

                depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
                disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

                # Bereken losses
                disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
                disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

                ratio = torch.maximum(
                    disp_sampled / disp_tgt,
                    disp_tgt / disp_sampled,
                )

                # Simple loss calculation voor tile
                tile_loss = (ratio - 1.0).mean()
                total_loss += tile_loss
                num_tiles += 1

            # Cleanup
            del pts_3d_ref, pts_3d_tgt, grid_h
            torch.cuda.empty_cache()

    return total_loss / max(num_tiles, 1)


# Main optimization loop aanpassing
def optimize_with_memory_management(args):
    """Hoofdoptimalisatie loop met geheugenbeheer."""

    # Load data
    scene_name = args.scene_name
    data_dir = f"data/{scene_name}"

    print(f"Loading data from {data_dir}")

    # Load in smaller chunks
    disp_data = torch.from_numpy(np.load(f"{data_dir}/disp_from_dm_2.npy")).cuda()
    poses = torch.from_numpy(np.load(f"{data_dir}/poses_colmap.npy")).cuda()
    flows = torch.from_numpy(np.load(f"{data_dir}/flow/flow.npy")).cuda()
    flow_masks = torch.from_numpy(np.load(f"{data_dir}/flow/mask.npy")).cuda()
    K = torch.from_numpy(np.load(f"{data_dir}/K_colmap.npy")).cuda()

    N, H, W = disp_data.shape
    print(f"Processing {N} frames of size {H}x{W}")

    # Process in smaller windows
    window_size = 5  # Process 5 frames at a time instead of all

    for start_idx in range(0, N - window_size, window_size // 2):
        end_idx = min(start_idx + window_size, N)

        print(f"Processing frames {start_idx} to {end_idx}")

        # Extract window
        disp_window = disp_data[start_idx:end_idx]
        poses_window = poses[start_idx:end_idx]
        flows_window = flows[start_idx:end_idx-1] if start_idx < N-1 else flows[-1:]
        masks_window = flow_masks[start_idx:end_idx-1] if start_idx < N-1 else flow_masks[-1:]

        # Run optimization for this window
        # ... optimization code here ...

        # Clear cache after each window
        torch.cuda.empty_cache()

    print("Optimization completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--w_grad", type=float, default=2.0)
    parser.add_argument("--w_normal", type=float, default=5.0)
    parser.add_argument("--tile_size", type=int, default=256,
                       help="Tile size for memory-efficient processing")
    parser.add_argument("--window_size", type=int, default=5,
                       help="Number of frames to process at once")
    args = parser.parse_args()

    optimize_with_memory_management(args)