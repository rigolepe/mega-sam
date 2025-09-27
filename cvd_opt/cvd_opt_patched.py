#!/usr/bin/env python
"""CVD optimization met patch-based processing voor geheugenbesparing."""

import torch
import torch.nn.functional as F


def consistency_loss_patched(
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
    patch_size=256,  # Process in patches van 256x256
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
    """Consistency loss met patch-based processing voor geheugenbesparing."""
    _, H, W = disp_data.shape

    # Bereken transformatie matrices één keer
    cam_1to2 = torch.bmm(
        torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
        torch.index_select(cam_c2w, dim=0, index=ii),
    )
    rot = cam_1to2[:, :3, :3]
    trans = cam_1to2[:, :3, 3:4]

    total_loss_flow = 0.0
    total_loss_d_ratio = 0.0
    num_patches = 0

    # Process in patches
    for y_start in range(0, H, patch_size):
        y_end = min(y_start + patch_size, H)
        for x_start in range(0, W, patch_size):
            x_end = min(x_start + patch_size, W)
            patch_h = y_end - y_start
            patch_w = x_end - x_start

            # Maak grid voor deze patch
            xx = torch.arange(x_start, x_end).view(1, -1).repeat(patch_h, 1)
            yy = torch.arange(y_start, y_end).view(-1, 1).repeat(1, patch_w)
            xx = xx.view(1, 1, patch_h, patch_w)
            yy = yy.view(1, 1, patch_h, patch_w)
            grid_patch = torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)

            # Extract patch data
            flows_patch = flows[:, :, y_start:y_end, x_start:x_end].permute(0, 2, 3, 1)
            flow_masks_patch = flow_masks[:, :, y_start:y_end, x_start:x_end].permute(0, 2, 3, 1).squeeze(-1)

            # Warp disp from target time
            pixel_locations = grid_patch + flows_patch
            resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
            normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

            disp_sampled = F.grid_sample(
                torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
                normalized_pixel_locations,
                align_corners=True,
            )

            # Get patch of reference disparity
            disp_ref_patch = torch.index_select(disp_data, dim=0, index=ii)[:, y_start:y_end, x_start:x_end]
            ref_depth_patch = 1.0 / torch.clamp(disp_ref_patch, 1e-3, 1e3)

            # 3D points voor patch
            grid_h_patch = torch.cat([grid_patch, torch.ones_like(grid_patch[..., 0:1])], dim=-1).unsqueeze(-1)

            # Bereken 3D punten alleen voor deze patch
            pts_3d_ref_patch = ref_depth_patch[..., None, None] * (K_inv[None, None, None] @ grid_h_patch)

            # Project naar target view - gebruik vooraf berekende rot en trans
            pts_3d_tgt_patch = (rot[:, None, None] @ pts_3d_ref_patch) + trans[:, None, None]

            # Bereken diepte in target view
            depth_tgt_patch = pts_3d_tgt_patch[:, :, :, 2:3, 0]
            disp_tgt_patch = 1.0 / torch.clamp(depth_tgt_patch, 0.1, 1e3)

            # Flow consistency loss voor patch
            pts_2D_tgt_patch = K[None, None, None] @ pts_3d_tgt_patch
            flow_masks_patch_ = flow_masks_patch * (pts_2D_tgt_patch[:, :, :, 2, 0] > 0.1)
            pts_2D_tgt_patch = pts_2D_tgt_patch[:, :, :, :2, 0] / torch.clamp(
                pts_2D_tgt_patch[:, :, :, 2:, 0], 1e-3, 1e3
            )

            # Bereken losses voor deze patch
            disp_sampled_clamped = torch.clamp(disp_sampled, 1e-3, 1e2)
            disp_tgt_clamped = torch.clamp(disp_tgt_patch, 1e-3, 1e2)

            ratio = torch.maximum(
                disp_sampled_clamped / disp_tgt_clamped,
                disp_tgt_clamped / disp_sampled_clamped,
            )

            # Voeg patch losses toe aan totaal
            if flow_masks_patch_.sum() > 0:
                patch_loss_flow = (pts_2D_tgt_patch - pixel_locations).abs().mean()
                patch_loss_d_ratio = (ratio - 1.0).mean()

                total_loss_flow += patch_loss_flow * flow_masks_patch_.sum()
                total_loss_d_ratio += patch_loss_d_ratio * flow_masks_patch_.sum()
                num_patches += flow_masks_patch_.sum()

            # Clear intermediate tensors om geheugen vrij te maken
            del pts_3d_ref_patch, pts_3d_tgt_patch, grid_h_patch
            torch.cuda.empty_cache()

    # Bereken gemiddelde loss
    if num_patches > 0:
        total_loss_flow /= num_patches
        total_loss_d_ratio /= num_patches

    # Combineer losses (simplified versie)
    total_loss = w_flow * total_loss_flow + w_ratio * total_loss_d_ratio

    return total_loss