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

"""Test camera tracking on a single scene."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=redefined-outer-name
# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import sys

sys.path.append("base/droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from lietorch import SE3

import torch.nn.functional as F
from droid import Droid

# import rerun as rr


def image_stream(
    image_list,
    metric_depth_list,
    scene_name,
    K=None,
    stride=1,
):
  """image generator."""
  del scene_name, stride

  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )  # np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()

  for t, (image_file) in enumerate(image_list):
    image = cv2.imread(image_file)

    depth = np.float32(metric_depth_list[t])
    # depth[depth < 1e-2] = 0.0

    # breakpoint()
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]

    image = torch.as_tensor(image).permute(2, 0, 1)

    depth = torch.as_tensor(depth)
    depth = F.interpolate(
        depth[None, None], (h1, w1), mode="nearest-exact"
    ).squeeze()
    depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

    # invalid depths (e.g. sky) are mapped to 0
    # mask = torch.where(depth > 1e-8, 1., 0.) # torch.ones_like(depth)
    mask = torch.ones_like(depth)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= w1 / w0
    intrinsics[1::2] *= h1 / h0

    yield t, image[None], depth, intrinsics, mask


def save_full_reconstruction(
    droid, full_traj, rgb_list, senor_depth_list, motion_prob, scene_name
):
  """Save full reconstruction."""
  from pathlib import Path
  t = full_traj.shape[0]
  images = np.array(rgb_list[:t])  # droid.video.images[:t].cpu().numpy()

  # disps = 1.0 / (np.array(senor_depth_list[:t]) + 1e-8)
  sensor_depth_array = np.array(senor_depth_list[:t])
  valid_mask = sensor_depth_array > 0
  disps = np.zeros_like(sensor_depth_array)
  # disps[~valid_mask] = 1e-8
  disps[valid_mask] = 1.0 / sensor_depth_array[valid_mask]

  poses = full_traj  # .cpu().numpy()
  intrinsics = droid.video.intrinsics[:t].cpu().numpy()

  Path("reconstructions/{}".format(scene_name)).mkdir(
      parents=True, exist_ok=True
  )
  np.save("reconstructions/{}/images.npy".format(scene_name), images)
  np.save("reconstructions/{}/disps.npy".format(scene_name), disps)
  np.save("reconstructions/{}/poses.npy".format(scene_name), poses)
  np.save("reconstructions/{}/intrinsics.npy".format(scene_name), intrinsics * 8.0)
  np.save("reconstructions/{}/motion_prob.npy".format(scene_name), motion_prob)

  intrinsics = intrinsics[0] * 8.0
  poses_th = torch.as_tensor(poses, device="cpu")
  cam_c2w = SE3(poses_th).inv().matrix().numpy()

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]
#   print("K ", K)
#   print("img_data ", images.shape)
#   print("disp_data ", disps.shape)

  #max_frames = min(1000, images.shape[0])
  print("outputs/%s_droid.npz" % scene_name)
  Path("outputs").mkdir(parents=True, exist_ok=True)

  np.savez(
      "outputs/%s_droid.npz" % scene_name,
      images=np.uint8(images[:, ::-1, ...].transpose(0, 2, 3, 1)),
      depths=np.float32(np.array(senor_depth_list[:t])),
      intrinsic=K,
      cam_c2w=cam_c2w[:],
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datapath")
  parser.add_argument("--weights", default="droid.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--disable_vis", action="store_true")

  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument(
      "--filter_thresh", type=float, default=2.0
  )  # motion threhold for keyframe
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--scene_name", help="scene_name")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)

  parser.add_argument("--depth_path", default="")
  #parser.add_argument("--opt_focal", action="")
  args = parser.parse_args()

  print("Running evaluation on {}".format(args.datapath))
  print(args)

  scene_name = args.scene_name.split("/")[-1]

  # Zoek naar zowel .png als .jpg bestanden
  image_list = sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.png")))
  if not image_list:
    image_list = sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.jpg")))
  depth_paths = sorted(glob.glob(os.path.join("%s" % (args.depth_path), "*.npz")))

  img_0 = cv2.imread(image_list[0])
  img_count = len(image_list)
  metric_depth_list = []
  fovs = []
  tstamps = []
  
  for depth_file in depth_paths:
    depth_data = np.load(depth_file)
    metric_depth_list.append(depth_data["depth"])
    fovs.append(depth_data["fov"])

  print("************** MOGE FOV ", np.median(fovs))
  ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
  K = np.eye(3)
  K[0, 0] = (
      ff * 1.0
  )  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
  K[1, 1] = (
      ff * 1.0
  )  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
  K[0, 2] = (
      img_0.shape[1] / 2.0
  )  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
  K[1, 2] = (
      img_0.shape[0] / 2.0
  )  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))

#   rr.init("rerun_example_my_data")
#   rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

  rgb_list = []
  senor_depth_list = []
  for t, image, depth, intrinsics, mask in tqdm(
      image_stream(
          image_list,
          metric_depth_list,
          scene_name,
          K=K,
      )
  ):
    # rr.set_time("frame_nr", sequence=t)
    # rr.log(f"img", rr.Image(image[0].permute(1,2,0) / 255.0))
    # rr.log(f"mask", rr.Image(mask))
    # rr.log(f"depth", rr.DepthImage(depth))

    rgb_list.append(image[0])
    senor_depth_list.append(depth)
    if t == 0:
      args.image_size = [image.shape[2], image.shape[3]]
      droid = Droid(args)
    elif t == img_count - 1:
      break # prevent last frame from being tracked twice

    droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

  # last frame
  droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

  traj_est, depth_est, motion_prob = droid.terminate(
      image_stream(
          image_list,
          metric_depth_list,
          scene_name,
          K=K,
      ),
      _opt_intr=True,
      full_ba=True,
      scene_name=scene_name,
  )

  if args.scene_name is not None:
    save_full_reconstruction(
        droid,
        traj_est,
        rgb_list,
        senor_depth_list,
        motion_prob,
        args.scene_name,
    )

    save_full_reconstruction(
        droid,
        traj_est,
        rgb_list,
        depth_est,
        motion_prob,
        args.scene_name + "_estimated",
    )
