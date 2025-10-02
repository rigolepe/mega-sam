import argparse
import cv2
import glob
import numpy as np
import os
import torch
from pathlib import Path

from PIL import Image
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))

from moge.model.v2 import MoGeModel

#file_path = Path(__file__).parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoGe')
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--fov_x', type=str, default='')
    
    args = parser.parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load model and preprocessing transform
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').to(DEVICE).eval()
    
    # video_files = sorted(glob.glob(os.path.join(args.img_path, '*.mp4')))
    # Zoek naar zowel .png als .jpg bestanden
    image_list = sorted(glob.glob(os.path.join("%s" % (args.img_path), "*.png")))
    if not image_list:
        image_list = sorted(glob.glob(os.path.join("%s" % (args.img_path), "*.jpg")))

    os.makedirs(args.outdir, exist_ok=True)

    for k, filename in enumerate(image_list):
        print(f'Progress {k+1}/{len(image_list)}: {filename}')
        
        image = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)


        # TODO: batch infer multiple
        if args.fov_x:
            output = model.infer(image_tensor, fov_x=torch.tensor(float(args.fov_x), device=DEVICE), force_projection=False, apply_mask=False)
        else:
            output = model.infer(image_tensor, force_projection=False, apply_mask=False)

        #     return_dict = {
        #     'points': points,
        #     'intrinsics': intrinsics,
        #     'depth': depth,
        #     'mask': mask_binary,
        #     'normal': normal
        # }

        mask = output['mask'].clone()
        mask[~torch.isfinite(output['depth'])] = False

        depth = output['depth'].clone()
        depth[~mask] = 0

        fov = np.rad2deg(2 * np.arctan(1 / (2 * output["intrinsics"][0, 0].cpu().numpy())))

        # print(fov)
        
        np.savez(
            os.path.join(args.outdir, filename.split("/")[-1][:-4] + ".npz"),
            depth=depth.squeeze(0).cpu().numpy().astype(np.float32),
            fov=fov,
        )