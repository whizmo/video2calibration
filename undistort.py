#!/usr/bin/env python
import argparse
import os
from glob import glob

import cv2
import numpy as np
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort images based on camera calibration.')
    parser.add_argument('calibration', help='input video file')
    parser.add_argument('input_mask', help='input mask')
    parser.add_argument('out', help='output directory')
    parser.add_argument('--crop_roi', action='store_true', default=False)
    parser.add_argument('--alpha', default=0.0, type=float)
    args = parser.parse_args()

    with open(args.calibration) as fr:
        c = yaml.safe_load(fr)
        mtx = np.array(c['camera_matrix'])
        dist = np.array(c['dist_coefs'])

    print(f"Camera matrix:\n{mtx}")
    print(f"dist:{dist}")

    if not os.path.exists(args.out):
        print(f"Creating output directory as it does not exist: {args.out}")
        os.makedirs(args.out)

    oldhw = None
    for fn in glob(args.input_mask):
        print(f'processing {fn}...')
        img = cv2.imread(fn)
        if img is None:
            print(f"failed to load {fn}")
            continue

        h,  w = img.shape[:2]
        if oldhw != (h,w):
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=args.alpha, newImgSize=(w, h))
            print(f"WxH: {w}x{h}")
            print(f"New camera matrix:\n{newcameramtx}")
            print(f"ROI:{roi}")
            oldhw = h,w

        img_und = cv2.undistort(img, mtx, dist,
                                newCameraMatrix=newcameramtx)

        if args.crop_roi:
            # crop the image
            x, y, w, h = roi
            img_und = img_und[y:y+h, x:x+w]

        name, ext = os.path.splitext(os.path.basename(fn))
        name += '_und' + ['', '_crop'][args.crop_roi] + ['', f'_{args.alpha}'][args.alpha != 0.0]
        out_fname = os.path.join(args.out, name + ext)
        write_ok = cv2.imwrite(out_fname, img_und)

        if not write_ok:
            print(f"Error writing file: {out_fname}")
            exit(-1)
