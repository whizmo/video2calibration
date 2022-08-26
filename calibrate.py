#!/usr/bin/env python
import argparse
import os
import pickle
from glob import glob

import cv2
import numpy as np
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera using a video of a chessboard or a sequence of images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input video file or glob mask')
    parser.add_argument('out', help='output calibration yaml file')
    parser.add_argument('--debug-dir', help='path to directory where images with detected chessboard will be written',
                        default=None)
    parser.add_argument('-c', '--corners', help='output corners file', default=None)
    parser.add_argument('-fs', '--framestep', help='use every nth frame in the video', default=1, type=int)
    parser.add_argument('-max', '--max-frames', help='limit the number of frames used for calibration', default=None, type=int)
    # parser.add_argument('--figure', help='saved visualization name', default=None)
    parser.add_argument('-ps', '--pattern_size', help='The pattern size', default=[9, 6], type=int, nargs=2)
    parser.add_argument('--save_dets_dir', help='path to directory where images with detected chessboard will be written',
                        default=None)
    parser.add_argument('--term_iters', help='Calibration termination criteria iterations', default=30, type=int)
    parser.add_argument('--term_eps', help='Calibration termination criteria epsilon', default=np.finfo(float).eps, type=float)
    parser.add_argument('--fix_principal_point',
                        help='The principal point is not changed during the global optimization. It stays at the center',
                        default=False, action='store_true')
    parser.add_argument('--zero_tangent_dist',
                        help='Tangential distortion coefficients (p1,p2) are set to zeros and stay zero.',
                        default=False, action='store_true')
    args = parser.parse_args()

    if '*' in args.input:
        source = glob(args.input)
    else:
        source = cv2.VideoCapture(args.input)
    # square_size = float(args.get('--square_size', 1.0))

    pattern_size = tuple(args.pattern_size)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    # pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    frame = -1
    used_frames = 0
    while True:
        frame += 1
        if isinstance(source, list):
            # glob
            if frame >= len(source):
                break
            if frame % args.framestep != 0:
                continue
            img = cv2.imread(source[frame])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            if not retval:
                break
            if frame % args.framestep != 0:
                continue

        print(f'Searching for chessboard in frame {frame}... ', end='')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        fcc_flags = cv2.CALIB_CB_FILTER_QUADS # + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(img, pattern_size, flags=fcc_flags)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            used_frames += 1
            img_points.append(corners.reshape(1, -1, 2))
            obj_points.append(pattern_points.reshape(1, -1, 3))
            print('ok')
            if args.max_frames is not None and used_frames >= args.max_frames:
                print(f'Found {used_frames} frames with the chessboard.')
                break
        else:
            print('not found')

        if args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, f'{frame:04d}{["_NF",""][found]}.png'), img_chess)
        if args.save_dets_dir and found:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(args.save_dets_dir, f'{frame:04d}.png'), img_chess)

    if args.corners:
        with open(args.corners, 'wb') as fw:
            pickle.dump(img_points, fw)
            pickle.dump(obj_points, fw)
            pickle.dump((w, h), fw)

# load corners
#    with open('corners.pkl', 'rb') as fr:
#        img_points = pickle.load(fr)
#        obj_points = pickle.load(fr)
#        w, h = pickle.load(fr)

    print('\ncalibrating...')
    calib_flags = cv2.CALIB_FIX_PRINCIPAL_POINT * args.fix_principal_point
    calib_flags += cv2.CALIB_ZERO_TANGENT_DIST * args.zero_tangent_dist
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.term_iters, args.term_eps)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None,
                                                                       flags=calib_flags, criteria=term_crit)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # # fisheye calibration
    # rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
    #     obj_points, img_points,
    #     (w, h), camera_matrix, np.array([0., 0., 0., 0.]),
    #     None, None,
    #     cv2.fisheye.CALIB_USE_INTRINSIC_GUESS, (3, 1, 1e-6))
    # print "RMS:", rms
    # print "camera matrix:\n", camera_matrix
    # print "distortion coefficients: ", dist_coefs.ravel()

    calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs': dist_coefs.tolist()}
    with open(args.out, 'w') as fw:
        yaml.dump(calibration, fw)
