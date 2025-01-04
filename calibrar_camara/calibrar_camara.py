

from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os
from os.path import join


def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

path = join(os.getcwd(), "calibrar_camara", "fotos_calibracion")

imgs_path = [join(path, f"{img_path}") for img_path in os.listdir(path)]
imgs = load_images(imgs_path)

dim = (7,7)

corners = [cv2.findChessboardCorners(img, dim) for img in imgs]
corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
corners_refined = [cv2.cornerSubPix(i, cor[1], dim, (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

imgs_copy = copy.deepcopy(imgs)
draw_imgs = [cv2.drawChessboardCorners(i, dim, c[1], c[0]) for i, c in zip(imgs_copy, corners)]

def get_chessboard_points(chessboard_shape, dx, dy):
    rows, cols = chessboard_shape  # Number of corners in rows and columns
    # Create a grid of 3D points where z = 0 for all points
    points = np.zeros((rows * cols, 3), np.float32)
    # Fill the X and Y coordinates
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * [dx, dy]
    return points

chessboard_points = get_chessboard_points(dim, 30, 30)

valid_corners = [cor[1] for cor in corners if cor[0]]
valid_corners = np.asarray(valid_corners, dtype=np.float32)

object_points = [chessboard_points for _ in valid_corners]
image_size = imgs[0].shape[1::-1]  # Ancho x Alto

rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, valid_corners, image_size, None, None)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)