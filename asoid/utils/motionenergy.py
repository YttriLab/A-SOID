
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import matplotlib.pyplot as plt
import configparser as cfg

import joblib
import os
import glob
from tqdm import tqdm


""" Egocentric alignment"""
def set_to_origin(mouse, ref_idx2: list):
    # number of timesteps and coordinates
    nts,ncoords = mouse.shape
    # move tail root to origin
    #mousenorm = mouse
    #all x - x_tail; y- y_tail
    #mousenorm = mouse - np.tile(mouse[:,ref_idx2],(1,(int(ncoords / 2))))
    #the array is build like this: bp1_x,bp1_y, bp2_x, bp_y etc.
    #create an array that is ref_bp2 x, y for all bp
    ref_coords = mouse[:,ref_idx2]
    norm_array = np.tile(ref_coords, (1,int(ncoords/2)))
    mousenorm = mouse - norm_array

    return mousenorm

def shift_from_origin(arr, x_shift, y_shift, inplace = False):
    """Shifts all points by (x_shift, y_shift)"""
    if not inplace:
        shift_arr = arr.copy()
        # Even rows, odd columns -> all y
        shift_arr[:, 1::2] += y_shift
        # Odd rows, even columns -> all x
        shift_arr[:, ::2] += x_shift
        return shift_arr
    else:
        # Even rows, odd columns
        arr[:, 1::2] += x_shift
        # Odd rows, even columns
        arr[:, ::2] += y_shift

def magic_transformer(row, ref_point_index):
    pointsT = np.zeros((row.size//2, 3))
    pointsT[:, :2] = row.reshape(row.size//2, 2)
    ref_v = np.zeros((1, 3))
    ref_v[:, 0] = 1
    r, _ = Rotation.align_vectors(ref_v, pointsT[ref_point_index:ref_point_index+1, :])
    return r.apply(pointsT)[:, :2].flatten()


def conv_2_egocentric(arr,  ref_rot_idxs: list, ref_origin_idx: list):
    """
    Calculates egocentric coordinates for mouse and return array
    :param arr: numpy array with bodypart coords X and Y
    :param ref_rot_idxs: reference bodypart index [idx_x, idx_y] that will be used to calculate rotation matrix for; Results in bp on x-axis
    :param ref_origin_idx: reference bodypart that will be new origin (0,0)
    :return: egocentric array
    """

    mouse_data = arr.copy()
    # set one bodypart to new origin mouse
    mousenorm = set_to_origin(mouse_data,ref_origin_idx)
    #rotate to y-axis
    #convert to bp idx
    ref_idx_bp = ref_rot_idxs[0]//2
    rot_mousenorm = np.apply_along_axis(magic_transformer, 1, mousenorm, ref_idx_bp)

    return rot_mousenorm

"""Collecting labels"""

def collect_labels(targets, label_number):
    collection_list = []
    total_labels = 0
    for f in targets:
        #find idx of labels
        l_list = np.argwhere(f == label_number).ravel()
        total_labels += len(l_list)
        collection_list.append(l_list)
    return collection_list , total_labels

"""All animation functions"""

def get_outline(order_list, keypoint_idx_dict):
    order_idx = np.array([np.array(keypoint_idx_dict[x]) for x in order_list])
    return order_idx.flatten()

def get_outline_array(data, bp_idx):

    return data[:,bp_idx]

def animate_blobs(arr, filename, outlines:dict, include_dots = False, center_shift = True, show = False, framerate = 30, resolution = (1500, 1500)):

    #center shift
    if center_shift:
        arr_shifted = shift_from_origin(arr, resolution[0]/2, resolution[1]/2)
    else:
        arr_shifted = arr

    polygons =  []
    for p_outline in outlines.keys():
        outlines[p_outline]["polygon"] = get_outline_array(arr_shifted, outlines[p_outline]["idx"])

    if show:
        scale_percent = 30 # percent of original size
        width = int(resolution[0] * scale_percent / 100)
        height = int(resolution[1] * scale_percent / 100)
        dim = (width, height)


    White = (255, 255, 255)

    #create videowriter
    #set video parameters
    codec = cv2.VideoWriter_fourcc(
                *"XVID"
            )  # codec in which we output the videofiles

    #out = cv2.VideoWriter(ouput_path, codec, framerate, resolution)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), framerate, resolution)

    for idx in np.arange(arr.shape[0]):
        #white background
    #     img = np.ones((resolution[0],resolution[1],3), np.uint8)
    #     img = img * 255
        #black background
        img = np.zeros((resolution[0],resolution[1],3), np.uint8)

        for poly_name, poly_value in outlines.items():
            polygon = poly_value["polygon"]
            poly_color = poly_value["color"]
            #generate frame precursors from pose info
            frame_coords = polygon[idx]
            #convert into list of tuples (cv2 input)
            #opencv does not take float, so convert points into int for px values
            bp_points1 = frame_coords.astype(int)
            #pts = [(50, 50), (300, 190), (400, 10)]
            bp_points2 = list((map(tuple, bp_points1)))
            #cv2.polylines(img, np.array([pts]), True, RED, 5)
            cv2.fillPoly(img, np.array([bp_points2]), color= poly_color)
            if include_dots:
                for bp in bp_points2:
                    cv2.circle(img,bp, 2, White, -1)
        if show:
            # resize image
            resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("show", resize)
        # write as video
        out.write(img)
        # exit clauses
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    out.release()

"""Motion energy"""


def calc_motion_energy_single(frames):
    #norm_diff_list = []
    #calculates motion energy per bout then puts it back into dictionary for later sorting by example
    norm_diff_dict = {}
    for key, example in frames.items():
        abs_diff = np.absolute(np.diff(example, axis=0))
        norm_diff = np.nanmean(abs_diff, axis=0)
        #norm_diff_list.append(norm_diff)
        norm_diff_dict[key] = norm_diff

    return norm_diff_dict