import streamlit as st
import joblib
import os
import glob

from scipy.spatial.transform import Rotation
import numpy as np

import cv2
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from PIL.ImageColor import getcolor

from stqdm import stqdm

from asoid.config.help_messages import *
from asoid.utils.load_workspace import load_data, load_motion_energy, save_data

#TODO: adapt to work with new iterations

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

class MotionEnergyMachine:

    def __init__(self, config):

        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")

        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")


        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.class_to_number = {s: i for i, s in enumerate(self.annotation_classes)}
        self.number_to_class = {i: s for i, s in enumerate(self.annotation_classes)}

        self.keypoints = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        keypoints_idx = np.arange(len(self.keypoints) * 2)
        keypoints_idx = np.reshape(keypoints_idx, (len(self.keypoints), 2))
        self.keypoints_to_idx = {self.keypoints[i]: list(keypoints_idx[i]) for i in range(len(self.keypoints))}

        [data, _] = load_data(self.working_dir, self.prefix)
        [self.processed_input_data, self.targets] = data

        #filenames
        self.filenames = [x.strip().split(".")[0] for x in config["Data"].get("DATA_INPUT_FILES").split(",")]

    def select_outline(self):
        "Allows GUI selection of polygons made up by bodyparts as corners for blob animation"
        # available colors
        colors = dict(cyan = (255, 255, 0)
                      , magenta = (255, 0, 255)
                      , red = (255, 0,0)
                      , lime = (0,255,0)
                      , blue = (0,0,255)
                      , yellow = (255, 255, 0)
                      , white = (255, 255, 255)
                      )

        poly_count = st.number_input("Number of Polygons"
                                     , help= POLY_COUNT_HELP
                                     , step = 1
                                     , min_value=1
                                     , key= "poly_count"
                                     )
        outline_dict = {}
        #TODO: Add image for visual explanation?
        for poly in range(int(poly_count)):
            polygon_key = "Polygon {}".format(poly)
            st.write(polygon_key)
            clm_1, clm_2 = st.columns([0.7, 0.3])
            poly_selection = clm_1.multiselect("Select the body parts to form the polygon"
                           ,self.keypoints
                           ,help = SINGLE_POLY_HELP
                           ,key = "poly_select{}".format(poly)
                                            )
            #hex code for cyan
            cyan = "#00ffff"

            color_selection = clm_2.color_picker("Select a color for that polygon"
                                                 , key="poly_select_{}".format(poly)
                                                 , value = cyan
                                                 , help = "If you want to change the color, just click on the color picker again."
                                                          " If you want to use the same color again, just copy the hex code and paste it into another color picker."
                                                          )



            outline_dict[polygon_key] = dict(order = poly_selection
                                             , color = getcolor(color_selection, "RGB")
                                            )

        #translate selected keypoints into indices
        for poly_part in outline_dict.keys():
            outline_dict[poly_part]["idx"] = []
            kp_names = outline_dict[poly_part]["order"]
            for kp in kp_names:
                outline_dict[poly_part]["idx"].append(self.keypoints_to_idx[kp])

        return outline_dict

    def create_blob_animation(self, sub_selected_classes, outline_dict, ref_origin_idx, ref_rot_idxs):
        """
        :params sub_selected_classes: List of subselected classes to generate animations for
        :params outline_dict, dict: Dictionary of user-defined polygons, including color and keypoint idxs
        :param ref_origin_idx, tuple/list: Idx of keypoint used for egocentric alignment as new origin
        :param ref_rot_idxs, tuple/list: Idx of keypoint used for egocentric alignment as x-axis
        """
        blob_info_box = st.empty()
        outpath = os.path.join(self.working_dir, self.prefix, "animations")

        # find all frames in all sequences for selected class:
        for i in stqdm(range(len(sub_selected_classes)), desc = "Collecting examples and animating blobs..."):
            selected_class = self.class_to_number[sub_selected_classes[i]]
            label_collection, total_labels = collect_labels(self.targets, selected_class)

            blob_info_box = st.info(f"Found {len(label_collection)} files with a total of {total_labels} for class {self.number_to_class[selected_class]}.")

            for sequence_number in stqdm(range(len(label_collection)), desc = "Going through files..."):
                transition_idx = np.where(np.diff(label_collection[sequence_number]) != 1)[0] + 1
                for i, t in enumerate(transition_idx):
                    collection_array = None
                    if i == 0:
                        for num, l_list in enumerate([label_collection[sequence_number][:t]]):
                            # select only frames from sequence with selected class
                            if collection_array is None:
                                collection_array = self.processed_input_data[sequence_number][l_list, :]
                            else:
                                collection_array = np.concatenate(
                                    [collection_array, self.processed_input_data[sequence_number][l_list, :]], axis=0)
                    else:
                        for num, l_list in enumerate([label_collection[sequence_number][transition_idx[i - 1]:t]]):
                            # select only frames from sequence with selected class
                            if collection_array is None:
                                collection_array = self.processed_input_data[sequence_number][l_list, :]
                            else:
                                collection_array = np.concatenate(
                                    [collection_array, self.processed_input_data[sequence_number][l_list, :]], axis=0)

                    ## motion energy works on >= 2 frames
                    if (len(collection_array) >= 2):
                        class_data_ego = conv_2_egocentric(collection_array,
                                                           ref_rot_idxs=ref_rot_idxs, ref_origin_idx=ref_origin_idx)
                        #create folder for class if not existing
                        vid_path = os.path.join(outpath, f"{self.number_to_class[selected_class]}")
                        os.makedirs(vid_path, exist_ok=True)
                        #generate video name on current parameters
                        video_name = os.path.join(vid_path,
                                                  f"{self.number_to_class[selected_class].replace(' ', '_')}_{self.filenames[sequence_number]}_example{i}.avi")
                        #animate and save file
                        animate_blobs(class_data_ego, video_name, outlines=outline_dict)

    def extract_frames_single(self, selected_behavior):
        """ Extracts frames from videos for a single behavior"""
        video_import_path = os.path.join(self.working_dir, self.prefix, "animations")
        info_box = st.empty()
        # frame_list = {}
        with st.spinner("Extracting frames from videos for actions"):

            files = glob.glob(str.join('', (os.path.join(video_import_path, selected_behavior), '/*.avi')),
                              recursive=True)
            if not files:
                raise FileNotFoundError("No videos found in folder. Please run the animation step first.")

            behavior_i = {}
            for idx in stqdm(range(len(files)), desc="Going through examples..."):
                curr_video = files[idx]
                curr_filename = os.path.basename(curr_video)[:-4]
                cap = cv2.VideoCapture(curr_video)
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                resize_f = 3
                resize_dim = (frameHeight // resize_f, frameWidth // resize_f)

                # convert video into numpy array of frames (binary)

                success, image = cap.read()
                count = 0
                frame_array = np.empty((frameCount, resize_dim[0], resize_dim[1]), np.dtype('uint8'))
                while count < frameCount - 1 and success:
                    success, img = cap.read()
                    # convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # convert to binary
                    thresh = 1
                    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
                    # resize for easier calculations
                    res_bw = cv2.resize(im_bw, (resize_dim[0], resize_dim[1]))
                    # add to numpy array
                    frame_array[count] = res_bw
                    count += 1
                behavior_i[curr_filename] = frame_array
                # frame_list[selected_behavior] = behavior_i
                cap.release()

            # save it for next time
            frames_collection_path = os.path.join(self.working_dir, self.prefix, "animations", selected_behavior,
                                                  "f_collection_{}".format(selected_behavior))
            with open(frames_collection_path, 'wb') as f:
                joblib.dump(behavior_i, f)


    def extract_frames(self):
        video_import_path = os.path.join(self.working_dir, self.prefix, "animations")
        info_box = st.empty()
        #frame_list = {}
        with st.spinner("Extracting frames from videos for actions"):
            for sdx in range(len(self.annotation_classes)):
                selected_behavior = self.annotation_classes[sdx]
                files = glob.glob(str.join('', (os.path.join(video_import_path, selected_behavior), '/*.avi')), recursive=True)
                if not files:
                    info_box.info("No animations found for {}, skipping that class.".format(selected_behavior))
                    continue
                behavior_i = {}
                for idx in stqdm(range(len(files)), desc = "Going through examples..."):
                    curr_video = files[idx]
                    curr_filename = os.path.basename(curr_video)[:-4]
                    cap = cv2.VideoCapture(curr_video)
                    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    resize_f = 3
                    resize_dim = (frameHeight // resize_f, frameWidth // resize_f)

                    # convert video into numpy array of frames (binary)

                    success, image = cap.read()
                    count = 0
                    frame_array = np.empty((frameCount, resize_dim[0], resize_dim[1]), np.dtype('uint8'))
                    while count < frameCount - 1 and success:
                        success, img = cap.read()
                        # convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # convert to binary
                        thresh = 1
                        im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
                        # resize for easier calculations
                        res_bw = cv2.resize(im_bw, (resize_dim[0], resize_dim[1]))
                        # add to numpy array
                        frame_array[count] = res_bw
                        count += 1
                    behavior_i[curr_filename]= frame_array
                    #frame_list[selected_behavior] = behavior_i
                    cap.release()

                # save it for next time
                frames_collection_path = os.path.join(self.working_dir, self.prefix, "animations",selected_behavior,  "f_collection_{}".format(selected_behavior))
                with open(frames_collection_path, 'wb') as f:
                    joblib.dump(behavior_i, f)

    def load_frames_single(self, selected_behavior):

        frames_collection_path = os.path.join(self.working_dir, self.prefix, "animations", selected_behavior,
                                              "f_collection_{}".format(selected_behavior))
        with open(frames_collection_path, 'rb') as fr:
            temp_frames = joblib.load(fr)

        return temp_frames

    def calc_motion_energy_single(self, selected_behavior):
        """Calculates motion energy for a single behavior and saves it to disk"""
        frames = self.load_frames_single(selected_behavior)
        motion_energy = calc_motion_energy_single(frames)
        self.save_motion_energy_single(motion_energy, selected_behavior)
        return motion_energy

    def calc_motion_energy_all(self):
        """Calculates motion energy for all behaviors and saves them seperately to disk
        :return: dictionary with motion energy for each behavior"""
        motion_energy_by_behavior = {}
        for i in stqdm(range(len(self.annotation_classes)), desc = "Calculating motion energy"):
            selected_behavior = self.annotation_classes[i]
            try:
                frames = self.load_frames_single(selected_behavior)
            except FileNotFoundError:
                continue
            motion_energy = calc_motion_energy_single(frames)
            self.save_motion_energy_single(motion_energy, selected_behavior)
            motion_energy_by_behavior[selected_behavior] = motion_energy

        if not motion_energy_by_behavior:
            #if it is empty because it did not find any animations
            raise FileNotFoundError

        return motion_energy_by_behavior

    def save_motion_energy_single(self, motion_energy, selected_behavior):
        """Saves motion energy for a single behavior"""
        # save it for next time by behavior
        motion_energy_path = os.path.join(self.working_dir, self.prefix, "animations", selected_behavior, "motion_energy.sav")
        with open(motion_energy_path, 'wb') as f:
            joblib.dump(motion_energy, f)

    def save_motion_energy_all(self, motion_energy):
        """Saves motion energy for all behaviors seperated by behavior"""
        # save it for next time by behavior
        for key in motion_energy.keys():
            motion_energy_path = os.path.join(self.working_dir, self.prefix, "animations", key, "motion_energy.sav")
            with open(motion_energy_path, 'wb') as f:
                joblib.dump(motion_energy[key], f)

    @st.cache_data
    def load_motion_energy_single(self, selected_behavior):
        """Loads motion energy for a single behavior"""
        motion_energy_path = os.path.join(self.working_dir, self.prefix, "animations", selected_behavior, "motion_energy.sav")
        with open(motion_energy_path, 'rb') as fr:
            temp_motion_energy = joblib.load(fr)

        return temp_motion_energy

    def load_motion_energy_all(self):
        """Loads motion energy for all behaviors"""
        motion_energy_by_behavior = {}
        for i in range(len(self.annotation_classes)):
            selected_behavior = self.annotation_classes[i]
            try:
                motion_energy = self.load_motion_energy_single(selected_behavior)
            except FileNotFoundError:
                continue
            motion_energy_by_behavior[selected_behavior] = motion_energy

        if not motion_energy_by_behavior:
            #if it is empty because it did not find any animations
            raise FileNotFoundError

        return motion_energy_by_behavior

    def view_motion_energy(self, motion_energy):
        """Visualizes motion energy for all behaviors in motion_energy dictionary"""
        #TODO UI adjustment of color range
        c_range = st.slider("Pick color range for motion energy display", min_value = 0 , max_value = 100,value = [0, 20])
        #c_range = [0, 20]
        motion_energy_keys = list(motion_energy.keys())

        n_cols = 5
        n_rows = int(np.ceil(len(motion_energy_keys)/n_cols))


        fig = make_subplots(rows=n_rows, cols=n_cols
                            , subplot_titles=motion_energy_keys
                            #, shared_xaxes=True, shared_yaxes=True
                            , column_widths=[1] * n_cols, row_heights=[1] * n_rows
                            , horizontal_spacing=0.05, vertical_spacing=0.1
                            )
        for i, action_type in enumerate(motion_energy_keys):
            #get all bouts per behavior
            all_bouts = list(motion_energy[action_type].values())
            #show the average motion energy across all bouts per behavior
            avg_motion_energy = np.nanmean(all_bouts, axis=0)
            fig.add_trace(go.Heatmap(z= avg_motion_energy
                                     ,name=action_type
                                     , showscale=False
                                     ,zmin=c_range[0]
                                     ,zmax=c_range[1]
                                     ,colorscale= px.colors.sequential.Viridis
                                     )
                          ,col = i % n_cols + 1
                          ,row = i // n_cols + 1)

        fig.update_layout(title='Motion energy'
                          )

        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        st.plotly_chart(fig, use_container_width=True, height= 600*n_rows)

    def get_motionenergy(self):

        param_container  = st.container()
        motion_container = st.container()

        #get user input for egocentric alignment
        #TODO: limit selection to two
        with param_container:
            st.subheader("Generate animations")
            ego_container = st.container()
            polygon_container = st.container()
            animation_info_box = st.empty()
            with ego_container:
                egocentric_bps = st.multiselect("Select body parts to align pose estimation to:", self.keypoints
                               , max_selections  = 2
                               ,help= EGO_SELECT_HELP)

                if len(egocentric_bps) >= 2:
                    #pick first two and transform into index
                    ref_origin_idx = self.keypoints_to_idx[egocentric_bps[0]]
                    ref_rot_idxs = self.keypoints_to_idx[egocentric_bps[1]]
                    animation_info_box.info("Reference for new origin: {} \n\n Reference for x-axis alignment: {}".format(egocentric_bps[0], egocentric_bps[1]))

            with polygon_container:
                #get user input to create polygons for blob animation using keypoints as corners
                outline_dict = self.select_outline()
                sub_selected_classes = st.multiselect("Select behavioral classes for animation",
                                                      self.annotation_classes, self.annotation_classes)

                if st.button("Create Animations"):
                    try:
                        self.create_blob_animation(sub_selected_classes, outline_dict, ref_origin_idx, ref_rot_idxs)
                        animation_info_box.info("Ready for motion energy calculcation.")
                    except:
                        animation_info_box.error("Enter required parameters first.")


        with motion_container:
            #for later
            st.subheader("Load and view motion energy")
            motion_energy = dict()
            selected_me_behaviors = st.multiselect("Select behavioral classes for motion energy view",
                                                   self.annotation_classes, self.annotation_classes)
            motion_info_box = st.empty()

            #button to load motion energy
            if st.checkbox("Load & view Motion Energy", key="me_load", value=False):
                #load motion energy based on selected behaviors
                for i in stqdm(range(len(selected_me_behaviors)), desc = "Collecting behaviors..."):
                    behavior = selected_me_behaviors[i]
                    try:
                        # load the motion energy
                        with st.spinner("Loading motion energy..."):
                            motion_energy[behavior] = self.load_motion_energy_single(behavior)
                    except FileNotFoundError:
                        motion_info_box.warning("Motion energy for {} not found. Calculating motion energy...".format(behavior))
                        try:
                            # if the file does not exist, calculate it
                            with st.spinner("Calculating motion energy..."):
                                motion_energy[behavior] = self.calc_motion_energy_single(behavior)
                        except FileNotFoundError:
                            # if the file does not exist, extract the frames and calculate it
                            try:
                                with st.spinner("Extracting frames..."):
                                    self.extract_frames_single(behavior)
                                with st.spinner("Calculating motion energy..."):
                                    motion_energy[behavior] = self.calc_motion_energy_single(behavior)
                            except FileNotFoundError:
                                # if nothing helps, skip the behavior
                                motion_info_box.warning("Frames for {} not found. Generate animations first. Skipping behavior.".format(behavior))
                                continue

                # if the dictionary is not empty, view it
                if motion_energy:
                    #view motion energy
                    motion_info_box.info("Motion energy loaded. Viewing motion energy...")
                    self.view_motion_energy(motion_energy)
                else:
                    motion_info_box.error("No motion energy found. Generate animations first.")
            else:
                motion_info_box.info("Select behaviors to load motion energy.")


    def main(self):

        blob_cont = st.container()
        with blob_cont:
            if self.working_dir is not None:
                self.get_motionenergy()