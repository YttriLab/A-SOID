import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import cv2
import joblib
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.colors as mcolors


from config.help_messages import VIEW_LOADER_HELP, POLY_COUNT_HELP ,SINGLE_POLY_HELP, EGO_SELECT_HELP
from utils.import_data import load_labels
from utils.load_workspace import load_data, load_motion_energy, save_data
from utils.motionenergy import conv_2_egocentric, collect_labels, animate_blobs, calc_motion_energy_single


def label_blocks(df, clm_block):

    df_labeled = df.copy()
    df_labeled["block"] = (df_labeled[clm_block].shift(1) != df_labeled[clm_block]).astype(int).cumsum()

    return df_labeled


def get_events(df: pd.DataFrame, event_clm, label_clm, event):
    """
    This function returns a series of indexes (frames) where an event started (first TRUE value) and ended (first FALSE value)
    :param df: pd.Dataframe containing behavior labels
    :param event_clm: column name to use as status (TRUE/FALSE) in df
    :param label_clm: column name to use as block labels (1,2,3,4...) in df, will be used to identify start of each event
    :return: onset and offset index of events defined in event_clm and label_clm
    """
    event_df = df.copy()
    event_df[event] = event_df[event_clm] == event
    st.write(event_df)
    block_df = label_blocks(event_df, event)
    """Take trial labels to find start of every labeled block (Events and NonEvents) and drop all else"""
    unique_df = block_df.drop_duplicates(subset= ["block"])
    """Sort only for Event OnSets (True) and skip Event Ends (False)"""
    event_true = unique_df[event] == True
    event_false = ~event_true
    st.write(unique_df)
    """only take index"""
    onset_idx = unique_df[event_true].index
    offset_idx = unique_df[event_false].index
    return onset_idx, offset_idx

def get_block_boundaries(df, label_clm, cat_clm):
    """
    this function returns the start and stop of label block
    :param df: dataframe
    :param label_clm: column created by label_block, is searched for onset and offset of each block
    :param cat_clm: column used by label_block to look for blocks, will be used as key in block_dict
    :return: block_dict in style "block" = list(tuple(onset, offset), ...)
    """
    """Take trial labels to find start of every labeled block (Trial and NonTrials) and drop all else"""
    df_descend = df.sort_index(ascending= False) # flip to find last entry
    unique_df = df.drop_duplicates(subset= [label_clm]) # finds first entry
    unique_df_desc = df_descend.drop_duplicates(subset= [label_clm]) # finds last entry
    unique_df_desc = unique_df_desc.sort_index(ascending= True) # flip again
    block_dict = {}
    for block in unique_df[cat_clm].values:

        cluster_start = unique_df[cat_clm] == block
        cluster_stop = unique_df_desc[cat_clm] == block
        onset_idx = list(unique_df[cluster_start].index)
        offset_idx = list(unique_df_desc[cluster_stop].index)
        block_list = []
        for i in range(len(onset_idx)):
            block_list.append((onset_idx[i], offset_idx[i]))

        block_dict[block] = block_list

    return block_dict



class Viewer:

    def __init__(self, config = None):
        self.label_files = None
        self.label_csvs = None

        if config is not None:
            self.working_dir = config["Project"].get("PROJECT_PATH")
            self.prefix = config["Project"].get("PROJECT_NAME")
            self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
            self.framerate = config["Project"].getint("FRAMERATE")
            self.duration_min = config["Processing"].getfloat("MIN_DURATION")


        else:
            self.working_dir = None
            self.prefix = None
            self.annotation_classes = None
            self.framerate = None
            self.duration_min = None


        pass

    def upload_labels(self):

        upload_container = st.container()

        self.label_files = upload_container.file_uploader('Upload annotation or classification files that you want to view',
                                                             accept_multiple_files=True
                                                             ,type="csv"
                                                             ,key='label'
                                                             ,help=VIEW_LOADER_HELP)


    def plot_labels_matplotlib(self, labels):
        params = {"ytick.color": "w",
                  "xtick.color": "w",
                  "axes.labelcolor": "w",
                  "axes.edgecolor": "w"}
        plt.rcParams.update(params)

        #time_delta = labels["time"].iloc[1]
        labels = labels.drop(columns=["time"], errors="ignore")
        classes = list(labels.columns)
        label_names = np.argmax(labels.values, axis=1)
        # dictionary for code and corresponding labels
        assigned_labels = dict(zip(label_names,classes))
        # plot ethogram
        # plot them for comparison
        fig,ax = plt.subplots(1,figsize=(9,3))

        cmap = cm.get_cmap('tab10',len(classes))
        ethogram = plt.imshow(label_names[None, :]
                              ,aspect="auto"
                              ,cmap=cmap,interpolation="nearest"
                            )
        ethogram.set_clim(0,len(classes))
        plt.xlabel("Frames")
        plt.yticks([])
        cbar = plt.colorbar(ethogram)

        cbar.set_ticks(np.arange(0,len(classes)) + 0.5)
        cbar.set_ticklabels(classes)

        plt.tight_layout()
        st.pyplot(fig, facecolor= "black", transparent=True)

    def plot_labels_plotly(self,labels):

        time_delta = labels["time"].iloc[1]
        labels = labels.drop(columns=["time"],errors="ignore")

        cat_df = pd.from_dummies(labels)
        cat_df["label"] = pd.Categorical(cat_df[cat_df.columns[0]])
        cat_df["codes"] = cat_df["label"].cat.codes

        classes = list(labels.columns)
        test_view = labels.values * cat_df["codes"].values[:, None]
        #test_view[test_view == 0] = -1
        #fig = make_subplots(1,2)
        fig = px.imshow(test_view.T
                        , aspect= "auto"
                        , color_continuous_scale='Edge'
                        #,contrast_rescaling='infer'
                        ,y = classes
                        ,x = np.arange(labels.shape[0])*time_delta /60
                        #, zmin = 1
                        #,binary_string=True
                         )
        #fig.add_trace(go.Image(z = test_view),1,1)
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(fig,use_container_width=False)

    # def plot_labels(self,labels):
    #     plot_cont = st.container()
    #     time_delta = labels["time"].iloc[1]
    #     st.write(time_delta)
    #     labels = labels.drop(columns=["time"],errors="ignore")
    #     classes = list(labels.columns)
    #     cat_df = pd.from_dummies(labels)
    #     #time_line = pd.date_range("00:00:00", periods= cat_df.shape[0],freq=f"{time_delta}L").time
    #     # plot histogram
    #     #cat_df.index = time_line
    #
    #     cat_df["label"] = pd.Categorical(cat_df[cat_df.columns[0]])
    #     cat_df["codes"] = cat_df["label"].cat.codes
    #     cmap = cm.get_cmap('tab10',len(classes))
    #     #st.write(cat_df["label"].cat.codes)
    #     #extract blocks of same labels
    #     #block_df = label_blocks(cat_df, "label")
    #     #st.write(block_df)
    #     behavior_dict = {}
    #     for behavior in classes:
    #         onset, offset = get_events(cat_df, "label", "block", "Walk")
    #         behavior_dict[behavior] = [onset, offset]
    #
    #     st.write(behavior_dict)
    #
    #     block_dict = get_block_boundaries()
    #
    #
    #     #find start of each, stop is the index before the next start
    #
    #     # figures = [
    #     #     px.histogram(cat_df, histnorm='percent'),
    #     #     px.histogram(cat_df, histnorm='percent')
    #     # ]
    #     #
    #     #
    #     # fig = make_subplots(rows=1,cols=len(figures), shared_yaxes=True,)
    #     #
    #     # for i,figure in enumerate(figures):
    #     #
    #     #     for trace in range(len(figure["data"])):
    #     #         fig.add_trace(trace= figure["data"][trace],row= 1,col=i+1)
    #     #         fig.update_xaxes(title_text="Behavior Classes",row=1,col=i+1)
    #     #
    #     # fig.update_layout(yaxis_title="Percentage [%]", height=400, width=600)
    #     st.write(cat_df.shape)
    #     fig = px.pie(cat_df,values='codes',names='label',color='label')
    #     #fig = px.imshow(labels.values,binary_string=True, aspect= "auto")
    #     #fig = px.imshow(cat_df["codes"][None,:], aspect= "auto")
    #     #fig.update_yaxes(showticklabels = False)
    #
    #     plot_cont.plotly_chart(fig,use_container_width=False)

    def main(self):

        label_exp = st.expander("View label files")
        with label_exp:
            self.upload_labels()
            self.label_csvs = {}
            if self.label_files:
                for file in self.label_files:
                    file.seek(0)
                    temp_name = file.name
                    labels = load_labels(file,origin = "BORIS", fps = self.framerate)
                    self.label_csvs[temp_name] = labels

                for num, f_name in enumerate(self.label_csvs.keys()):

                    with st.expander(label = f_name ):
                        self.plot_labels_matplotlib(self.label_csvs[f_name])


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

    def select_outline(self):
        "Allows GUI selection of polygons made up by bodyparts as corners for blob animation"
        # available colors
        colors = dict(cyan = (255, 255, 0)
                      , magenta = (255, 0, 255))

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
            poly_selection = st.multiselect("Select the body parts to form the polygon"
                           ,self.keypoints
                           ,help = SINGLE_POLY_HELP
                           ,key = "poly_select{}".format(poly)
                                            )
            color_selection = st.selectbox("Select a color for that polygon"
                                           , list(colors.keys())
                                           ,key = "poly_select{}".format(poly)
                                           ,)

            outline_dict[polygon_key] = dict(order = poly_selection
                                             ,color = colors[color_selection]
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
                                                  f"{self.number_to_class[selected_class].replace(' ', '_')}_seq{sequence_number}_example{i}.avi")
                        #animate and save file
                        animate_blobs(class_data_ego, video_name, outlines=outline_dict)

    def extract_frames(self):
        video_import_path = os.path.join(self.working_dir, self.prefix, "animations")
        #frame_list = {}
        with st.spinner("Extracting frames from videos for actions"):
            for sdx in range(len(self.annotation_classes)):
                selected_behavior = self.annotation_classes[sdx]
                files = glob.glob(str.join('', (os.path.join(video_import_path, selected_behavior), '/*.avi')), recursive=True)
                if not files:
                    #when there are no files, just ignore it
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

    def calc_motion_energy(self):
        motion_energy_by_behavior = {}
        for i in stqdm(range(len(self.annotation_classes)), desc = "Calculating motion energy"):
            selected_behavior = self.annotation_classes[i]
            try:
                frames = self.load_frames_single(selected_behavior)
            except FileNotFoundError:
                continue
            motion_energy = calc_motion_energy_single(frames)
            motion_energy_by_behavior[selected_behavior] = motion_energy

        if not motion_energy_by_behavior:
            #if it is empty because it did not find any animations
            raise FileNotFoundError

        return motion_energy_by_behavior

    def view_motion_energy(self, motion_energy):

        #TODO UI adjustment of color range
        #c_range = st.slider("Color range", min_value = 0 , max_value = 100, [0, 20])
        c_range = [0, 20]
        motion_energy_keys = list(motion_energy.keys())
        fig = make_subplots(rows=1, cols=len(motion_energy_keys)
                            , subplot_titles= motion_energy_keys
                            )

        for i, action_type in enumerate(motion_energy_keys):
            #get all bouts per behavior
            all_bouts = list(motion_energy[action_type].values())
            #show the average motion energy across all bouts per behavior
            avg_motion_energy = np.nanmean(all_bouts, axis=0)
            fig.add_trace(go.Heatmap(z= avg_motion_energy
                                     ,name=action_type
                                     , showscale=True
                                     #colorbar=dict(title='Intensity',
                                     #              x=1.05, len=0.5),
                                     ,zmin=c_range[0]
                                     ,zmax=c_range[1]
                                     ,colorscale= px.colors.sequential.Viridis
                                     )
                          ,col = i +1
                          ,row = 1)

        fig.update_layout(title='Motion energy')
        fig.update_xaxes(showticklabels = False)
        fig.update_yaxes(showticklabels=False)
        #fig.update(layout_coloraxis_showscale=False)

        st.plotly_chart(fig, use_container_width=True)

    def get_motionenergy(self):

        param_exp = st.expander("Generate animations")
        motion_container = st.container()

        #get user input for egocentric alignment
        #TODO: limit selection to two
        with param_exp:
            ego_container = st.container()
            polygon_container = st.container()
            animation_info_box = st.empty()
            with ego_container:
                egocentric_bps = st.multiselect("Select body parts to align pose estimation to:", self.keypoints
                               #, max_selections  = 2 #only available for higher versions of streamlit
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
            motion_info_box = st.empty()
            #for later
            motion_energy = None

            try:
                # try to load the file if it already exists
                motion_energy = load_motion_energy(self.working_dir, self.prefix)
                self.view_motion_energy(motion_energy)

            except FileNotFoundError:
                if st.button("Calculate Motion Energy"):
                    motion_info_box = st.info("This may take a long time...")
                    try:
                        motion_energy = self.calc_motion_energy()
                        save_data(self.working_dir, self.prefix, "motionenergy.sav", motion_energy)
                    except FileNotFoundError:
                        # extract frames
                        try:
                            self.extract_frames()
                            motion_energy = self.calc_motion_energy()
                            save_data(self.working_dir, self.prefix, "motionenergy.sav", motion_energy)
                        except FileNotFoundError:
                            motion_info_box.error("Generate animations first.")

                if motion_energy is not None:
                    self.view_motion_energy(motion_energy)


    def main(self):

        blob_cont = st.container()
        with blob_cont:
            st.write("---")
            if self.working_dir is not None:
                self.get_motionenergy()