import os
import tempfile
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import moviepy.editor
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from utils.extract_features import frameshift_predict_proba, feature_extraction, frameshift_predict
from utils.load_workspace import load_test, load_iterX, \
    save_data, load_newest_model, load_test_performance
from utils.project_utils import update_config


def get_outline(order_list, D3=False):
    # TODO: Adjust to user-defined bps
    # bps = ['nose','ear_left','ear_right','neck','hip_left','hip_right','tail_base']
    bps = ['nose', 'neck', 'hip_left', 'hip_right', 'tail_base']
    keypoint_idx_dict = {}
    for num, bp in enumerate(bps):
        # for 3D
        if D3:
            keypoint_idx_dict[bp] = (num * 3, num * 3 + 1, num * 3 + 2)
        # for 2D
        else:
            keypoint_idx_dict[bp] = (num * 2, num * 2 + 1)

    order_idx = np.array([np.array(keypoint_idx_dict[x]) for x in order_list])
    return order_idx.flatten()


def get_outline_array(data, bp_idx):
    return data[:, bp_idx]


def animate_blobs_matplotlib(arr, framerate=30, D3=False, limiter=None, topdown=False):
    fig = plt.figure(figsize=(4, 2))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    # TODO: solving sizing problem
    if limiter is None:
        # assuming that the arr is a numpy array with x,y,x,y,x,y this will take every second column
        x_lim = np.amax(arr[:, ::2].flatten())
        y_lim = np.amax(arr[:, 1::2].flatten())
    else:
        x_lim = limiter[0]
        y_lim = limiter[1]
    if D3:
        ax = fig.add_subplot(111, projection="3d")
        # TODO: fix for 3D
        ax.set_xlim3d(0, x_lim)
        ax.set_ylim3d(0, y_lim)
        ax.set_zlim3d(0, limiter[2])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if topdown:
            # let's watch it from above (topdown)
            ax.view_init(azim=-90, elev=90)
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])

    ax.invert_yaxis()
    # white background for easier contrast:
    ax.set_facecolor("w")

    # TODO: clean up
    # outline for monkey data
    # ['nose', 'head', 'neck', 'RShoulder', 'RHand', 'Lshoulder', 'Lhand', 'hip', 'RKnee', 'RFoot', 'LKnee', 'Lfoot', 'tail']
    # order_dict = {"head": dict(order=["nose","head","neck","head"]
    #                            ,plt_type="line")
    #     ,"chest": dict(order=["neck","RShoulder","hip","Lshoulder"]
    #                    ,plt_type="poly")
    #     ,"Rarm": dict(order=["neck","RShoulder","RHand","RShoulder"]
    #                   ,plt_type="line")
    #     ,"Larm": dict(order=["neck","Lshoulder","Lhand","Lshoulder"]
    #                   ,plt_type="line")
    #     ,"Rleg": dict(order=["hip","RKnee","RFoot","RKnee"]
    #                   ,plt_type="line")
    #     ,"Lleg": dict(order=["hip","LKnee","Lfoot","LKnee"]
    #                   ,plt_type="line")
    #     ,"tail": dict(order=["hip","tail"]
    #                   ,plt_type="line")
    #               }

    # outline for mice data

    # TODO: adjust to user-defined outline
    temp_order_dict = {"head": dict(order=['nose', 'neck']
                                    , plt_type="line"),
                       "body": dict(order=['neck', 'hip_left', 'tail_base', 'hip_right']
                                    , plt_type="poly")
                       }
    # TODO: adjust to user-defined animals

    animal_list = [{"head": dict(order=['nose', 'neck']
                                 , plt_type="line"
                                 , color="cyan"
                                 ), "body": dict(order=['neck', 'hip_left', 'tail_base', 'hip_right']
                                                 , plt_type="poly"
                                                 , color="cyan"
                                                 )
                    },
                   {"head": dict(order=['nose', 'neck']
                                 , plt_type="line"
                                 , color="magenta"
                                 )
                       , "body": dict(order=['neck', 'hip_left', 'tail_base', 'hip_right']
                                      , plt_type="poly"
                                      , color="magenta"
                                      )
                    }
                   ]

    data_list = [arr[:, :arr.shape[1] // 2], arr[:, arr.shape[1] // 2:]]

    for animal_j in range(len(animal_list)):
        for bp in animal_list[animal_j].keys():
            temp_polygon = get_outline_array(data_list[animal_j],
                                             get_outline(animal_list[animal_j][bp]["order"], D3=D3))
            temp_color = animal_list[animal_j][bp]["color"]
            # reshape into [frames[bodyparts(x,y)]]
            if D3:
                temp_polygon = temp_polygon.reshape(
                    (-1, int(temp_polygon.shape[1] / 3), 3)
                )
                animal_list[animal_j][bp]["patch"] = Poly3DCollection(temp_polygon[0],
                                                                      closed=True,
                                                                      fc=temp_color,
                                                                      ec=temp_color)

                ax.add_collection3d(animal_list[animal_j][bp]["patch"])
                # add initial body part points
                animal_list[animal_j][bp]["scatter"] = ax.scatter(temp_polygon[0, :, 0],
                                                                  temp_polygon[0, :, 1],
                                                                  temp_polygon[0, :, 2],
                                                                  c="k", s=3)
            else:
                temp_polygon = temp_polygon.reshape(
                    (-1, int(temp_polygon.shape[1] / 2), 2)
                )
                animal_list[animal_j][bp]["patch"] = patches.Polygon(temp_polygon[0],
                                                                     closed=True,
                                                                     fc=temp_color,
                                                                     ec=temp_color)
                ax.add_patch(animal_list[animal_j][bp]["patch"])
                # add initial body part points
                animal_list[animal_j][bp]["scatter"] = ax.scatter(temp_polygon[0, :, 0],
                                                                  temp_polygon[0, :, 1],
                                                                  c="k", s=3)
            animal_list[animal_j][bp]["polygon"] = temp_polygon

    def init():
        return animal_list

    def animate(i):
        # change position and shape of patch frame by frame
        for animal_i in range(len(animal_list)):
            for bp in animal_list[animal_i].keys():
                temp_values = animal_list[animal_i][bp]["polygon"][i]
                if D3:
                    animal_list[animal_i][bp]["patch"].set_verts(temp_values)
                    # add points
                    animal_list[animal_i][bp]["scatter"]._offsets3d = temp_values.T
                else:
                    animal_list[animal_i][bp]["patch"].set_xy(temp_values)
                    # add points
                    animal_list[animal_i][bp]["scatter"].set_offsets(temp_values)
        return (
            animal_list
        )

    # animate
    ani = FuncAnimation(
        fig,
        animate,
        np.arange(0, arr.shape[0]),
        init_func=init,
        interval=1000 / framerate,
        blit=False,
    )

    # convert to HTML for easier display
    plt.close()
    return ani.to_jshtml()


# @st.cache(allow_output_mutation=True)
def create_anim(low_conf_list, pose_data, resolution):
    # creates animations for all low conf idx beforehand and saves them in cache!
    plot_dict = {x: None for x in low_conf_list}
    with st.spinner('creating animation...'):
        for low_conf_idx in low_conf_list:
            # TODO: Set to bout size
            bout_data = pose_data[low_conf_idx: low_conf_idx + 12]
            plot_dict[low_conf_idx] = animate_blobs_matplotlib(bout_data, limiter=resolution, D3=False)
    return plot_dict


def extract_vid_segments(video, frames2integ, low_conf_list, resolution):
    length = round(round(frames2integ/video.fps)*0.1, 2)
    plot_dict = {x: None for x in low_conf_list}
    count = 0
    with st.spinner('Creating video segments...'):
        for low_conf_idx in low_conf_list:
            # select a random time point
            start = low_conf_idx/video.fps
            # cut a subclip
            out_clip = video.subclip(start, start + length)
            out_clip.write_videofile(f'./out{count}.mp4')
            # video_bytes = io.BytesIO(out_clip.read())
            plot_dict[low_conf_idx] = f'./out{count}.mp4'
            count += 1
    return plot_dict


def pick_next_bout(low_conf_list, choice_dict):
    """Dummy function to pick random"""
    # get all other choices that are currently taken
    other_choices = []
    for clm in choice_dict.keys():
        other_choices.append(choice_dict[clm])
    # get remaining potential choices
    pot_choice_list = [pot_choice for pot_choice in low_conf_list if pot_choice not in other_choices]
    if pot_choice_list:
        return st.session_state["rndm_gen"].choice(pot_choice_list)
    else:
        # if no choice remain, return None
        return None


def init_active_learning(low_conf_idx, video, frames2integ, vid_resolution, annotation_classes, columns=3, reset=False):
    if reset:
        # Initialization
        st.session_state['low_conf_idx'] = low_conf_idx
        # init dictionary to collect all choices
        st.session_state['collection'] = {x: {"choice": None, "submitted": False} for x in
                                          st.session_state['low_conf_idx']}
        st.session_state["refined"] = False
        st.write("Creating animations for active learning...")
        # converts str config entry into [int, int] format

        resolution = vid_resolution
        # TODO: adjust to work with multiple pose estimation files in data (currently only works with one)
        # st.session_state["animations"] = create_anim(st.session_state['low_conf_idx'], pose_data,
        #                                              resolution=resolution)
        st.session_state["animations"] = extract_vid_segments(video, frames2integ,
                                                              st.session_state['low_conf_idx'],
                                                              resolution=resolution)
        # st.session_state["animations"] = None
        clm_list = st.columns(columns)
        st.session_state["rndm_gen"] = np.random.default_rng(42)
        st.session_state["choice_clms"] = {x: st.session_state["rndm_gen"].choice(st.session_state['low_conf_idx'])
                                           for x in clm_list}
        st.session_state["classes"] = annotation_classes
    else:
        # Initialization
        if 'low_conf_idx' not in st.session_state:
            st.session_state['low_conf_idx'] = low_conf_idx
        # init dictionary to collect all choices
        if 'collection' not in st.session_state:
            st.session_state['collection'] = {x: {"choice": None, "submitted": False} for x in
                                              st.session_state['low_conf_idx']}
        if "refined" not in st.session_state:
            st.session_state["refined"] = False
        if "animations" not in st.session_state:
            st.write("Creating animations for active learning...")
            # converts str config entry into [int, int] format
            # resolution = [int(str(val).strip()) for val in config["Project"].get("RESOLUTION").split(",")]
            resolution = vid_resolution
            # TODO: adjust to work with multiple pose estimation files in data (currently only works with one)
            # st.session_state["animations"] = create_anim(st.session_state['low_conf_idx'], pose_data,
            #                                              resolution=resolution)
            st.session_state["animations"] = extract_vid_segments(video, frames2integ,
                                                                  st.session_state['low_conf_idx'],
                                                                  resolution=resolution)
        clm_list = st.columns(columns)
        if "rndm_gen" not in st.session_state:
            st.session_state["rndm_gen"] = np.random.default_rng(42)
        if "choice_clms" not in st.session_state:
            st.session_state["choice_clms"] = {x: st.session_state["rndm_gen"].choice(st.session_state['low_conf_idx'])
                                               for x in clm_list}
        if "classes" not in st.session_state:
            # st.session_state["classes"] = [str(cls).strip() for cls in config["Project"].get("CLASSES").split(",")]
            st.session_state["classes"] = annotation_classes
    return clm_list


def reset_active_learning():
    # resets all session keys from previous iteration to allow to cycle through each it
    session_keys = ["low_conf_idx", "collection", "refined", "animations", "choice_clms"]
    for sess_key in session_keys:
        del st.session_state[sess_key]


class Refine:

    def __init__(self, working_dir, prefix, software, annotation_classes, frames2integ,
                 scalar, targets_test,
                 predict, proba, outlier_indices,
                 iterX_model, iterX_f1_scores,
                 new_videos, new_pose_list, p_cutoff, num_outliers,
                 config, label_filename, label_df, iteration):

        self.working_dir = working_dir
        self.prefix = prefix
        self.software = software
        self.classes = annotation_classes.split(', ')
        self.frames2integ = frames2integ
        self.scalar = scalar
        self.data_test = load_test(self.working_dir, self.prefix)
        self.targets_test = targets_test

        self.iterX_model = iterX_model
        self.iterX_f1_scores = iterX_f1_scores
        self.new_videos = new_videos
        # set videos up
        if self.new_videos[0] is not None:
            # Make temp file path from uploaded file
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp:
                bytes_data = self.new_videos[0].getvalue()
                temp.write(bytes_data)
            self.clip = moviepy.editor.VideoFileClip(temp.name)
        self.resolution = self.clip.size
        self.new_pose = new_pose_list
        self.p_cutoff = p_cutoff
        self.num_outliers = num_outliers
        self.config = config
        self.label_filename = label_filename

        self.predict = predict
        self.proba = proba
        self.outlier_indices = outlier_indices

        self.iterX_X_train_list = None
        self.iterX_Y_train_list = None

        self.label_df = label_df
        self.new_X = None
        self.new_Y = None
        self.new_model = None
        self.new_f1_scores = []
        self.new_macro_scores = []
        self.iteration = int(config["Processing"].get("ITERATION"))
        self.new_iter_prefix = None

    def predict_behavior_proba(self):
        # new test pose sequences, extract features, bin them
        if not self.software == 'CALMS21 (PAPER)':
            self.predict, self.proba = frameshift_predict_proba(self.new_pose, len(self.new_pose), self.scalar,
                                                                self.iterX_model, framerate=self.frames2integ)
        else:
            self.predict, self.proba = frameshift_predict_proba(self.new_pose, len(self.new_pose), self.scalar,
                                                                self.iterX_model, framerate=120)

    def subsample_outliers(self):
        idx_lowconf = np.where(self.proba.max(1) < self.p_cutoff)[0]
        # setting a random seed for sampling, seed is identical to train/test split
        np.random.seed(42)
        # KEEP IN MIND THIS HAS NO PREFERENCE FOR BEHAVIOR A VS B, JUST GRABBING THE WORST AND SUBSAMPLE
        try:
            # attempt sampling up to max_samples_per iteration
            idx_sampled = np.random.choice(np.arange(np.hstack(idx_lowconf).shape[0]),
                                           self.num_outliers, replace=False)
        except:
            # otherwise just grab all
            idx_sampled = np.random.choice(np.arange(np.hstack(idx_lowconf).shape[0]),
                                           np.hstack(idx_lowconf).shape[0], replace=False)
        self.outlier_indices = idx_lowconf[idx_sampled]

    def save_predict_proba(self):
        # save partitioned datasets, useful for cross-validation
        save_data(self.working_dir, self.prefix, f'{self.new_videos[0].name}-predict_proba.sav',
                  [self.new_videos,
                   self.new_pose,
                   self.p_cutoff,
                   self.num_outliers,
                   self.predict,
                   self.proba,
                   ])
        save_data(self.working_dir, self.prefix, f'{self.new_videos[0].name}-{self.label_filename}-refinements.sav',
                  [self.outlier_indices,
                   self.label_filename])

    def refine_outliers(self):
        if st.button('refine again'.upper()):
            clm_list = init_active_learning(self.outlier_indices, self.clip, self.frames2integ,
                                            self.resolution, self.classes, reset=True)
            self.active_learning_ui(clm_list)
        else:
            clm_list = init_active_learning(self.outlier_indices, self.clip, self.frames2integ,
                                            self.resolution, self.classes)
            self.active_learning_ui(clm_list)

    def active_learning_ui(self, clm_list):
        for num, col in enumerate(clm_list):
            # get current column
            current_colm = list(st.session_state["choice_clms"].keys())[num]
            # remove old thing
            placeholder = col.empty()
            # if there are still some idxs left:
            if len(st.session_state['low_conf_idx']) > 0:
                # take the current idx that was randomly picked
                idx = st.session_state["choice_clms"][current_colm]
                # if that choice is'nt in the list anymore, pick another:
                # the random pick will return None if there is no valid choice left
                if idx is not None:
                    if idx not in st.session_state['low_conf_idx']:
                        st.session_state["choice_clms"][current_colm] = pick_next_bout(
                            st.session_state['low_conf_idx'],
                            st.session_state["choice_clms"])
                        idx = st.session_state["choice_clms"][current_colm]
                    # wait for the user to make a choice
                    if not st.session_state['collection'][idx]["submitted"]:
                        # create form (special type of container)
                        header_container = st.container()
                        with placeholder.form("form_col{}_{}".format(num, idx)):
                            st.subheader(f"Low confidence bout {idx}")
                            # this is where the animation comes in
                            # TODO: fix size
                            # st.components.v1.html(st.session_state["animations"][idx], height=300)
                            st.video(st.session_state["animations"][idx])
                            # some pseudo parameters to init
                            classes = st.session_state["classes"]
                            returned_choice = st.radio("Select the correct class: ", classes,
                                                       index=int(np.argmax(self.proba[idx])),
                                                       key="radio_{}".format(idx))
                            st.session_state['collection'][idx]["submitted"] = \
                                st.form_submit_button("Submit",
                                                      "Press to confirm your choice")
                            if st.session_state['collection'][idx]["submitted"]:
                                # TODO: make format work with BORIS for cross-platform compatibility?
                                try:
                                    df = pd.read_csv(os.path.join(self.working_dir,
                                                                  self.prefix,
                                                                  self.label_filename), header=0)

                                    new_annotation = {'Index': [idx],
                                                      'Annotation': [returned_choice],
                                                      'Label': [classes.index(returned_choice)]}

                                    df2 = pd.DataFrame(new_annotation)
                                    df = df.append(df2, ignore_index=True)
                                    df.to_csv(os.path.join(self.working_dir,
                                                           self.prefix,
                                                           self.label_filename), index=False)

                                except FileNotFoundError:
                                    new_annotation = {'Index': [idx],
                                                      'Annotation': [returned_choice],
                                                      'Label': [classes.index(returned_choice)]}
                                    df = pd.DataFrame(new_annotation)
                                    df.to_csv(os.path.join(self.working_dir,
                                                           self.prefix,
                                                           self.label_filename), index=False)

                                # do this to instantely refresh the page after pressing submit!
                                # Otherwise it waits for another run
                                # autorefresh without the user interacting with anything
                                # show that user some appreciation
                                with placeholder:
                                    st.success("Submitted!")
                                    # wait for them to see it :)
                                    time.sleep(0.2)
                                st.experimental_rerun()
                    # if the choice was submitted, get a new random idx, and remove the old
                    else:
                        # choices.append(returned_choice)
                        # remove solved index, the order of those matters
                        st.session_state["choice_clms"][current_colm] = pick_next_bout(
                            st.session_state['low_conf_idx'],
                            st.session_state["choice_clms"])
                        # remove old ones, for numpy version:
                        # find index of idx then delete element from list (return copy without it)
                        idx_idx = np.argwhere(st.session_state['low_conf_idx'] == idx)
                        st.session_state['low_conf_idx'] = np.delete(st.session_state['low_conf_idx'], idx_idx)
                        # we need this so the new column is shown as soon as the "submitted" is gone
                        st.experimental_rerun()
            else:
                # if all are done, set a state to true
                st.session_state["refined"] = True

    def create_new_train_dataset(self):
        if self.iteration == 0:
            [self.iterX_model,
             self.iterX_X_train_list,
             self.iterX_Y_train_list,
             _, _, _] = load_iterX(self.working_dir, self.prefix)
        else:
            [self.iterX_model, self.iterX_X_train_list, self.iterX_Y_train_list] = \
                load_newest_model(self.working_dir, self.prefix)
        st.write(int(round(self.frames2integ/10)))
        pose_list = [np.vstack([np.vstack(self.new_pose)[idx:idx + int(round(self.frames2integ/10)), :]
                                for idx in self.outlier_indices])]
        features_new, _ = feature_extraction(pose_list, len(pose_list), framerate=self.frames2integ)
        scaled_feature_new = self.scalar.transform(features_new)
        self.new_X = np.vstack((self.iterX_X_train_list[-1][0],
                                scaled_feature_new))
        self.new_Y = np.hstack((self.iterX_Y_train_list[-1][0],
                                self.label_df['Label']))

    def train_new_classifier(self):
        self.new_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                criterion='gini',
                                                class_weight='balanced_subsample'
                                                )
        self.new_model.fit(self.new_X[self.new_Y < max(np.unique(self.targets_test)), :],
                           self.new_Y[self.new_Y < max(np.unique(self.targets_test))])

    def save_newest_model(self):
        # save partitioned datasets, useful for cross-validation
        self.iteration += 1
        self.new_iter_prefix = str.join('', (self.prefix, f'/iter_{self.iteration}'))
        os.makedirs(os.path.join(self.working_dir, self.new_iter_prefix), exist_ok=True)
        save_data(self.working_dir, self.new_iter_prefix, 'newest_model.sav',
                  [self.new_model,
                   self.new_X,
                   self.new_Y
                   ])
        parameters_dict = {
            "Processing": dict(
                ITERATION=self.iteration,
            )
        }
        update_config(os.path.join(self.working_dir, self.prefix), updated_params=parameters_dict)

    def validate(self):
        self.new_iter_prefix = str.join('', (self.prefix, f'/iter_{self.iteration}'))
        try:
            [self.new_f1_scores, self.new_macro_scores] = load_test_performance(self.working_dir,
                                                                                self.new_iter_prefix)
        except:
            try:
                [self.new_model, self.new_X, self.new_Y] = load_newest_model(self.working_dir,
                                                                             self.new_iter_prefix)
                predict = frameshift_predict(self.data_test, len(self.data_test), self.scalar,
                                             self.new_model, framerate=120)
                self.new_f1_scores.append(f1_score(
                    self.targets_test[self.targets_test < max(np.unique(self.targets_test))],
                    predict[self.targets_test < max(np.unique(self.targets_test))], average=None))
                self.new_macro_scores.append(f1_score(
                    self.targets_test[self.targets_test < max(np.unique(self.targets_test))],
                    predict[self.targets_test < max(np.unique(self.targets_test))], average='macro'))
                self.save_temp_performance()
            except FileNotFoundError:
                st.error('No refined model found, please merge dataset!')
        cols = st.columns(len(self.new_f1_scores[0]))
        for c, col in enumerate(cols):
            col.metric(f"{self.classes[c]}", f"{100 * round(self.new_f1_scores[0][c], 2)}%",
                       f"{100 * round(self.new_f1_scores[0][c] - self.iterX_f1_scores[-1][0][c], 2)}% from last")

    def save_temp_performance(self):
        save_data(self.working_dir, self.new_iter_prefix, 'test_performance.sav',
                  [self.new_f1_scores,
                   self.new_macro_scores,
                   ])

    def info_box(self):
        info_box = st.empty()
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        with info_box:
            if not st.session_state["low_conf_idx"].shape[0] == 0:
                st.info(
                    "Remaining samples to refine {} of {}".format(st.session_state["low_conf_idx"].shape[0],
                                                                  self.outlier_indices.shape[0]))
        if st.session_state["low_conf_idx"].shape[0] == 0:
            # if all are done: Tell the user the good news
            info_box.success("You refined all selected samples.")
            self.label_df = pd.read_csv(os.path.join(self.working_dir,
                                                     self.prefix,
                                                     self.label_filename), header=0)

            if button_col1.button('Merge dataset'.upper()):
                self.create_new_train_dataset()
                self.train_new_classifier()
                self.save_newest_model()
                st.success(f'Merged the {len(self.label_df)} refinements into model')
        # if self.new_model:
        if button_col5.button('Validate performance'.upper()):
            self.validate()

    def add_labels(self):
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('Merge dataset'.upper()):
            self.create_new_train_dataset()
            self.train_new_classifier()
            self.save_newest_model()
            st.success(f'Merged the {len(self.label_df)} refinements into model')
        if button_col5.button('Validate performance'.upper()):
            self.validate()

    # def main(self):
    #     self.predict_behavior_proba()
    #     self.subsample_outliers()
    #     self.refine_outliers()
    #     self.info_box()
