import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import streamlit as st
import io
import ffmpeg
from io import StringIO
import cv2
from stqdm import stqdm
import re
# import time
import tkinter as tk
import base64
from moviepy.editor import VideoFileClip
from asoid.utils.import_data import load_pose
from asoid.utils.preprocessing import adp_filt, sort_nicely
from asoid.utils.load_workspace import load_iterX, load_features, save_data, load_refinement, load_refine_params

from asoid.utils.extract_features_2D import feature_extraction
from asoid.utils.extract_features_3D import feature_extraction_3d
from asoid.utils.predict import bsoid_predict_numba, bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale
from asoid.config.help_messages import *

def frame_extraction(video_file, frame_dir):
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(
        int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(
            video_info['avg_frame_rate'].rpartition('/')[2]))
    if st.button('Start frame extraction for {} frames '
                 'at {} frames per second'.format(num_frames, avg_frame_rate)):
        st.info('Extracting frames from the video... ')
        # if frame_dir
        try:
            (ffmpeg.input(video_file)
             .filter('fps', fps=avg_frame_rate)
             .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                     s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))),
                     sws_flags='bilinear', start_number=0)
             .run(capture_stdout=True, capture_stderr=True))
            st.info(
                'Done extracting **{}** frames from video **{}**.'.format(num_frames, video_file))
        except ffmpeg.Error as e:
            st.error('stdout:', e.stdout.decode('utf8'))
            st.error('stderr:', e.stderr.decode('utf8'))
        st.info('Done extracting {} frames from {}'.format(num_frames, video_file))
        st.success('Done. Type "R" to refresh.')


def create_labeled_vid_old(labels, proba,
                           outlier_method, p_cutoff,
                           counts, frames2integ,
                           framerate, output_fps, annotation_classes,
                           frame_dir, output_path):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    number_of_frames = int(frames2integ / 10)

    # Center coordinates
    center_coordinates = (50, 50)
    # Radius of circle
    radius = 20
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness_circle = -1

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (width - 50, height - 50)
    fontScale = 0.5
    fontColor = (0, 0, 255)
    thickness_text = 1
    lineType = 2
    all_ex_idx = {key: [] for key in annotation_classes}
    for b in range(len(list(all_ex_idx.keys()))):
        with st.spinner(f'generating videos for behavior {annotation_classes[int(b)]}'):
            idx_b = np.where(labels == b)[0]
            # if there is such label

            if idx_b is not None:
                # st.write(outlier_method)
                if outlier_method == 'Low Confidence':
                    idx_b_poor = np.where(np.max(proba[idx_b, :], axis=1) < p_cutoff)[0]
                    behav_ex_idx = []
                    # if there are poor predictions
                    if idx_b_poor is not None:
                        try:
                            examples_b = np.random.choice(idx_b[idx_b_poor], counts, replace=False)
                        except:
                            examples_b = np.random.choice(idx_b[idx_b_poor], len(idx_b[idx_b_poor]), replace=False)
                        count = 0
                elif outlier_method == 'Random':
                    behav_ex_idx = []
                    try:
                        examples_b = np.random.choice(idx_b, counts, replace=False)
                    except:
                        examples_b = np.random.choice(idx_b, len(idx_b), replace=False)
                    count = 0

                for ex, example_b in enumerate(stqdm(examples_b, desc="creating videos")):
                    # just in case if future frames are not present
                    if (example_b - 1) * number_of_frames + int(3 * number_of_frames) - 1 < len(images):
                        video_name = 'behavior_{}_example_{}.mp4'.format(annotation_classes[int(b)], int(count))
                        grp_images = []
                        for f in range(number_of_frames):
                            rgb_im = cv2.imread(
                                os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                            # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                            cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        thickness_text,
                                        lineType)

                            # TODO: put timestamp text on image

                            grp_images.append(rgb_im)

                        for f in range(number_of_frames, int(2 * number_of_frames)):
                            rgb_im = cv2.imread(
                                os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))

                            # Draw a circle with blue line borders of thickness of 2 px
                            rgb_im = cv2.circle(rgb_im, center_coordinates, radius, color, thickness_circle)

                            cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        thickness_text,
                                        lineType)
                            # TODO: put timestamp text on image

                            grp_images.append(rgb_im)

                        for f in range(int(2 * number_of_frames), int(3 * number_of_frames)):
                            rgb_im = cv2.imread(
                                os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                            # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                            cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        thickness_text,
                                        lineType)
                            # TODO: put timestamp text on image

                            grp_images.append(rgb_im)

                        video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps,
                                                (width, height))
                        for j, image in enumerate(grp_images):
                            video.write(image)
                        cv2.destroyAllWindows()
                        video.release()
                        videoClip = VideoFileClip(os.path.join(output_path, video_name))
                        vid_prefix = video_name.rpartition('.mp4')[0]
                        gif_name = f"{vid_prefix}.gif"
                        videoClip.write_gif(os.path.join(output_path, gif_name))
                        behav_ex_idx.append(example_b)
                        count += 1
        all_ex_idx[annotation_classes[int(b)]] = behav_ex_idx

    return all_ex_idx


def create_labeled_vid(labels, proba,
                       outlier_method, p_cutoff,
                       counts, frames2integ,
                       framerate, output_fps, min_n_seconds, annotation_classes,
                       frame_dir, output_path):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    number_of_frames = int(frames2integ / 10)
    min_ex_bins = int(min_n_seconds * (framerate / number_of_frames))

    # Center coordinates
    center_coordinates = (50, 50)
    # Radius of circle
    radius = 20
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness_circle = -1

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (width - 50, height - 50)
    fontScale = 0.5
    fontColor = (0, 0, 255)
    thickness_text = 1
    lineType = 2

    # compute bout start/end and lengths

    bout_end_idx = np.where(np.diff(labels) != 0)[0]
    bout_start_idx = np.hstack((0, bout_end_idx + 1))
    bout_len = np.hstack((np.diff(bout_start_idx), len(labels) - bout_start_idx[-1]))

    all_ex_idx = {key: [] for key in annotation_classes}
    for b in range(len(list(all_ex_idx.keys()))):
        with st.spinner(f'generating videos for behavior {annotation_classes[int(b)]}'):

            # get bout starts and their lengths for label b
            idx_b = np.where(labels[bout_start_idx] == b)[0]
            example_indices = []
            idx_start_ls = []
            examples_b = []
            if idx_b is not None:
                len_b = bout_len[labels[bout_start_idx] == b]
                # for each example, get the actual label index start to end

                indices_start = [int(bout_start_idx[idx_b[b]]) for b in range(len(idx_b))]
                indices_end = [int(bout_start_idx[idx_b[b]] + len_b[b]) for b in range(len(idx_b))]

                if outlier_method == 'Low Confidence':
                    idx_b_poor = np.where(np.max(proba[idx_b, :], axis=1) < p_cutoff)[0]
                    behav_ex_idx = []
                    # if there are poor predictions
                    if idx_b_poor is not None:
                        try:
                            examples_b = np.random.choice(idx_b[idx_b_poor], counts, replace=False)
                        except:
                            examples_b = np.random.choice(idx_b[idx_b_poor], len(idx_b[idx_b_poor]), replace=False)
                        count = 0
                elif outlier_method == 'Random':
                    behav_ex_idx = []

                    for i, idx_start in enumerate(indices_start):
                        # find index greater than duration
                        if indices_end[i] - idx_start >= min_ex_bins:
                            idx_start_ls.append(i)

                    try:
                        # subsample to user choice
                        chosen_starts = np.random.choice(idx_start_ls,
                                                         counts, replace=False)
                    except:
                        chosen_starts = np.random.choice(idx_start_ls,
                                                         len(idx_start_ls), replace=False)

                    # get start stop for user choice
                    for j in chosen_starts:
                        examples_b.append([indices_start[j], indices_end[j]])
                    # st.write(examples_b)


                    count = 0
                for ex, example_b in enumerate(stqdm(examples_b, desc="creating videos")):
                    # just in case if future frames are not present
                    video_name = 'behavior_{}_example_{}.mp4'.format(annotation_classes[int(b)], int(count))
                    grp_images = []
                    # for example in 0.1s as binning size, this will iterate through frame index of 0.1s
                    # so will need to upsample to match framerate
                    for f in range(int(number_of_frames * (example_b[1] - example_b[0]))):
                        rgb_im = cv2.imread(
                            os.path.join(frame_dir, images[example_b[0] * number_of_frames + f]))

                        # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                        # cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                        #             bottomLeftCornerOfText,
                        #             font,
                        #             fontScale,
                        #             fontColor,
                        #             thickness_text,
                        #             lineType)

                        # TODO: put timestamp text on image
                        cv2.putText(rgb_im, f'{np.round(f * 1 / framerate, 1)}s',
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness_text,
                                    lineType)

                        grp_images.append(rgb_im)

                        # if (example_b - 1) * number_of_frames + int(3 * number_of_frames) - 1 < len(images):
                        #     video_name = 'behavior_{}_example_{}.mp4'.format(annotation_classes[int(b)], int(count))
                        #     grp_images = []
                        #     for f in range(number_of_frames):
                        #         rgb_im = cv2.imread(
                        #             os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                        #         # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                        #         cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                        #                     bottomLeftCornerOfText,
                        #                     font,
                        #                     fontScale,
                        #                     fontColor,
                        #                     thickness_text,
                        #                     lineType)
                        #
                        #         # TODO: put timestamp text on image
                        #
                        #         grp_images.append(rgb_im)
                        #
                        #     for f in range(number_of_frames, int(2 * number_of_frames)):
                        #         rgb_im = cv2.imread(
                        #             os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                        #
                        #         # Draw a circle with blue line borders of thickness of 2 px
                        #         rgb_im = cv2.circle(rgb_im, center_coordinates, radius, color, thickness_circle)
                        #
                        #         cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                        #                     bottomLeftCornerOfText,
                        #                     font,
                        #                     fontScale,
                        #                     fontColor,
                        #                     thickness_text,
                        #                     lineType)
                        #         # TODO: put timestamp text on image
                        #
                        #         grp_images.append(rgb_im)
                        #
                        #     for f in range(int(2 * number_of_frames), int(3 * number_of_frames)):
                        #         rgb_im = cv2.imread(
                        #             os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                        #         # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                        #         cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                        #                     bottomLeftCornerOfText,
                        #                     font,
                        #                     fontScale,
                        #                     fontColor,
                        #                     thickness_text,
                        #                     lineType)
                        #         # TODO: put timestamp text on image
                        #
                        #         grp_images.append(rgb_im)

                    video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps,
                                            (width, height))
                    for j, image in enumerate(grp_images):
                        video.write(image)
                    cv2.destroyAllWindows()
                    video.release()
                    videoClip = VideoFileClip(os.path.join(output_path, video_name))
                    vid_prefix = video_name.rpartition('.mp4')[0]
                    # gif_name = f"{vid_prefix}.gif"
                    # videoClip.write_gif(os.path.join(output_path, gif_name))
                    behav_ex_idx.append(example_b)
                    count += 1
        all_ex_idx[annotation_classes[int(b)]] = behav_ex_idx

    return all_ex_idx


def create_videos(new_pose_csvs, selected_bodyparts, llh_value, iterX_model, framerate, frames2integ,
                  outlier_method, p_cutoff, num_outliers, output_fps, min_n_seconds, annotation_classes,
                  is_3d, software, multi_animal, frame_dir, shortvid_dir):
    examples_idx = None
    features = [None]
    predict_arr = None
    examples_idx = None
    action_button = st.button("Predict labels and create example videos")
    message_box = st.empty()
    if action_button:
        st.session_state['disabled'] = True
        message_box.info('Predicting labels... ')
        # extract features, bin them
        features = []
        # filter here
        for i, f in enumerate(stqdm([new_pose_csvs], desc="Extracting spatiotemporal features from pose")):
        # # for i, f in enumerate([new_pose_csvs]):
        #     current_pose = pd.read_csv(f,
        #                                header=[0, 1, 2], sep=",", index_col=0
        #                                )
        #     bp_level = 1
        #     bp_index_list = []
        #     for bp in selected_bodyparts:
        #         bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
        #         bp_index_list.append(bp_index)
        #     selected_pose_idx = np.sort(np.array(bp_index_list).flatten())
        #     # get rid of likelihood columns for deeplabcut
        #     idx_llh = selected_pose_idx[2::3]
        #     # the loaded sleap file has them too, so exclude for both
        #     idx_selected = [i for i in selected_pose_idx if i not in idx_llh]
        #     filt_pose, _ = adp_filt(current_pose, idx_selected, idx_llh, llh_value)

            current_pose = load_pose(f, origin=software.lower(), multi_animal=multi_animal)
            # take user selected bodyparts
            bp_level = 1
            bp_index_list = []
            for bp in selected_bodyparts:
                bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
                bp_index_list.append(bp_index)
            selected_pose_idx = np.sort(np.array(bp_index_list).flatten())

            # get likelihood column idx directly from dataframe columns
            idx_llh = [i for i, s in enumerate(current_pose.columns) if "likelihood" in s]

            # # get rid of likelihood columns for deeplabcut
            # idx_llh = self.selected_pose_idx[2::3]

            # the loaded sleap file has them too, so exclude for both
            idx_selected = [i for i in selected_pose_idx if i not in idx_llh]

            # filtering does not work for 3D yet
            # check if there is a z coordinate

            if "z" in current_pose.columns.get_level_values(2):
                if is_3d is not True:
                    st.error("3D data detected. But parameter is set to 2D project.")
                print("3D project detected. Skipping likelihood adaptive filtering.")
                # if yes, just drop likelihood columns and pick the selected bodyparts
                filt_pose = current_pose.iloc[:, idx_selected].values
            else:
                filt_pose, _ = adp_filt(current_pose, idx_selected, idx_llh, llh_value)

            # using feature scaling from training set
            if not is_3d:
                feats, _ = feature_extraction([filt_pose], 1, frames2integ)
            else:
                feats, _ = feature_extraction_3d([filt_pose], 1, frames2integ)

            features.append(feats)
        for i in stqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
                pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
                predict_arr = np.array(predict).flatten()
        try:
            examples_idx = create_labeled_vid(predict_arr, pred_proba[0],
                                              outlier_method, p_cutoff,
                                              num_outliers, frames2integ,
                                              framerate, output_fps, min_n_seconds, annotation_classes,
                                              frame_dir, shortvid_dir)
            st.balloons()
            message_box.success('Done. Type "R" to refresh.')
        except:
            st.info('Terminated early. Type "R" to refresh.')

    return features[0], predict_arr, examples_idx


def prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                 outlier_methods, threshold, min_duration,
                 framerate, videos_dir, project_dir, iter_dir):
    left_col, right_col = st.columns(2)
    le_exapnder = left_col.expander('video'.upper(), expanded=True)
    ri_exapnder = right_col.expander('pose'.upper(), expanded=True)
    outlier_method = None
    p_cutoff = None
    num_outliers = 0
    output_fps = None
    min_n_seconds = 0
    frame_dir = None
    shortvid_dir = None

    if software == 'CALMS21 (PAPER)':
        st.error('CALMS21 (PAPER) is not supported for this feature.')
        st.stop()
    else:
        # try:
        new_videos = le_exapnder.file_uploader('Upload Video Files',
                                               accept_multiple_files=False,
                                               type=['avi', 'mp4'], key='video')
        new_pose_csvs = [ri_exapnder.file_uploader('Upload Corresponding Pose Files',
                                                   accept_multiple_files=False
                                                   , type=ftype
                                                    , key='pose')]

        if new_videos is not None and new_pose_csvs[0] is not None:
            st.session_state['uploaded_vid'] = new_videos
            # new_pose_list = []
            # for i, f in enumerate(new_pose_csvs):
            #     current_pose = pd.read_csv(f,
            #                                header=[0, 1, 2], sep=",", index_col=0
            #                                )
            #     bp_level = 1
            #     bp_index_list = []
            #     for bp in selected_bodyparts:
            #         bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
            #         bp_index_list.append(bp_index)
            #     selected_pose_idx = np.sort(np.array(bp_index_list).flatten())
            #
            #     # get rid of likelihood columns for deeplabcut
            #     idx_llh = selected_pose_idx[2::3]
            #     # the loaded sleap file has them too, so exclude for both
            #     idx_selected = [i for i in selected_pose_idx if i not in idx_llh]
            #     filt_pose, _ = adp_filt(current_pose, idx_selected, idx_llh)
            #     # new_pose_list.append(filt_pose)
            #     # new_pose_list.append(np.array(current_pose.iloc[:, idx_selected]))
            #     new_pose_list.append(current_pose)
            #
            #
            # st.session_state['uploaded_pose'] = new_pose_list
            col1, col3 = st.columns(2)
            col1_exp = col1.expander('Parameters'.upper(), expanded=True)
            col3_exp = col3.expander('Output folders'.upper(), expanded=True)

            outlier_method = col1_exp.selectbox('Outlier method',
                                                outlier_methods, index=0,
                                                disabled=st.session_state.disabled
                                                , help=OUTLIER_HELP)
            if outlier_method == 'Low Confidence':
                p_cutoff = col1_exp.number_input('Threshold value to sample outliers from',
                                                 min_value=0.0, max_value=1.0, value=threshold,
                                                 disabled=st.session_state.disabled)
            min_n_seconds = col1_exp.number_input('Minimum number of seconds for example',
                                                  min_value=min_duration, max_value=10.0, value=min_duration * 10,
                                                  disabled=st.session_state.disabled
                                                  , help = MINIMUM_SECONDS_HELP)

            num_outliers = col1_exp.number_input('Number of examples to refine',
                                                 min_value=3, max_value=None, value=5,
                                                 disabled=st.session_state.disabled
                                                 , help = REFINMEMENT_SAMPLE_HELP)
            st.session_state['refinements'] = {key:
                                                   {k: {'choice': None, 'submitted': False}
                                                    for k in range(num_outliers)}
                                               for key in annotation_classes}

            output_fps = col1_exp.number_input('Video playback fps',
                                               min_value=1, max_value=None, value=framerate,
                                               disabled=st.session_state.disabled
                                               , help = PLAYBACK_SPEED_HELP)

            col1_exp.write(f'Equivalent to {round(output_fps / framerate, 2)} X speed')
            if st.session_state['uploaded_vid'].name.endswith('mp4'):
                frame_dir = os.path.join(videos_dir,
                                         str.join('', (st.session_state['uploaded_vid'].name.rpartition('.mp4')[0],
                                                       '_pngs')))
            else:
                frame_dir = os.path.join(videos_dir,
                                         str.join('', (st.session_state['uploaded_vid'].name.rpartition('.avi')[0],
                                                       '_pngs')))
            os.makedirs(frame_dir, exist_ok=True)
            col3_exp.success(f'Entered **{frame_dir}** as the frame directory.')
            if st.session_state['uploaded_vid'].name.endswith('mp4'):
                shortvid_dir = os.path.join(project_dir, iter_dir,
                                            str.join('', (st.session_state['uploaded_vid'].name.rpartition('.mp4')[0],
                                                          '_refine_vids')))
            else:
                shortvid_dir = os.path.join(project_dir, iter_dir,
                                            str.join('', (st.session_state['uploaded_vid'].name.rpartition('.avi')[0],
                                                          '_refine_vids')))
            os.makedirs(shortvid_dir, exist_ok=True)
            col3_exp.success(f'Entered **{shortvid_dir}** as the refined video directory.')

        else:
            # st.session_state['uploaded_pose'] = []
            st.session_state['uploaded_vid'] = None

    return outlier_method, p_cutoff, num_outliers, output_fps, min_n_seconds, frame_dir, shortvid_dir


def prompt_setup_existing(outlier_methods, framerate, videos_dir, project_dir, iter_dir, selected_refine_dir):
    video_name = selected_refine_dir.rpartition('_refine_vids')[0]
    shortvid_dir = os.path.join(project_dir, iter_dir, selected_refine_dir)
    st.session_state['disabled'] = True
    col1, col3 = st.columns(2)
    col1_exp = col1.expander('Parameters'.upper(), expanded=True)
    col3_exp = col3.expander('Output folders'.upper(), expanded=True)

    outlier_method = col1_exp.selectbox('Outlier method',
                                        outlier_methods,
                                        index=outlier_methods.index(st.session_state['outlier_method']),
                                        disabled=st.session_state.disabled)
    if outlier_method == 'Low Confidence':
        p_cutoff = col1_exp.number_input('Threshold value to sample outliers from',
                                         min_value=0.0, max_value=1.0, value=st.session_state['p_cutoff'],
                                         disabled=st.session_state.disabled)
    min_n_seconds = col1_exp.number_input('Minimum number of seconds for example',
                                          min_value=0.1, max_value=10.0,
                                          value=st.session_state['min_n_seconds'],
                                          disabled=st.session_state.disabled)

    num_outliers = col1_exp.number_input('Number of examples to refine',
                                         min_value=3, max_value=None, value=st.session_state['num_outliers'],
                                         disabled=st.session_state.disabled)

    output_fps = col1_exp.number_input('Video playback fps',
                                       min_value=1, max_value=None, value=st.session_state['output_fps'],
                                       disabled=st.session_state.disabled)

    col1_exp.write(f'equivalent to {round(output_fps / framerate, 2)} X speed')
    frame_dir = col3_exp.text_input('Enter a directory for frames',
                                    os.path.join(videos_dir,
                                                 str.join('', (
                                                     video_name,
                                                     '_pngs'))),
                                    disabled=st.session_state.disabled, on_change=disable
                                    )

    try:
        os.listdir(frame_dir)
        col3_exp.success(f'Entered **{frame_dir}** as the frame directory.')
    except FileNotFoundError:
        if col3_exp.button('create frame directory'):
            os.makedirs(frame_dir, exist_ok=True)
            # st.rerun()

    try:
        os.listdir(shortvid_dir)
        col3_exp.success(f'Entered **{shortvid_dir}** as the refined video directory.')
    except FileNotFoundError:
        if col3_exp.button('create refined video directory'):
            os.makedirs(shortvid_dir, exist_ok=True)
            # st.rerun()

    return frame_dir, shortvid_dir


def disable():
    st.session_state["disabled"] = True


class Refinement:
    def __init__(self, ri, config):
        self.ri = ri
        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.software = config["Project"].get("PROJECT_TYPE")
        self.is_3d = config["Project"].getboolean("IS_3D")
        self.multi_animal = config["Project"].getboolean("MULTI_ANIMAL")
        self.ftype = [x.strip() for x in config["Project"].get("FILE_TYPE").split(",")]
        self.exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
        self.annotation_classes_ex = self.annotation_classes.copy()
        if self.exclude_other:
            self.annotation_classes_ex.pop(self.annotation_classes_ex.index('other'))
        #self.threshold = config["Processing"].getfloat("SCORE_THRESHOLD")
        self.selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        self.threshold = 0.1
        self.llh_value = config["Processing"].getfloat("LLH_VALUE")
        self.iteration = config["Processing"].getint("ITERATION")
        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")
        self.selected_iter = ri.selectbox('Select Iteration #', np.arange(self.iteration + 1), self.iteration)
        self.project_dir = os.path.join(self.working_dir, self.prefix)
        self.iter_folder = str.join('', ('iteration-', str(self.selected_iter)))
        self.videos_dir = os.path.join(self.project_dir, 'videos')
    def main(self):
       os.makedirs(os.path.join(self.project_dir, self.iter_folder), exist_ok=True)
       os.makedirs(self.videos_dir, exist_ok=True)
       refined_vid_dirs = [d for d in os.listdir(os.path.join(self.project_dir, self.iter_folder))
                           if os.path.isdir(os.path.join(self.project_dir, self.iter_folder, d))]
       refined_vid_dirs.extend(['Add New Video'])
       outlier_methods = ['Random', 'Low Confidence']
       try:
           if st.session_state['uploaded_vid'].name.endswith('mp4'):
               new_vid_name = str.join('',
                                       (st.session_state['uploaded_vid'].name.rpartition('.mp4')[0], '_refine_vids'))
           else:
               new_vid_name = str.join('',
                                       (st.session_state['uploaded_vid'].name.rpartition('.avi')[0], '_refine_vids'))
           selected_refine_dir = self.ri.radio('Select Refinement', refined_vid_dirs,
                                          index=refined_vid_dirs.index(new_vid_name),
                                          horizontal=True, key='selected_refine'
                                          , help=REFINEMENT_SELECT_HELP)
       except:
           selected_refine_dir = self.ri.radio('Select Refinement', refined_vid_dirs,
                                          horizontal=True, key='selected_refine'
                                          , help=REFINEMENT_SELECT_HELP)

       if self.software == 'CALMS21 (PAPER)':
           st.error("The CALMS21 data set is not designed to be used with the refinement step.")
           st.stop()
       else:
           if 'disabled' not in st.session_state:
               st.session_state['disabled'] = False

           # if 'refinements' not in st.session_state or 'refined' not in st.session_state:
           if 'refined' not in st.session_state:
               try:
                   [st.session_state['outlier_method'],
                    st.session_state['p_cutoff'],
                    st.session_state['min_n_seconds'],
                    st.session_state['num_outliers'],
                    st.session_state['output_fps'],
                    ] = load_refine_params(os.path.join(self.project_dir, self.iter_folder),
                                           selected_refine_dir)
                   try:
                       [st.session_state['video_path'],
                        st.session_state['features'],
                        st.session_state['predict'],
                        st.session_state['examples_idx'],
                        st.session_state['refined']] = load_refinement(os.path.join(self.project_dir, self.iter_folder),
                                                                       selected_refine_dir)
                   except:
                       st.session_state['refined'] = {key: {k: None for k in range(st.session_state['num_outliers'])}
                                                      for key in self.annotation_classes}
               except:
                   st.session_state['refined'] = {key: {k: None for k in range(20)}
                                                  for key in self.annotation_classes}
           if 'video_path' not in st.session_state:
               st.session_state['video_path'] = None
           if 'features' not in st.session_state:
               st.session_state['features'] = None
           if 'predict' not in st.session_state:
               st.session_state['predict'] = None
           if 'outlier_method' not in st.session_state:
               st.session_state['outlier_method'] = None
           if 'p_cutoff' not in st.session_state:
               st.session_state['p_cutoff'] = None
           if 'min_n_seconds' not in st.session_state:
               st.session_state['min_n_seconds'] = None
           if 'num_outliers' not in st.session_state:
               st.session_state['num_outliers'] = None
           if 'output_fps' not in st.session_state:
               st.session_state['output_fps'] = None
           if 'examples_idx' not in st.session_state:
               st.session_state['examples_idx'] = None

           if st.session_state['selected_refine'] == 'Add New Video' or \
                   len(os.listdir(os.path.join(self.project_dir, self.iter_folder, st.session_state['selected_refine']))) < 3:
               st.session_state['disabled'] = False
               [st.session_state['outlier_method'],
                st.session_state['p_cutoff'], st.session_state['num_outliers'], st.session_state['output_fps'],
                st.session_state['min_n_seconds'],
                frame_dir, shortvid_dir] = prompt_setup(self.software, self.ftype, self.selected_bodyparts, self.annotation_classes,
                                                        outlier_methods, self.threshold, self.duration_min, self.framerate,
                                                        self.videos_dir, self.project_dir, self.iter_folder)
               try:
                   save_data(os.path.join(self.project_dir, self.iter_folder), shortvid_dir,
                             'refine_params.sav',
                             [st.session_state['outlier_method'],
                              st.session_state['p_cutoff'],
                              st.session_state['min_n_seconds'],
                              st.session_state['num_outliers'],
                              st.session_state['output_fps'],
                              ])
               except:
                   pass

               if st.session_state['video'] is not None and st.session_state['pose'] is not None:
                   if os.path.exists(os.path.join(self.videos_dir, st.session_state['video'].name)):
                       temporary_location = str(os.path.join(self.videos_dir, st.session_state['video'].name))
                   else:
                       g = io.BytesIO(st.session_state['video'].read())  # BytesIO Object
                       temporary_location = str(os.path.join(self.videos_dir, st.session_state['video'].name))
                       with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                           out.write(g.read())  # Read bytes into file
                       out.close()

                   if os.path.exists(frame_dir):
                       framedir_ = os.listdir(frame_dir)
                       if len(framedir_) < 2:
                           frame_extraction(video_file=temporary_location, frame_dir=frame_dir)
                       else:

                           if os.path.exists(shortvid_dir):
                               viddir_ = os.listdir(shortvid_dir)
                               if len(viddir_) < 3:
                                   frames2integ = round(float(self.framerate) * (self.duration_min / 0.1))
                                   [iterX_model, _, _] = load_iterX(self.project_dir, self.iter_folder)

                                   st.session_state['features'], st.session_state['predict'], \
                                       st.session_state['examples_idx'] = \
                                       create_videos(
                                           st.session_state['pose'],
                                           self.selected_bodyparts,
                                           self.llh_value,
                                           iterX_model,
                                           self.framerate,
                                           frames2integ,
                                           st.session_state['outlier_method'],
                                           st.session_state['p_cutoff'],
                                           st.session_state['num_outliers'],
                                           st.session_state['output_fps'],
                                           st.session_state['min_n_seconds'],
                                           self.annotation_classes,
                                           is_3d=self.is_3d,
                                           software=self.software,
                                           multi_animal=self.multi_animal,
                                           frame_dir=frame_dir,
                                           shortvid_dir=shortvid_dir)
                                   st.session_state['video_path'] = \
                                       os.path.join(self.videos_dir,
                                                    str.join('', (selected_refine_dir.rpartition('_refine_vids')[0],
                                                                  '.mp4')))
                                   save_data(os.path.join(self.project_dir, self.iter_folder), shortvid_dir,
                                             'refinements.sav',
                                             [st.session_state['video_path'],
                                              st.session_state['features'],
                                              st.session_state['predict'],
                                              st.session_state['examples_idx'],
                                              st.session_state['refined']
                                              ])
           else:

               if 'curr_vid' not in st.session_state:
                   st.session_state['curr_vid'] = None
               # if switch vid
               if selected_refine_dir != st.session_state['curr_vid']:
                   [st.session_state['outlier_method'],
                    st.session_state['p_cutoff'],
                    st.session_state['min_n_seconds'],
                    st.session_state['num_outliers'],
                    st.session_state['output_fps'],
                    ] = load_refine_params(os.path.join(self.project_dir, self.iter_folder),
                                           selected_refine_dir)

                   [st.session_state['video_path'],
                    st.session_state['features'],
                    st.session_state['predict'],
                    st.session_state['examples_idx'],
                    st.session_state['refined']] = load_refinement(os.path.join(self.project_dir, self.iter_folder),
                                                                   selected_refine_dir)

                   [_, shortvid_dir] = \
                       prompt_setup_existing(outlier_methods, self.framerate,
                                             self.videos_dir, self.project_dir, self.iter_folder,
                                             selected_refine_dir)
                   st.session_state['curr_vid'] = selected_refine_dir
                   set_def_index = 0
               else:
                   [st.session_state['outlier_method'],
                    st.session_state['p_cutoff'],
                    st.session_state['min_n_seconds'],
                    st.session_state['num_outliers'],
                    st.session_state['output_fps'],
                    ] = load_refine_params(os.path.join(self.project_dir, self.iter_folder),
                                           selected_refine_dir)
                   [_, shortvid_dir] = \
                       prompt_setup_existing(outlier_methods, self.framerate,
                                             self.videos_dir, self.project_dir, self.iter_folder,
                                             st.session_state['curr_vid'])
                   set_def_index = 0
               st.session_state['video_path'] = \
                   os.path.join(self.videos_dir, str.join('', (selected_refine_dir.rpartition('_refine_vids')[0], '.mp4')))
               st.session_state['disabled'] = True
               behav_choice = st.radio("Select the behavior: ", self.annotation_classes_ex,
                                       index=int(0), horizontal=True,
                                       key="behavior_choice"
                                       , help=REFINEMENT_HELP)
               st.info('Make sure you :orange[Save/Update Refinements] before moving to another behavior! '
                       'Or else it will clear your modification.')

               existing_outliers = [d for d in os.listdir(shortvid_dir)
                                    if d.endswith('.mp4') and behav_choice in d]

               if len(existing_outliers) > 0:
                   alltabs = st.tabs([f'{i}' for i in range(len(existing_outliers))])
               else:
                   alltabs = None

               st.write('')
               st.write('')

               col_option, col_option2, col_msg = st.columns([1, 1, 1])
               save_button = col_option.button('Save/Update Refinements', key='save_ref')
               clear_vid_button = col_msg.button(':red[Delete Video and Choice]', key='clear_vid')

               if clear_vid_button:
                   try:
                       for file_name in glob.glob(shortvid_dir + "/*"):
                           os.remove(file_name)
                       st.session_state['examples_idx'] = None
                       st.session_state['refined'] = {key: {k: None for k in range(st.session_state['num_outliers'])}
                                                      for key in self.annotation_classes_ex}
                       col_msg.info('Cleared. Type "R" to refresh.')
                       st.session_state['disabled'] = False
                   except:
                       pass

               else:
                   if alltabs is not None:
                       for i, tab_ in enumerate(alltabs):
                           with tab_:
                               colL, colR = st.columns([2, 1.5])
                               colL.video(os.path.join(shortvid_dir,
                                                       f'behavior_{behav_choice}_example_{i}.mp4'))
                               with colR:
                                   tooltip_exp = st.expander(":green[How to use the data editor:]")
                                   with tooltip_exp:
                                       st.write()

                                   selected_set = st.radio('Select Refinement Set',
                                                           ('Default Filled', 'Previously Saved'),
                                                           horizontal=True, index=set_def_index,
                                                           key=f'ref_set_{i}'
                                                           , help=DATAEDITOR_HELP)

                                   if selected_set == 'Default Filled':
                                       time_ = np.arange(0,

                                                         (st.session_state['examples_idx'][behav_choice][i][1] -
                                                          st.session_state['examples_idx'][behav_choice][i][0]) / 10,
                                                         self.duration_min)
                                       data_df = pd.DataFrame(
                                           {
                                               "Time (s)": [t for t in time_],
                                               "Behavior": [behav_choice
                                                            for _ in range(len(time_))]
                                               ,
                                           }
                                       )
                                       edited_df = st.data_editor(
                                           data_df,
                                           column_config={
                                               "Behavior": st.column_config.SelectboxColumn(
                                                   "Behavior Category",
                                                   # help="The category of the app",
                                                   width="medium",
                                                   options=self.annotation_classes_ex,
                                               )
                                           },
                                           key=f'{i}',
                                           hide_index=True,
                                       )
                                       st.session_state['refined'][behav_choice][i] = edited_df
                                   elif selected_set == 'Previously Saved':
                                       # st.write(st.session_state['curr_vid'])
                                       [st.session_state['video_path'],
                                        st.session_state['features'],
                                        st.session_state['predict'],
                                        st.session_state['examples_idx'],
                                        st.session_state['refined']] = load_refinement(
                                           os.path.join(self.project_dir, self.iter_folder),
                                           st.session_state['curr_vid'])
                                       data_df = st.session_state['refined'][behav_choice][i]
                                       st.dataframe(data_df)

                                   # edited_df = st.data_editor(
                                   #     data_df,
                                   #     column_config={
                                   #         "Behavior": st.column_config.SelectboxColumn(
                                   #             "Behavior Category",
                                   #             # help="The category of the app",
                                   #             width="medium",
                                   #             options=annotation_classes_ex,
                                   #         )
                                   #     },
                                   #     key=f'{i}',
                                   #     hide_index=True,
                                   # )
                                   # st.session_state['refined'][behav_choice][i] = edited_df

                   else:
                       st.warning('no video'.upper())
                   if save_button:
                       save_data(os.path.join(self.project_dir, self.iter_folder), selected_refine_dir, 'refinements.sav',
                                 [st.session_state['video_path'],
                                  st.session_state['features'],
                                  st.session_state['predict'],
                                  st.session_state['examples_idx'],
                                  st.session_state['refined']
                                  ])