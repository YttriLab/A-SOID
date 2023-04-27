import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import joblib
import streamlit as st
import io
import ffmpeg
# from app import swap_app
import cv2
from stqdm import stqdm
import random
import re
import categories
import base64
from moviepy.editor import VideoFileClip
from utils.load_workspace import load_features, load_test_targets, \
    load_iterX, load_new_pose, load_predict_proba
from utils.manual_active_learning import Refine
from utils.load_workspace import load_iterX, load_features
from utils.extract_features import feature_extraction, feature_extraction_with_extr_scaler, \
    frameshift_predict, frameshift_predict_proba, bsoid_predict_numba, bsoid_predict_numba_noscale
from utils.import_data import load_pose, get_bodyparts, get_animals
from config.help_messages import *

from config.help_messages import NO_CONFIG_HELP, IMPRESS_TEXT

CATEGORY = categories.REFINE
TITLE = "Refine behaviors"


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


def convert_int(s):
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def create_labeled_vid(labels, counts, frames2integ,
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

    for b in np.unique(labels):
        with st.spinner(f'generating videos for behavior {annotation_classes[int(b)]}'):
            idx_b = np.where(labels == b)[0]
            try:
                examples_b = np.random.choice(idx_b, counts, replace=False)
            except:
                examples_b = np.random.choice(idx_b, len(idx_b), replace=False)

            for ex, example_b in enumerate(stqdm(examples_b, desc="creating videos")):
                video_name = 'behavior_{}_example_{}.mp4'.format(annotation_classes[int(b)], int(ex))
                grp_images = []

                for f in range(number_of_frames):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                    # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)
                    grp_images.append(rgb_im)

                for f in range(number_of_frames, int(2 * number_of_frames)):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))

                    # Draw a circle with blue line borders of thickness of 2 px
                    rgb_im = cv2.circle(rgb_im, center_coordinates, radius, color, thickness_circle)

                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)

                    grp_images.append(rgb_im)

                for f in range(int(2 * number_of_frames), int(3 * number_of_frames)):
                    rgb_im = cv2.imread(os.path.join(frame_dir, images[(example_b - 1) * number_of_frames + f]))
                    # bgr = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
                    cv2.putText(rgb_im, f'{round(output_fps / framerate, 2)}X',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness_text,
                                lineType)
                    grp_images.append(rgb_im)

                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
                for j, image in enumerate(grp_images):
                    video.write(image)
                cv2.destroyAllWindows()
                video.release()
                videoClip = VideoFileClip(os.path.join(output_path, video_name))
                vid_prefix = video_name.rpartition('.mp4')[0]
                gif_name = f"{vid_prefix}.gif"
                videoClip.write_gif(os.path.join(output_path, gif_name))
    return


def create_videos(processed_input_data, scalar, iterX_model, framerate, frames2integ,
                  num_outliers, output_fps, annotation_classes,
                  frame_dir, shortvid_dir):
    if st.button("Predict labels and create example videos"):
        st.info('Predicting labels... ')
        features = []
        scaled_features = []
        # extract features, bin them
        for i, data in enumerate(processed_input_data):
            # using feature scaling from training set
            feats, scaled_feats = feature_extraction_with_extr_scaler([data]
                                                                      , 1
                                                                      , frames2integ
                                                                      , scalar
                                                                      )
            features.append(feats)
            scaled_features.append(scaled_feats)
        for i in stqdm(range(len(scaled_features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([scaled_features[i]], iterX_model)
                predict_arr = np.array(predict).flatten()
        create_labeled_vid(predict_arr, num_outliers, frames2integ,
                           framerate, output_fps, annotation_classes,
                           frame_dir, shortvid_dir)
        st.balloons()


def prompt_setup(software, ftype, threshold, framerate, working_dir, prefix):
    left_col, right_col = st.columns(2)
    left_expand = left_col.expander('Select a video file:', expanded=True)
    right_expand = right_col.expander('Select the corresponding pose file:', expanded=True)
    p_cutoff = None
    num_outliers = None
    output_fps = None
    frame_dir = None
    shortvid_dir = None
    new_pose_list = None
    if software == 'CALMS21 (PAPER)':
        ROOT = Path(__file__).parent.parent.parent.resolve()
        new_pose_sav = os.path.join(ROOT.joinpath("new_test"), './new_pose.sav')
        new_pose_list = load_new_pose(new_pose_sav)
        new_videos = None
    else:
        new_videos = left_expand.file_uploader('Upload video files',
                                               accept_multiple_files=False,
                                               type=['avi', 'mp4'], key='video')
        new_pose_csvs = [right_expand.file_uploader('Upload corresponding pose csv files',
                                                    accept_multiple_files=False,
                                                    type=ftype, key='pose')]
        try:
            new_pose_list = []
            for i, f in enumerate(new_pose_csvs):
                current_pose = pd.read_csv(new_pose_csvs[i],
                                           header=[0, 1, 2], sep=",", index_col=0)
                idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
                new_pose_list.append(np.array(current_pose.iloc[:, idx_selected]))
            col1, col3 = st.columns(2)
            col1_exp = col1.expander('Parameters'.upper(), expanded=True)
            # col2_exp = col2.expander('Samples to refine'.upper(), expanded=True)
            col3_exp = col3.expander('Output folders'.upper(), expanded=True)
            p_cutoff = col1_exp.number_input('Threshold value to sample outliers from',
                                             min_value=0.0, max_value=1.0, value=threshold)
            num_outliers = col1_exp.number_input('Number of potential outliers to refine',
                                                 min_value=10, max_value=None, value=20)
            output_fps = col1_exp.number_input('Video playback fps',
                                               min_value=1, max_value=None, value=5)
            col1_exp.write(f'equivalent to {round(output_fps / framerate, 2)} X speed')
            frame_dir = col3_exp.text_input('Enter a directory for frames',
                                            os.path.join(working_dir, prefix, new_videos.name.rpartition('.mp4')[0],
                                                         'pngs'),
                                            )

            try:
                os.listdir(frame_dir)
                col3_exp.success(f'Entered **{frame_dir}** as the frame directory.')
            except FileNotFoundError:
                if col3_exp.button('create frame directory'):
                    os.makedirs(frame_dir, exist_ok=True)
                    st.experimental_rerun()

            shortvid_dir = col3_exp.text_input('Enter a directory for refined videos',
                                               os.path.join(working_dir, prefix, new_videos.name.rpartition('.mp4')[0],
                                                            'refine_vids'),
                                               )
            try:
                os.listdir(shortvid_dir)
                col3_exp.success(f'Entered **{shortvid_dir}** as the refined video directory.')
            except FileNotFoundError:
                if col3_exp.button('create refined video directory'):
                    os.makedirs(shortvid_dir, exist_ok=True)
                    st.experimental_rerun()
        except:
            pass
    return new_videos, new_pose_list, p_cutoff, num_outliers, output_fps, frame_dir, shortvid_dir


def main(config=None):
    st.markdown("""---""")
    if config is not None:
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        ftype = config["Project"].get("FILE_TYPE")
        # threshold = config["Processing"].getfloat("SCORE_THRESHOLD")
        threshold = 0.1
        iteration = config["Processing"].getint("ITERATION")
        framerate = config["Project"].getint("FRAMERATE")
        duration_min = config["Processing"].getfloat("MIN_DURATION")



        if software == 'CALMS21 (PAPER)':
            ROOT = Path(__file__).parent.parent.parent.resolve()
            targets_test_csv = os.path.join(ROOT.joinpath("test"), './test_labels.csv')
            targets_test_df = pd.read_csv(targets_test_csv, header=0)
            targets_test = np.array(targets_test_df['annotation'])
        else:
            [new_videos, new_pose_list, p_cutoff, num_outliers, output_fps, frame_dir, shortvid_dir] = \
                prompt_setup(software, ftype, threshold, framerate, working_dir, prefix)
            if 'refinements' not in st.session_state:
                st.session_state['refinements'] = {key:
                                                       {k: {'choice': None, 'submitted': False}
                                                        for k in range(num_outliers)}
                                                   for key in annotation_classes}
            # st.write(st.session_state['refinements'])
            # for b_chosen in list(user_choices.keys()):
            #     user_choices[b_chosen] = {key: [] for key in range(num_outliers)}
            if new_videos is not None and len(new_pose_list) > 0:
                if os.path.exists(new_videos.name):
                    temporary_location = f'{new_videos.name}'
                else:
                    g = io.BytesIO(new_videos.read())  # BytesIO Object
                    temporary_location = f'{new_videos.name}'
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
                            if len(viddir_) < 2:
                                frames2integ = round(float(framerate) * (duration_min / 0.1))
                                [_, _, scalar, _] = load_features(working_dir, prefix)
                                [iterX_model, _, _, _, _, _] = load_iterX(working_dir, prefix)
                                create_videos(new_pose_list, scalar, iterX_model, framerate, frames2integ,
                                              num_outliers, output_fps, annotation_classes,
                                              frame_dir=frame_dir, shortvid_dir=shortvid_dir)

                            else:
                                col_option, col_msg = st.columns(2)
                                # col_msg.success('refinement candidates have been saved!')
                                if col_option.checkbox('Redo? Uncheck after check to prevent from auto-clearing',
                                                       False,
                                                       key='vr'):
                                    try:
                                        for file_name in glob.glob(shortvid_dir + "/*"):
                                            os.remove(file_name)
                                    except:
                                        pass

                                behav_choice = st.selectbox("Select the behavior: ", annotation_classes,
                                                            index=int(0),
                                                            key="behavior_choice")
                                checkbox_autofill = st.checkbox('autofill')
                                alltabs = st.tabs([f'{i}' for i in range(num_outliers)])

                                # if not st.session_state['refinements'][behav_choice]:
                                # st.write(st.session_state['refinements'])

                                for i, tab_ in enumerate(alltabs):
                                    with tab_:
                                        colL, colR = st.columns([3, 1])
                                        file_ = open(
                                            os.path.join(shortvid_dir, f'behavior_{behav_choice}_example_{i}.gif'),
                                            "rb")
                                        contents = file_.read()
                                        data_url = base64.b64encode(contents).decode("utf-8")
                                        file_.close()
                                        colL.markdown(
                                            f'<img src="data:image/gif;base64,{data_url}" alt="gif">',
                                            unsafe_allow_html=True,
                                        )
                                        # st.write([annotation_classes[i] for i in range(len(annotation_classes))],
                                        #          'hello')
                                        with colR.form(key=f'form_{i}'):
                                            returned_choice = st.radio("Select the correct class: ",
                                                                         annotation_classes,
                                                                         index=annotation_classes.index(behav_choice),
                                                                         key="radio_{}".format(i))

                                            # st.session_state['refinements'][behav_choice][i]["submitted"] = \
                                            #     st.form_submit_button("Submit",
                                            #                           "Press to confirm your choice")
                                            if st.form_submit_button("Submit",
                                                                      "Press to confirm your choice"):
                                                # st.write('hello')
                                                st.session_state['refinements'][behav_choice][i]["submitted"] = True
                                                st.session_state['refinements'][behav_choice][i][
                                                    "choice"] = returned_choice
                                            # else:
                                            #     st.experimental_rerun()

                                            # if st.session_state['refinements'][behav_choice][i]["submitted"] == True:

                                            if checkbox_autofill:
                                                if st.session_state['refinements'][behav_choice][i]["submitted"] == False:
                                                    st.session_state['refinements'][behav_choice][i][
                                                        "choice"] = behav_choice
                                                    # st.experimental_rerun()
                                            else:
                                                if st.session_state['refinements'][behav_choice][i]["submitted"] == False:
                                                    st.session_state['refinements'][behav_choice][i][
                                                        "choice"] = None
                                                    # st.experimental_rerun()
                                        st.write(st.session_state['refinements'])

                                        # np.
                                        # try:
                                        #     if returned_choice == 'other':
                                        #         new_behav = colR.text_input('input name of behavior', )
                                        #     st.write(new_behav)
                                        # except:
                                        #     pass
                                        # st.session_state['refinements'][behav_choice][i] = returned_choice
                                        # user_choices[behav_choice][i] = returned_choice
                                        # st.write(st.session_state['refinements'])
                                            # st.write(user_choices)



    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        # button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        # if button_col1.button('◀  PRIOR STEP'):
        #     swap_app('C-auto-active-learning')
        # if button_col5.button('NEXT STEP ▶'):
        #     swap_app('E-predict')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
