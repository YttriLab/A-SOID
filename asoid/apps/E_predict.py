import io
import os
import re
from pathlib import Path

import cv2
import ffmpeg
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config.help_messages import *
from config.help_messages import NO_CONFIG_HELP, IMPRESS_TEXT
from stqdm import stqdm
from utils.extract_features import feature_extraction, \
    bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale
from utils.load_workspace import load_new_pose, load_iterX

TITLE = "Predict behaviors"


def pie_predict(predict, iter_folder, annotation_classes, placeholder):
    behavior_classes = np.arange(len(annotation_classes))
    # plot_col = placeholder.columns([, 1])
    # option_col.write('')
    # option_col.write('')
    # option_col.write('')
    # option_col.write('')
    # option_col.write('')
    # option_col.write('')
    plot_col_top = placeholder.empty()
    option_expander = placeholder.expander("Configure Plot")
    behavior_colors = {k: [] for k in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())

    if len(annotation_classes) == 4:
        default_colors = ["red", "darkorange", "dodgerblue", "gray"]
    else:
        np.random.seed(42)
        selected_idx = np.random.choice(np.arange(len(all_c_options)), len(behavior_classes), replace=False)
        default_colors = [all_c_options[s] for s in selected_idx]

    for i, class_id in enumerate(behavior_classes):
        behavior_colors[class_id] = option_expander.selectbox(f'Color for {annotation_classes[i]}',
                                                              all_c_options,
                                                              index=all_c_options.index(default_colors[i]),
                                                              key=f'color_option{i}')

    # behavior_classes = st.session_state['classifier'].classes_

    # predict = []
    # TODO: find a color workaround if a class is missing
    # for f in range(len(st.session_state['features'][condition])):
    #     predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    predict_dict = {'iteration': np.repeat(iter_folder, len(np.hstack(predict))),
                    'behavior': np.hstack(predict)}
    df_raw = pd.DataFrame(data=predict_dict)
    labels = df_raw['behavior'].value_counts(sort=False).index
    values = df_raw['behavior'].value_counts(sort=False).values
    # summary dataframe
    df = pd.DataFrame()
    # do i need this?
    behavior_labels = []
    for l in labels:
        behavior_labels.append(behavior_classes[int(l)])
    df["values"] = values
    df['labels'] = behavior_labels
    df["colors"] = df["labels"].apply(lambda x:
                                      behavior_colors.get(x))  # to connect Column value to Color in Dict
    with plot_col_top:
        fig = go.Figure(
            data=[go.Pie(labels=[annotation_classes[int(i)] for i in df["labels"]], values=df["values"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='value',
                          textfont_size=16,
                          marker=dict(colors=df["colors"],
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True)


def disable():
    st.session_state["disabled"] = True


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


def prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                 framerate, videos_dir, project_dir, iter_dir):
    left_col, right_col = st.columns(2)
    le_exapnder = left_col.expander('video'.upper(), expanded=True)
    ri_exapnder = right_col.expander('pose'.upper(), expanded=True)
    frame_dir = None
    shortvid_dir = None
    if software == 'CALMS21 (PAPER)':
        ROOT = Path(__file__).parent.parent.parent.resolve()
        new_pose_sav = os.path.join(ROOT.joinpath("new_test"), './new_pose.sav')
        new_pose_list = load_new_pose(new_pose_sav)
    else:
        # try:
        new_videos = le_exapnder.file_uploader('Upload Video Files',
                                               accept_multiple_files=False,
                                               type=['avi', 'mp4'], key='video')
        new_pose_csvs = [ri_exapnder.file_uploader('Upload Corresponding Pose Files',
                                                   accept_multiple_files=False,
                                                   type=ftype, key='pose')]
        if new_videos is not None and new_pose_csvs[0] is not None:
            st.session_state['uploaded_vid'] = new_videos
            new_pose_list = []
            for i, f in enumerate(new_pose_csvs):
                current_pose = pd.read_csv(f,
                                           header=[0, 1, 2], sep=",", index_col=0)
                bp_level = 1
                bp_index_list = []
                for bp in selected_bodyparts:
                    bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
                    bp_index_list.append(bp_index)

                selected_pose_idx = np.sort(np.array(bp_index_list).flatten())

                # get rid of likelihood columns for deeplabcut
                idx_llh = selected_pose_idx[2::3]
                # the loaded sleap file has them too, so exclude for both
                idx_selected = [i for i in selected_pose_idx if i not in idx_llh]
                #
                # idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
                new_pose_list.append(np.array(current_pose.iloc[:, idx_selected]))

            st.session_state['uploaded_pose'] = new_pose_list
            # col1, col3 = st.columns(2)

            col3_exp = st.expander('Output folders'.upper(), expanded=True)
            frame_dir = col3_exp.text_input('Enter a directory for frames',
                                            os.path.join(videos_dir,
                                                         str.join('', (
                                                             st.session_state['uploaded_vid'].name.rpartition('.mp4')[
                                                                 0],
                                                             '_pngs'))),
                                            disabled=st.session_state.disabled, on_change=disable
                                            )

            try:
                os.listdir(frame_dir)
                col3_exp.success(f'Entered **{frame_dir}** as the frame directory.')
            except FileNotFoundError:
                if col3_exp.button('create frame directory'):
                    os.makedirs(frame_dir, exist_ok=True)
                    col3_exp.info('Created. Type "R" to refresh.')
                    # st.experimental_rerun()

        else:
            st.session_state['uploaded_pose'] = []
            st.session_state['uploaded_vid'] = None

        # except:
        #     pass
    return frame_dir


def convert_int(s):
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def create_annotated_videos(prediction, prob,
                            framerate, frames2integ, annotation_classes,
                            frame_dir, videos_dir, iter_folder):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.25
    fontColor = (0, 0, 255)
    thickness_text = 2
    lineType = 2
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    bottomLeftCornerOfText = (25, height - 50)
    repeat_n = int(frames2integ / 10)
    predictions_match = np.pad(prediction.repeat(repeat_n), (repeat_n, 0), 'edge')[:len(images)]

    new_images = []
    for j in stqdm(range(len(images)), desc=f'Annotating {st.session_state["video"].name}'):
        image = images[j]
        rgb_im = cv2.imread(os.path.join(frame_dir, image))
        cv2.putText(rgb_im, f'{annotation_classes[int(predictions_match[j])]}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness_text,
                    lineType)
        new_images.append(rgb_im)
    video_type = str.join('', ('.', st.session_state['video'].type.rpartition('video/')[2]))
    video_prefix = st.session_state['video'].name.rpartition(video_type)[0]
    annotated_vid_str = str.join('', ('_annotated_', iter_folder))
    annotated_vid_name = str.join('', (video_prefix, annotated_vid_str, video_type))
    st.session_state['new_annotated_vid_path'][iter_folder] = os.path.join(videos_dir, annotated_vid_name)

    video = cv2.VideoWriter(st.session_state['new_annotated_vid_path'][iter_folder],
                            fourcc, framerate, (width, height))
    for j, image in enumerate(new_images):
        video.write(image)
    cv2.destroyAllWindows()
    video.release()
    return predictions_match


def predict_annotate_video(iterX_model, framerate, frames2integ,
                           annotation_classes,
                           frame_dir, videos_dir, iter_folder):
    features = [None]
    predict_arr = None
    predictions_match = None
    action_button = st.button("Create Labeled Video")
    message_box = st.empty()
    processed_input_data = st.session_state['uploaded_pose']
    if action_button:
        st.session_state['disabled'] = True
        message_box.info('Predicting labels... ')
        # extract features, bin them
        features = []
        for i, data in enumerate(processed_input_data):
            # using feature scaling from training set
            feats, _ = feature_extraction([data], 1, frames2integ)
            features.append(feats)
            # st.write(features)

        for i in stqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
                pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
                predict_arr = np.array(predict).flatten()
        # try:
        with st.spinner('Creating annotated video'):
            predictions_match = create_annotated_videos(predict_arr, pred_proba[0],
                                                        framerate, frames2integ, annotation_classes,
                                                        frame_dir, videos_dir, iter_folder)
            st.balloons()
            message_box.success('Done. Type "R" to refresh.')
        # except:
        #     st.info('Terminated early. Type "R" to refresh.')

    return features[0], predict_arr, predictions_match


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:
        # st.warning("If you did not do it yet, remove and reupload the config file to make sure that you use the latest configuration!")
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        ftype = config["Project"].get("FILE_TYPE")
        selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        # threshold = config["Processing"].getfloat("SCORE_THRESHOLD")
        threshold = 0.1
        iteration = config["Processing"].getint("ITERATION")
        framerate = config["Project"].getint("FRAMERATE")
        duration_min = config["Processing"].getfloat("MIN_DURATION")
        selected_iter = ri.selectbox('Select Iteration #', np.arange(iteration + 1), iteration)
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)
        videos_dir = os.path.join(project_dir, 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        frames2integ = round(float(framerate) * (duration_min / 0.1))
        # st.write(st.session_state)
        if software == 'CALMS21 (PAPER)':
            ROOT = Path(__file__).parent.parent.parent.resolve()
            targets_test_csv = os.path.join(ROOT.joinpath("test"), './test_labels.csv')
            targets_test_df = pd.read_csv(targets_test_csv, header=0)
            targets_test = np.array(targets_test_df['annotation'])
        else:
            if 'disabled' not in st.session_state:
                st.session_state['disabled'] = False
            if 'uploaded_pose' not in st.session_state:
                st.session_state['uploaded_pose'] = []
            #
            if 'video_path' not in st.session_state:
                st.session_state['video_path'] = None
            if 'new_features' not in st.session_state:
                st.session_state['new_features'] = None
            if 'new_predict' not in st.session_state:
                st.session_state['new_predict'] = None
            if 'new_predict_match' not in st.session_state:
                st.session_state['new_predict_match'] = None
            if 'new_annotated_vid_path' not in st.session_state:
                st.session_state['new_annotated_vid_path'] = None

            if st.session_state['new_features'] is None:
                st.session_state['new_features'] = {str.join('', ('iteration-', str(i))): None
                                                    for i in np.arange(iteration + 1)}
                st.session_state['new_predict'] = {str.join('', ('iteration-', str(i))): None
                                                   for i in np.arange(iteration + 1)}
                st.session_state['new_predict_match'] = {str.join('', ('iteration-', str(i))): None
                                                         for i in np.arange(iteration + 1)}
                st.session_state['new_annotated_vid_path'] = {str.join('', ('iteration-', str(i))): None
                                                              for i in np.arange(iteration + 1)}
            if iter_folder not in st.session_state['new_annotated_vid_path']:
                st.session_state['new_features'][iter_folder] = None
                st.session_state['new_predict'][iter_folder] = None
                st.session_state['new_predict_match'][iter_folder] = None
                st.session_state['new_annotated_vid_path'][iter_folder] = None
            st.session_state['disabled'] = False
            # st.session_state

            frame_dir = prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                                     framerate, videos_dir, project_dir, iter_folder)

            if st.session_state['video'] is not None and len(st.session_state['uploaded_pose']) > 0:
                if os.path.exists(os.path.join(videos_dir, st.session_state['video'].name)):
                    temporary_location = str(os.path.join(videos_dir, st.session_state['video'].name))
                else:
                    g = io.BytesIO(st.session_state['video'].read())  # BytesIO Object
                    temporary_location = str(os.path.join(videos_dir, st.session_state['video'].name))
                    with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                        out.write(g.read())  # Read bytes into file
                    out.close()

                if os.path.exists(frame_dir):
                    framedir_ = os.listdir(frame_dir)
                    if len(framedir_) < 2:
                        frame_extraction(video_file=temporary_location, frame_dir=frame_dir)
                    else:
                        if st.session_state['new_annotated_vid_path'][iter_folder] is None:

                            [iterX_model, _, _, _, _, _] = load_iterX(project_dir, iter_folder)
                            st.info(f'loaded {iter_folder} model')

                            st.session_state['new_features'][iter_folder], \
                                st.session_state['new_predict'][iter_folder], \
                                st.session_state['new_predict_match'][iter_folder] = \
                                predict_annotate_video(iterX_model, framerate, frames2integ,
                                                       annotation_classes,
                                                       frame_dir, videos_dir, iter_folder)
                        else:
                            # st.write('hi')
                            # st.write(st.session_state['predict'][80:100],
                            #          st.session_state['predict_match'][240:300])
                            # st.write(np.repeat(st.session_state['predict'][80:100], 6))
                            placeholder = st.empty()
                            video_col, summary_col = st.columns([2, 1.5])
                            pie_predict(st.session_state['new_predict_match'][iter_folder],
                                        iter_folder,
                                        annotation_classes,
                                        summary_col)
                            # st.write(np.unique(st.session_state['predict'][iter_folder], return_counts=True))
                            video_col.video(st.session_state['new_annotated_vid_path'][iter_folder])







    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
