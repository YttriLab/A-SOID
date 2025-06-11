import datetime
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

from sklearn.preprocessing import LabelEncoder
from stqdm import stqdm
from asoid.utils.extract_features_2D import feature_extraction
from asoid.utils.extract_features_3D import feature_extraction_3d
from asoid.utils.predict import bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale, weighted_smoothing
from asoid.utils.load_workspace import load_new_pose, load_iterX
from asoid.utils.preprocessing import adp_filt, sort_nicely
from asoid.utils.import_data import load_pose
from asoid.config.help_messages import *

from asoid.utils.reporting import extract_descriptors, prep_labels_single

TITLE = "Predict behaviors"
PREDICT_HELP = ("In this step, you can predict the behavior in sessions (pose) using the trained model from :orange[Active Learning]."
                "\n\n The predictions will be used to generate summary statistics and visualizations."
                "\n\n **Optional:** You can also generate annotated videos with the predicted labels."
                "\n\n---\n\n"
                "**Step 1**: Select an iteration*."
                "\n\n **Step 2**: Select a session (pose) to visualize results or add new data."
                "\n\n **Step 3**: Predict the behavior."
                "\n\n **Step 4**: Generate summary statistics and visualizations."
                "\n\n:grey[* This refers to the iterations of refinement, followed by active learning.]"
                "\n\n---\n\n"
                ":red[This step requires several GUI refreshs of the app (press 'R' on your keyboard when indicated by the] :green[green info box] :red[).]\n\n"
                ":blue[The predictions will be saved in the project directory for your further analysis.]")


def pie_predict(predict_npy, iter_folder, annotation_classes, placeholder, top_most_container, vidname):
    plot_col_top = placeholder.empty()
    behavior_colors = {k: [] for k in annotation_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)), len(annotation_classes), replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    option_expander = top_most_container.expander("Configure Plot")
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_name in enumerate(annotation_classes):
        if i % 4 == 0:
            behavior_colors[class_name] = col1.selectbox(f'select color for {class_name}',
                                                         all_c_options,
                                                         index=all_c_options.index(default_colors[i]),
                                                         key=f'color_option{i}'
                                                         )

        elif i % 4 == 1:
            behavior_colors[class_name] = col2.selectbox(f'select color for {class_name}',
                                                         all_c_options,
                                                         index=all_c_options.index(default_colors[i]),
                                                         key=f'color_option{i}'
                                                         )
        elif i % 4 == 2:
            behavior_colors[class_name] = col3.selectbox(f'select color for {class_name}',
                                                         all_c_options,
                                                         index=all_c_options.index(default_colors[i]),
                                                         key=f'color_option{i}'
                                                         )
        elif i % 4 == 3:
            behavior_colors[class_name] = col4.selectbox(f'select color for {class_name}',
                                                         all_c_options,
                                                         index=all_c_options.index(default_colors[i]),
                                                         key=f'color_option{i}'
                                                         )

    predict = np.load(predict_npy, allow_pickle=True)
    predict_dict = {'iteration': np.repeat(iter_folder, len(np.hstack(predict))),
                    'behavior': np.hstack(predict)}
    df_raw = pd.DataFrame(data=predict_dict)
    labels = df_raw['behavior'].value_counts(sort=False).index
    values = df_raw['behavior'].value_counts(sort=False).values
    # summary dataframe
    df = pd.DataFrame()
    # I need this just in case there is a missing behavior
    behavior_labels = []
    for l in labels:
        behavior_labels.append(annotation_classes[int(l)])
    df["values"] = values
    df['labels'] = behavior_labels
    df["colors"] = df["labels"].apply(lambda x:
                                      behavior_colors.get(x))  # to connect Column value to Color in Dict
    with plot_col_top:
        fig = go.Figure(
            data=[go.Pie(labels=df["labels"], values=df["values"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='value',
                          textfont_size=16,
                          marker=dict(colors=df["colors"],
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': f"{str.join('', (vidname, '_', iter_folder, '_duration_pie'))}",
                'height': 600,
                'width': 600,
                'scale': 3  # Multiply title/legend/axis/canvas sizes by this factor
            }
        })

    return behavior_colors


def ethogram_plot(predict_npy, iter_folder, annotation_classes, exclude_other,
                  behavior_colors, framerate, placeholder2, vidname):
    plot_col_top = placeholder2.empty()
    predict = np.load(predict_npy, allow_pickle=True)
    annotation_classes_ex = annotation_classes.copy()
    colors_classes = list(behavior_colors.values()).copy()
    if exclude_other:
        annotation_classes_ex.pop(annotation_classes_ex.index('other'))
        colors_classes.pop(annotation_classes.index('other'))
    prefill_array = np.zeros((len(predict),
                              len(annotation_classes_ex)))
    default_colors_wht = ['black']
    default_colors_wht.extend(colors_classes)
    css_cmap = [mcolors.CSS4_COLORS[default_colors_wht[j]] for j in range(len(default_colors_wht))]
    count = 0
    for b in range(len(annotation_classes_ex)):
        idx_b = np.where(predict == b)[0]
        prefill_array[idx_b, count] = b + 1
        count += 1
    unique_id = np.unique(prefill_array[:, :])
    le = LabelEncoder()
    relabeled_1d = le.fit_transform(prefill_array[:, :].ravel())
    relabeled_2d = relabeled_1d.reshape(prefill_array.shape[0], -1)
    fig = go.Figure(data=go.Heatmap(z=relabeled_2d.T,
                                    y=annotation_classes_ex,
                                    colorscale=[css_cmap[int(i)] for i in unique_id],
                                    showscale=False,

                                    ))

    fig.update_layout(
        xaxis=dict(
            title='Frame Number',
            tickmode='array',
            # tickvals=[*range(0, prefill_array.shape[0] + 1, framerate)]
            # tickvals=np.arange(0, prefill_array.shape[0] + 1, (prefill_array.shape[0] + 1) / 20),
            # ticktext=np.round(np.arange(0,
            #                             np.round(prefill_array.shape[0] + 1 / framerate, 1),
            #                             np.round(((prefill_array.shape[0] + 1) / framerate) / 20, 1)), 1)
        )
    )
    fig['layout']['yaxis']['autorange'] = "reversed"

    plot_col_top.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png',  # one of png, svg, jpeg, webp
            'filename': f"{str.join('', (vidname, '_', iter_folder, '_ethogram'))}",
            'height': 720,
            'width': 1280,
            'scale': 3  # Multiply title/legend/axis/canvas sizes by this factor
        }
    })


def get_duration_bouts(predict, behavior_classes, framerate):
    behav_durations = []
    bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    bout_durations = np.hstack([np.diff(bout_start_idx), len(predict) - np.max(bout_start_idx)])
    bout_start_label = predict[bout_start_idx]
    for b in behavior_classes:
        idx_b = np.where(bout_start_label == int(b))[0]
        if len(idx_b) > 0:
            behav_durations.append(bout_durations[idx_b] / framerate)
        else:
            a = np.empty((1,))
            a[:] = np.nan
            behav_durations.append(a)
    return behav_durations


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def ridge_predict(predict_npy, iter_folder, annotation_classes, exclude_other,
                  behavior_colors, framerate,
                  placeholder, vidname):
    predict = np.load(predict_npy, allow_pickle=True)
    annotation_classes_ex = annotation_classes.copy()
    colors_classes = list(behavior_colors.values()).copy()

    if exclude_other:
        annotation_classes_ex.pop(annotation_classes_ex.index('other'))
        colors_classes.pop(annotation_classes.index('other'))
    behavior_classes = np.arange(len(annotation_classes_ex))

    with placeholder:

        duration_ = get_duration_bouts(predict, behavior_classes, framerate)
        css_cmap = [mcolors.CSS4_COLORS[j] for j in colors_classes]
        duration_matrix = boolean_indexing(duration_)
        fig = go.Figure()
        for data_line, color, name in zip(duration_matrix, css_cmap, annotation_classes_ex):
            fig.add_trace(go.Box(x=data_line[(data_line <= np.nanpercentile(data_line, 99)) &
                                             (data_line >= np.nanpercentile(data_line, 0))],
                                 jitter=0.5,
                                 whiskerwidth=0.5,
                                 fillcolor=color,
                                 marker_size=2,
                                 line_width=1.5,
                                 line_color='#EEEEEE',
                                 name=name), )
        fig.update_traces(orientation='h')
        fig.update_layout(xaxis=dict(title='bout duration (seconds)'),
                          xaxis_range=[0, np.nanpercentile(np.array(duration_matrix), 99)],
                          )
        fig['layout']['yaxis']['autorange'] = "reversed"

        st.plotly_chart(fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': f"{str.join('', (vidname, '_', iter_folder, '_bout_durations'))}",
                'height': 720,
                'width': 1280,
                'scale': 3  # Multiply title/legend/axis/canvas sizes by this factor
            }
        })
        # st.plotly_chart(fig, use_container_width=True)


def disable():
    st.session_state["disabled"] = True


def frame_extraction(video_file, frame_dir, placeholder=None):
    if placeholder is None:
        placeholder = st.empty()
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(
        int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(
            video_info['avg_frame_rate'].rpartition('/')[2]))
    if placeholder.button('Start frame extraction for {} frames '
                          'at {} frames per second'.format(num_frames, avg_frame_rate)):
        placeholder.info('Extracting frames from the video... ')
        # if frame_dir
        try:
            (ffmpeg.input(video_file)
             .filter('fps', fps=avg_frame_rate)
             .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                     s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))),
                     sws_flags='bilinear', start_number=0)
             .run(capture_stdout=True, capture_stderr=True))
            placeholder.info(
                'Done extracting **{}** frames from video **{}**.'.format(num_frames, video_file))
        except ffmpeg.Error as e:
            placeholder.error('stdout:', e.stdout.decode('utf8'))
            placeholder.error('stderr:', e.stderr.decode('utf8'))
        placeholder.info('Done extracting {} frames from {}'.format(num_frames, video_file))
        placeholder.success('Done. Press "R" on your keyboard to refresh.')
        st.balloons()
        # TODO: Find suitable replacement for manual refresh
        # st.rerun()

def prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                 framerate, videos_dir, project_dir, iter_dir):
    left_col, right_col = st.columns([3, 1])
    pose_expander = left_col.expander('pose'.upper(), expanded=True)
    param_expander = right_col.expander('smoothing size'.upper(), expanded=True)
    # ri_exapnder = right_col.expander('video'.upper(), expanded=True)
    frame_dir = None
    shortvid_dir = None
    if software == 'CALMS21 (PAPER)':
        ROOT = Path(__file__).parent.parent.parent.resolve()
        new_pose_sav = os.path.join(ROOT.joinpath("new_test"), './new_pose.sav')
        new_pose_list = load_new_pose(new_pose_sav)
    else:
        new_pose_csvs = pose_expander.file_uploader('Upload Corresponding Pose Files',
                                                   accept_multiple_files=True,
                                                   type=ftype, key='pose')
        st.session_state['smooth_size'] = param_expander.number_input('Minimum frames per behavior',
                                                                      min_value=0, max_value=None, value=12)


def create_annotated_videos(vidpath_out,
                            framerate, annotation_classes,
                            frame_dir,
                            video_checkbox, predictions_match):
    if video_checkbox:
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

        new_images = []
        if len(predictions_match) != len(images):
            st.error('Number of frames in video and pose file do not match.')
            if len(predictions_match) < len(images):
                st.warning(f"Cutting of video frames ({len(images)})to match labels ({len(predictions_match)})...")
                len_vid = len(predictions_match)
            elif len(predictions_match) > len(images):
                st.warning(f"Cutting of labels ({len(predictions_match)}) to match video frames ({len(images)})...")
                len_vid = len(images)
        else:
            len_vid = len(images)
        for j in stqdm(range(len_vid), desc=f'Annotating {st.session_state["video"].name}'):
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

        video = cv2.VideoWriter(vidpath_out,
                                fourcc, framerate, (width, height))
        for j, image in enumerate(new_images):
            video.write(image)
        cv2.destroyAllWindows()
        video.release()


def predict_annotate_video(ftype, software, is_3d, multi_animal, selected_bodyparts, llh_value, iterX_model, framerate, frames2integ,
                           annotation_classes,
                           frame_dir, videos_dir, iter_folder,
                           video_checkbox, colL):
    features = [None]
    predict_arr = None
    predictions_match = None
    action_button = colL.button('Predict and Generate Summary Statistics')
    message_box = st.empty()
    new_pose_csvs = st.session_state['pose']
    repeat_n = int(frames2integ / 10)
    total_n_frames = []

    if action_button:
        st.session_state['disabled'] = True
        message_box.info('Predicting labels... ')
        # extract features, bin them
        features = []
        # for i, data in enumerate(processed_input_data):
        for i, f in enumerate(stqdm(new_pose_csvs, desc="Extracting spatiotemporal features from pose")):
        # for i, f in enumerate(new_pose_csvs):

            current_pose = load_pose(f, software, multi_animal)

            bp_level = 1
            bp_index_list = []

            if i == 0:
                st.info("Selected keypoints/bodyparts (from config): " + ", ".join(selected_bodyparts))
                st.info("Available keypoints/bodyparts in pose file: " + ", ".join(
                    current_pose.columns.get_level_values(bp_level).unique()))
                # check if all bodyparts are in the pose file

                if len(selected_bodyparts) > len(current_pose.columns.get_level_values(bp_level).unique()):
                    st.error(f'Not all selected keypoints/bodyparts are in the pose file: {f.name}')
                    st.stop()
                elif len(selected_bodyparts) < len(current_pose.columns.get_level_values(bp_level).unique()):
                    # subselection would take care of this, so we need to make sure that they all exist
                    for bp in selected_bodyparts:
                        if bp not in current_pose.columns.get_level_values(bp_level).unique():
                            st.error(f'At least one keypoint "{bp}" is missing in pose file: {f.name}')
                            st.stop()
         
                for bp in selected_bodyparts:
                    bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
                    bp_index_list.append(bp_index)
                selected_pose_idx = np.sort(np.array(bp_index_list).flatten())

                # get likelihood column idx directly from dataframe columns
                # idx_llh = [i for i, s in enumerate(current_pose.columns) if "likelihood" in s]
                idx_llh = [i for i, s in enumerate(current_pose.columns) if "likelihood" in s and s in current_pose.columns[selected_pose_idx]]

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

            total_n_frames.append(filt_pose.shape[0])
            features.append(feats)
        for i in stqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
                pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
                predict_arr = np.array(predict).flatten()

            predictions_raw = np.pad(predict_arr.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames[i]]
            predictions_match = weighted_smoothing(predictions_raw, size=st.session_state['smooth_size'])
            curr_file_name = st.session_state['pose'][i].name
            curr_ftype = curr_file_name.split('.')[-1]
            pose_prefix = curr_file_name.rpartition(str.join('', ('.', curr_ftype)))[0]
            annotated_str = str.join('', ('_annotated_', iter_folder))
            annotated_vid_name = str.join('', (pose_prefix, annotated_str, '.mp4'))

            vidpath_out = os.path.join(videos_dir, annotated_vid_name)

            with st.spinner('Matching video frames'):
                create_annotated_videos(vidpath_out,
                                        framerate, annotation_classes,
                                        frame_dir,
                                        video_checkbox, predictions_match)

                message_box.success('Done. Press "R" on your keyboard to refresh.')
                st.balloons()
                #TODO: Find suitable replacement for manual refresh
                # st.rerun()
            np.save(vidpath_out.replace('mp4', 'npy'),
                    predictions_match)
        st.balloons()

def annotate_video(video_path, framerate, annotation_classes,
                   frame_dir, videos_dir, iter_folder,
                   predictions_match):
    # video_type = str.join('', ('.', video_path.rpartition('.mp4')[2]))

    video_prefix = video_path.replace('.mp4', '')
    annotated_vid_str = str.join('', ('_annotated_', iter_folder))
    annotated_vid_name = str.join('', (video_prefix, annotated_vid_str, '.mp4'))
    vidpath_out = os.path.join(annotated_vid_name)

    # st.write(video_type, video_prefix, annotated_vid_str, annotated_vid_name, vidpath_out)
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
    new_images = []

    if len(predictions_match) != len(images):
        st.warning('Number of frames in video and pose file do not match.')
        if len(predictions_match) < len(images):
            st.warning(f"Cutting of video frames ({len(images)})to match labels ({len(predictions_match)})...")
            len_vid = len(predictions_match)
        elif len(predictions_match) > len(images):
            st.warning(f"Cutting of labels ({len(predictions_match)}) to match video frames ({len(images)})...")
            len_vid = len(images)
    else:
        len_vid = len(images)

    for j in stqdm(range(len_vid), desc=f'Annotating {annotated_vid_name}'):
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

    video = cv2.VideoWriter(vidpath_out,
                            fourcc, framerate, (width, height))
    for j, image in enumerate(new_images):
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def just_annotate_video(predict_npy, framerate,
                        annotation_classes,
                        frame_dir, videos_dir, iter_folder,
                        colL):
    video_path = str.join('', (predict_npy.replace('npy', 'mp4').rpartition('_annotated_')[0], '.mp4'))
    action_button = colL.button('Create Labeled Video')
    if action_button:
        predict = np.load(predict_npy, allow_pickle=True)
        with colL:
            with st.spinner('Matching video frames'):
                annotate_video(video_path, framerate, annotation_classes,
                               frame_dir, videos_dir, iter_folder,
                               predict)
                st.balloons()
                st.success('Done. Press "R" on your keyboard to refresh.')
                #TODO: Find suitable replacement for manual refresh
                #st.rerun()

def save_predictions(predict_npy, source_file_name, annotation_classes, framerate):
    """takes numerical labels and transforms back into one-hot encoded file (BORIS style). Saves as csv"""
    predict = np.load(predict_npy, allow_pickle=True)

    df = pd.DataFrame(predict, columns=["labels"])
    time_clm = np.round(np.arange(0, df.shape[0]) / framerate, 2)
    # convert numbers into behavior names
    class_dict = {i: x for i, x in enumerate(annotation_classes)}
    df["classes"] = df["labels"].copy()
    for cl_idx, cl_name in class_dict.items():
        df["classes"].iloc[df["labels"] == cl_idx] = cl_name

    # for simplicity let's convert this back into BORIS type file
    dummy_df = pd.get_dummies(df["classes"]).astype(int)
    # add 0 columns for each class that wasn't predicted in the file
    not_predicted_classes = [x for x in annotation_classes if x not in np.unique(df["classes"].values)]
    for not_predicted_class in not_predicted_classes:
        dummy_df[not_predicted_class] = 0

    dummy_df["time"] = time_clm
    dummy_df = dummy_df.set_index("time")
    if not os.path.isfile(source_file_name):
        # save to csv
        dummy_df.to_csv(source_file_name)
        st.info(f'Saved ethogram csv as {source_file_name}.')
    return dummy_df


def describe_labels_single(df_label, annotation_classes, framerate, placeholder):
    """ This function describes the labels in a table and visualizes them in streamlit"""
    count_df = extract_descriptors(df_label, annotation_classes, framerate)
    placeholder.dataframe(count_df, hide_index=True, use_container_width=True)


def main(ri=None, config=None):
    st.markdown("""---""")

    st.title("Predict new Data")
    st.expander("What is this?", expanded=False).markdown(PREDICT_HELP)


    if config is not None:
        # st.warning("If you did not do it yet, remove and reupload the config file to make sure that you use the latest configuration!")
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        ftype = [x.strip() for x in config["Project"].get("FILE_TYPE").split(",")]
        selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
        is_3d = config["Project"].getboolean("IS_3D")
        multi_animal = config["Project"].getboolean("MULTI_ANIMAL")
        # threshold = config["Processing"].getfloat("SCORE_THRESHOLD")
        threshold = 0.1
        llh_value = config["Processing"].getfloat("LLH_VALUE")
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
        annotated_vids = [d for d in os.listdir(videos_dir)
                          if d.endswith(str.join('', (iter_folder, '.npy')))]
        sort_nicely(annotated_vids)
        annotated_vids_trim = [annotated_vids[i].rpartition('_annotated_')[0]
                               for i in range(len(annotated_vids))]
        annotated_vids_trim.extend(['Add New Data'])
        annotated_vids.extend(['Add New Data'])

        selection = ri.selectbox('Select Data to for prediction', annotated_vids_trim
                                 , help = "Select a pose file/video to visualize and or add new data")
        selected_annot_video = annotated_vids[annotated_vids_trim.index(selection)]

        if selected_annot_video != 'Add New Data':
            annot_vid_path = os.path.join(videos_dir, selected_annot_video.replace('npy', 'mp4'))
        else:
            annot_vid_path = 'Add New Data'

        if software == 'CALMS21 (PAPER)':
            st.error("The CALMS21 data set is not designed to be used with the predict step.")
            st.stop()
        else:
            if 'disabled' not in st.session_state:
                st.session_state['disabled'] = False
            # if 'uploaded_pose' not in st.session_state:
            #     st.session_state['uploaded_pose'] = []
            if 'smooth_size' not in st.session_state:
                st.session_state['smooth_size'] = None

            st.session_state['disabled'] = False

            if annot_vid_path == 'Add New Data':

                try:
                    # st.write(project_dir, iter_folder)
                    [iterX_model, _, _] = load_iterX(project_dir, iter_folder)
                    ri.info(f'loaded {iter_folder} model')
                    prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                                 framerate, videos_dir, project_dir, iter_folder)
                    if st.session_state['pose'] is not None:
                        placeholder = st.empty()
                        
                        predict_annotate_video(ftype, software, is_3d, multi_animal, selected_bodyparts, llh_value,
                                               iterX_model, framerate, frames2integ,
                                               annotation_classes,
                                               None, videos_dir, iter_folder,
                                               None, placeholder)
                except FileNotFoundError:
                    st.error(f'Please train a {iter_folder} model in :orange[Active Learning] step.')

            else:
                top_most_container = st.container()
                video_col, summary_col = st.columns([2, 1.5])
                # display behavioral pie chart
                behavior_colors = pie_predict(annot_vid_path.replace('mp4', 'npy'),
                                              iter_folder,
                                              annotation_classes,
                                              summary_col,
                                              top_most_container,
                                              selection)
                ethogram_plot(annot_vid_path.replace('mp4', 'npy'),
                              iter_folder,
                              annotation_classes,
                              exclude_other,
                              behavior_colors,
                              framerate,
                              video_col,
                              selection)
                annotation_classes_ex = annotation_classes.copy()
                if exclude_other:
                    annotation_classes_ex.pop(annotation_classes_ex.index('other'))
                labels = save_predictions(annot_vid_path.replace('mp4', 'npy'),
                                          annot_vid_path.replace('mp4', 'csv'),
                                          annotation_classes_ex,
                                          framerate)
                single_label = prep_labels_single(labels, annotation_classes_ex)
                describe_labels_single(single_label, annotation_classes_ex, framerate, summary_col)
                ridge_predict(annot_vid_path.replace('mp4', 'npy'),
                              iter_folder,
                              annotation_classes,
                              exclude_other,
                              behavior_colors,
                              framerate,
                              summary_col,
                              selection)
                # display video from video path
                if os.path.isfile(annot_vid_path):
                    # video_col.write(f"### Video: {annot_vid_path}")
                    video_col.video(annot_vid_path)
                else:
                    video_checkbox = video_col.checkbox("Create Labeled Video?")
                    if video_checkbox:
                        frame_dir = str.join('', (annot_vid_path.rpartition('_annotated_')[0], '_pngs'))
                        os.makedirs(frame_dir, exist_ok=True)
                        video_expander = video_col.expander('VIDEO', expanded=True)
                        video_expander.file_uploader(f'Upload Corresponding Video: '
                                                     f'{selected_annot_video.rpartition("_annotated_")[0]}',
                                                     accept_multiple_files=False,
                                                     type=['avi', 'mp4'], key='video')
                        if os.path.exists(frame_dir):
                            framedir_ = os.listdir(frame_dir)
                            if len(framedir_) < 2:
                                if st.session_state['video']:
                                    if os.path.exists(os.path.join(videos_dir, st.session_state['video'].name)):
                                        temporary_location = str(
                                            os.path.join(videos_dir, st.session_state['video'].name))
                                    else:
                                        g = io.BytesIO(st.session_state['video'].read())  # BytesIO Object
                                        temporary_location = str(
                                            os.path.join(videos_dir, st.session_state['video'].name))
                                        with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                                            out.write(g.read())  # Read bytes into file
                                        out.close()
                                    frame_extraction(temporary_location, frame_dir, video_col)
                            else:
                                just_annotate_video(annot_vid_path.replace('mp4', 'npy'),
                                                    framerate,
                                                    annotation_classes,
                                                    frame_dir, videos_dir, iter_folder,
                                                    video_col)


    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
