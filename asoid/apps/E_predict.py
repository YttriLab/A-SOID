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
from utils.view_results import Viewer
from sklearn.preprocessing import LabelEncoder
from utils.import_data import load_labels_auto
import datetime


TITLE = "Predict behaviors"


def pie_predict(predict_npy, iter_folder, annotation_classes, placeholder):
    plot_col_top = placeholder.empty()
    option_expander = placeholder.expander("Configure Plot")
    behavior_colors = {k: [] for k in annotation_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())

    if len(annotation_classes) == 4:
        default_colors = ["red", "darkorange", "dodgerblue", "gray"]
    else:
        np.random.seed(42)
        selected_idx = np.random.choice(np.arange(len(all_c_options)), len(annotation_classes), replace=False)
        default_colors = [all_c_options[s] for s in selected_idx]

    for i, class_name in enumerate(annotation_classes):
        behavior_colors[class_name] = option_expander.selectbox(f'Color for {class_name}',
                                                                all_c_options,
                                                                index=all_c_options.index(default_colors[i]),
                                                                key=f'color_option{i}')
    predict = np.load(predict_npy, allow_pickle=True)
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
        st.plotly_chart(fig, use_container_width=True)

    return behavior_colors


def ethogram_plot(predict_npy, iter_folder, annotation_classes, exclude_other,
                  behavior_colors, framerate, placeholder2):
    plot_col_top = placeholder2.empty()
    predict = np.load(predict_npy, allow_pickle=True)
    annotation_classes_ex = annotation_classes.copy()
    colors_classes = list(behavior_colors.values()).copy()

    if exclude_other:
        annotation_classes_ex.pop(annotation_classes_ex.index('other'))
        colors_classes.pop(annotation_classes.index('other'))
    # st.write(colors_classes)
    prefill_array = np.zeros((len(predict),
                              len(annotation_classes_ex)))
    default_colors_wht = ['black']
    # st.write(colors_classes)
    default_colors_wht.extend(colors_classes)
    # st.write(default_colors_wht)
    # default_colors_wht.extend(['black'])

    css_cmap = [mcolors.CSS4_COLORS[default_colors_wht[j]] for j in range(len(default_colors_wht))]
    # st.write(css_cmap)
    count = 0
    for b in range(len(annotation_classes_ex)):
        # print(b)
        idx_b = np.where(predict == b)[0]
        prefill_array[idx_b, count] = b + 1
        count += 1

    seed_num = placeholder2.number_input('seed for segment',
                                         min_value=0, max_value=None, value=42,
                                         key=f'rand_seed')
    # np.random.seed(seed_num)
    length_ = placeholder2.slider('number of frames',
                                  min_value=25, max_value=len(predict),
                                  value=int(len(predict) / 20),
                                  key=f'length_slider')

    if placeholder2.checkbox('use randomized time',
                             value=False,
                             key=f'randtime_ckbx'):
        np.random.seed(seed_num)
        rand_start = np.random.choice(prefill_array.shape[0] - length_, 1, replace=False)
        unique_id = np.unique(prefill_array[int(rand_start):int(rand_start + length_), :])
        le = LabelEncoder()
        relabeled_1d = le.fit_transform(prefill_array[int(rand_start):int(rand_start + length_), :].ravel())
        relabeled_2d = relabeled_1d.reshape(int(rand_start + length_) - int(rand_start), -1)
        fig = go.Figure(data=go.Heatmap(z=relabeled_2d.T,
                                        y=annotation_classes_ex,
                                        colorscale=[css_cmap[int(i)] for i in unique_id],
                                        showscale=False
                                        ))
        fig.update_layout(
            xaxis=dict(
                title='Time (s)',
                tickmode='array',
                tickvals=np.arange(0, length_ + 1, (length_ + 1)/5),
                ticktext=np.round(np.arange(np.round(rand_start / framerate, 1),
                                   np.round((rand_start + length_ + 1) / framerate, 1),
                                   np.round(((rand_start + length_ + 1) / framerate - rand_start / framerate)/5, 1)), 1)
            )
        )
        fig['layout']['yaxis']['autorange'] = "reversed"
    else:
        rand_start = 0
        unique_id = np.unique(prefill_array[int(rand_start):int(rand_start + length_), :])
        le = LabelEncoder()
        relabeled_1d = le.fit_transform(prefill_array[int(rand_start):int(rand_start + length_), :].ravel())
        relabeled_2d = relabeled_1d.reshape(int(rand_start + length_) - int(rand_start), -1)
        fig = go.Figure(data=go.Heatmap(z=relabeled_2d.T,
                                        y=annotation_classes_ex,
                                        colorscale=[css_cmap[int(i)] for i in unique_id],
                                        showscale=False,

                                        ))

        fig.update_layout(
            xaxis=dict(
                title='Time (s)',
                tickmode='array',
                tickvals=np.arange(0, length_ + 1, (length_ + 1)/5),
                ticktext=np.round(np.arange(0,
                           np.round(length_ + 1 / framerate, 1),
                           np.round(((length_ + 1) / framerate) / 5, 1)), 1)
            )
        )
        fig['layout']['yaxis']['autorange'] = "reversed"

    plot_col_top.plotly_chart(fig, use_container_width=True)


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
                            frame_dir, videos_dir, iter_folder, video_checkbox):
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
    video_type = str.join('', ('.', st.session_state['video'].type.rpartition('video/')[2]))
    video_prefix = st.session_state['video'].name.rpartition(video_type)[0]
    annotated_vid_str = str.join('', ('_annotated_', iter_folder))
    annotated_vid_name = str.join('', (video_prefix, annotated_vid_str, video_type))
    st.session_state['new_annotated_vid_path'][iter_folder] = os.path.join(videos_dir, annotated_vid_name)
    if video_checkbox:
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
    colL, colR = st.columns(2)
    video_checkbox = colR.checkbox("Create Labeled Video?")
    action_button = colL.button('Predict and Generate Summary Statistics')
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

        for i in stqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
                pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
                predict_arr = np.array(predict).flatten()

        with st.spinner('Matching video frames'):
            predictions_match = create_annotated_videos(predict_arr, pred_proba[0],
                                                        framerate, frames2integ, annotation_classes,
                                                        frame_dir, videos_dir, iter_folder, video_checkbox)
            st.balloons()
            message_box.success('Done. Type "R" to refresh.')
        np.save(st.session_state['new_annotated_vid_path'][iter_folder].replace('mp4', 'npy'),
                predictions_match)
    return features[0], predict_arr, predictions_match


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
    dummy_df = pd.get_dummies(df["classes"])
    # add 0 columns for each class that wasn't predicted in the file
    not_predicted_classes = [x for x in annotation_classes if x not in np.unique(df["classes"].values)]
    for not_predicted_class in not_predicted_classes:
        dummy_df[not_predicted_class] = 0

    dummy_df["time"] = time_clm
    dummy_df = dummy_df.set_index("time")

    # save to csv
    dummy_df.to_csv(source_file_name)


def convert_dummies_to_labels(labels, annotation_classes):
    """
    This function converts dummy variables to labels
    :param labels: pandas dataframe with dummy variables
    :return: pandas dataframe with labels and codes
    """
    conv_labels = pd.from_dummies(labels)
    cat_df = pd.DataFrame(conv_labels.values, columns=["labels"])
    if annotation_classes is not None:
        cat_df["labels"] = pd.Categorical(cat_df["labels"] , ordered=True, categories=annotation_classes)
    else:
        cat_df["labels"] = pd.Categorical(cat_df["labels"], ordered=True, categories=cat_df["labels"].unique())
    cat_df["codes"] = cat_df["labels"].cat.codes

    return cat_df


def prep_labels_single(labels, annotation_classes):
    """
    This function loads the labels from a single file and prepares them for plotting
    :param labels: pandas dataframe with labels
    :return: pandas dataframe with labels
    """
    labels = labels.drop(columns=["time"], errors="ignore")
    labels = convert_dummies_to_labels(labels, annotation_classes)

    return labels


def count_events(df_label):
    """ This function counts the number of events for each label in a dataframe"""
    df_label_cp = df_label.copy()
    # prepare event counter
    event_counter = pd.DataFrame(df_label_cp["labels"].unique(), columns=["labels"])
    event_counter["events"] = 0

    # Count the number of isolated blocks of labels for each unique label
    # go through each unique label and create a binary column
    for label in df_label_cp["labels"].unique():
        df_label_cp[label] = (df_label_cp["labels"] == label)
        df_label_cp[label].iloc[df_label_cp[label] == False] = np.NaN
        # go through each unique label and count the number of isolated blocks
        df_label_cp[f"{label}_block"] = np.where(df_label_cp[label].notnull(),
                                                 (df_label_cp[label].notnull() & (df_label_cp[label] != df_label_cp[
                                                     label].shift())).cumsum(),
                                                 np.nan)
        event_counter["events"].iloc[event_counter["labels"] == label] = df_label_cp[f"{label}_block"].max()

    return event_counter


def describe_labels_single(df_label, framerate, placeholder):
    """ This function describes the labels in a table"""

    event_counter = count_events(df_label)

    count_df = df_label.value_counts().to_frame().reset_index().rename(columns={0: "frame count"})
    # heatmap already shows this information
    #count_df["percentage"] = count_df["frame count"] / count_df["frame count"].sum() *100

    if framerate is not None:
        count_df["total duration"] = count_df["frame count"] / framerate
        count_df["total duration"] = count_df["total duration"].apply(lambda x: str(datetime.timedelta(seconds=x)))

    count_df["bouts"] = event_counter["events"]

    count_df.set_index("codes", inplace=True)
    count_df.sort_index(inplace=True)
    #rename all columns to include their units
    count_df.rename(columns={"bouts": "bouts [-]",
                             "frame count": "frame count [-]",
                             "total duration": "total duration [hh:mm:ss]",
                             "percentage": "percentage [%]",

                             },
                    inplace=True)
    #TODO: autosize columns with newer streamlit versions (e.g., using use_container_width=True)
    placeholder.dataframe(count_df, hide_index=True)


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
        exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
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
        annotated_vids = [d for d in os.listdir(videos_dir)
                          if d.endswith(str.join('', (iter_folder, '.npy')))]
        annotated_vids.extend(['Add New Video'])

        selected_annot_video = ri.radio('Select Video Prediction',
                                        annotated_vids, horizontal=True)
        if selected_annot_video != 'Add New Video':
            annot_vid_path = os.path.join(videos_dir, selected_annot_video.replace('npy', 'mp4'))
        else:
            annot_vid_path = 'Add New Video'

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

            st.session_state['disabled'] = False

            if annot_vid_path == 'Add New Video':
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
                            [iterX_model, _, _] = load_iterX(project_dir, iter_folder)
                            st.info(f'loaded {iter_folder} model')
                            st.session_state['new_features'][iter_folder], \
                                st.session_state['new_predict'][iter_folder], \
                                st.session_state['new_predict_match'][iter_folder] = \
                                predict_annotate_video(iterX_model, framerate, frames2integ,
                                                       annotation_classes,
                                                       frame_dir, videos_dir, iter_folder)
            else:
                video_col, summary_col = st.columns([2, 1.5])
                # display behavioral pie chart
                behavior_colors = pie_predict(annot_vid_path.replace('mp4', 'npy'),
                                              iter_folder,
                                              annotation_classes,
                                              summary_col)
                ethogram_plot(annot_vid_path.replace('mp4', 'npy'),
                              iter_folder,
                              annotation_classes,
                              exclude_other,
                              behavior_colors,
                              framerate,
                              video_col,

                              )
                if not os.path.isfile(annot_vid_path.replace('mp4', 'csv')):
                    save_predictions(annot_vid_path.replace('mp4', 'npy'),
                                     annot_vid_path.replace('mp4', 'csv'),
                                     annotation_classes,
                                     framerate)
                    st.info(f'Saved ethogram csv as {annot_vid_path.replace("mp4", "csv")}. Type "R" to view in app.')

                else:
                    labels = load_labels_auto(annot_vid_path.replace('mp4', 'csv'),
                                              origin="BORIS", fps=framerate)
                    single_label = prep_labels_single(labels, annotation_classes)
                    describe_labels_single(single_label, framerate, summary_col)
                # display video from video path
                if os.path.isfile(annot_vid_path):
                    video_col.video(annot_vid_path)

    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
