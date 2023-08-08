import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from config.help_messages import *
from config.help_messages import NO_CONFIG_HELP, IMPRESS_TEXT
# import time
from utils.load_workspace import load_new_pose

TITLE = "Predict behaviors"



def disable():
    st.session_state["disabled"] = True


def prompt_setup(software, ftype, annotation_classes, framerate, videos_dir, project_dir, iter_dir):
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
                                           header=[0, 1, 2], sep=",", index_col=0
                                           )
                idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
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


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:
        # st.warning("If you did not do it yet, remove and reupload the config file to make sure that you use the latest configuration!")
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
        selected_iter = ri.selectbox('Select Iteration #', np.arange(iteration + 1), iteration)
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)
        videos_dir = os.path.join(project_dir, 'videos')
        os.makedirs(videos_dir, exist_ok=True)

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
            if 'features' not in st.session_state:
                st.session_state['features'] = None
            if 'predict' not in st.session_state:
                st.session_state['predict'] = None

            st.session_state['disabled'] = False

            frame_dir = prompt_setup(software, ftype, annotation_classes, framerate,
                                     videos_dir, project_dir, iter_folder)
            st.write(st.session_state)

            # TODO: extract features from st.session_state.uplaoded_pose and
            # TODO: create annotated video overlaying st.session_state['uploaded_vid'] (after making a copy)



    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
