import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import joblib
import streamlit as st
from app import swap_app
import categories
from utils.load_workspace import load_features, load_test_targets, \
    load_iterX, load_new_pose, load_predict_proba
from utils.manual_active_learning import Refine

from config.help_messages import NO_CONFIG_HELP, IMPRESS_TEXT

CATEGORY = categories.REFINE
TITLE = "Refine Behaviors"


def prompt_setup(software, ftype, threshold):
    left_col, right_col = st.columns(2)
    left_expand = left_col.expander('Select your new video files:', expanded=True)
    right_expand = right_col.expander('Select your corresponding new pose files:', expanded=True)
    if software == 'CALMS21 (PAPER)':
        ROOT = Path(__file__).parent.parent.parent.resolve()
        new_pose_sav = os.path.join(ROOT.joinpath("new_test"), './new_pose.sav')
        new_pose_list = load_new_pose(new_pose_sav)
        new_videos = None
    else:
        new_videos = left_expand.file_uploader('Upload video files',
                                               accept_multiple_files=True,
                                               type=['avi', 'mp4'], key='video')
        new_pose_csvs = right_expand.file_uploader('Upload corresponding pose csv files',
                                                   accept_multiple_files=True,
                                                   type=ftype, key='pose')
        new_pose_list = []
        for i, f in enumerate(new_pose_csvs):
            current_pose = pd.read_csv(new_pose_csvs[i],
                                       header=[0, 1, 2], sep=",", index_col=0)
            idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
            new_pose_list.append(np.array(current_pose.iloc[:, idx_selected]))

    col1, col2, col3 = st.columns(3)
    col1_exp = col1.expander('Confidence threshold'.upper(), expanded=True)
    col2_exp = col2.expander('Samples to refine'.upper(), expanded=True)
    col3_exp = col3.expander('Store refined labels'.upper(), expanded=True)
    p_cutoff = col1_exp.number_input('Threshold value to sample outliers from',
                                     min_value=0.0, max_value=1.0, value=threshold)
    num_outliers = col2_exp.number_input('Number of potential outliers to refine',
                                         min_value=10, max_value=None, value=20)
    label_filename = col3_exp.text_input('Filename to store refined labels',
                                         value='refined_behaviors')
    label_filename = str.join('', (label_filename, '.csv'))
    return new_videos, new_pose_list, p_cutoff, num_outliers, label_filename


def main(config=None):
    st.markdown("""---""")
    if config is not None:

        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        ftype = config["Project"].get("FILE_TYPE")
        threshold = config["Processing"].getfloat("SCORE_THRESHOLD")
        iteration = config["Processing"].getint("ITERATION")

        if software == 'CALMS21 (PAPER)':
            ROOT = Path(__file__).parent.parent.parent.resolve()
            targets_test_csv = os.path.join(ROOT.joinpath("test"), './test_labels.csv')
            targets_test_df = pd.read_csv(targets_test_csv, header=0)
            targets_test = np.array(targets_test_df['annotation'])
        else:
            targets_test = np.hstack(load_test_targets(working_dir, prefix))
        [iterX_model, _, _, iterX_f1_scores, _, _] = load_iterX(working_dir, prefix)
        files = []
        ref_files = []
        for file in glob.glob(str.join('', (os.path.join(working_dir, prefix), '/*predict_proba.sav'))):
            files.append(file)
        for ref_file in glob.glob(str.join('', (os.path.join(working_dir, prefix), '/*refinements.sav'))):
            ref_files.append(ref_file)
        if len(files) > 0:
            selection_container = st.container()
            left_selection, right_selection = selection_container.columns(2)
            if st.checkbox('Add new files?'):
                [new_videos, new_pose_list,
                 p_cutoff, num_outliers, label_filename] = prompt_setup(software, ftype, threshold)
                if st.button('Predict new behavior'):
                    [_, _, scalar, frames2integ] = load_features(working_dir, prefix)
                    predict, proba, outlier_indices, label_df = None, None, None, None
                    refinement = Refine(working_dir, prefix, software, annotation_classes, frames2integ,
                                        scalar, targets_test,
                                        predict, proba, outlier_indices,
                                        iterX_model, iterX_f1_scores,
                                        new_videos, new_pose_list, p_cutoff, num_outliers, config,
                                        label_filename, label_df, iteration)
                    refinement.predict_behavior_proba()
                    refinement.subsample_outliers()
                    refinement.save_predict_proba()
            selected_pred = left_selection.selectbox('Select the previously loaded predictions',
                                                     [os.path.basename(files[i])
                                                      for i in range(len(files))])
            selected_refinements = right_selection.selectbox('Select the refinement file',
                                                             [os.path.basename(ref_files[i])
                                                              for i in range(len(ref_files))])
            with open(os.path.join(working_dir, prefix, selected_pred), 'rb') as fr:
                [new_videos, new_pose_list, p_cutoff, num_outliers,
                 predict, proba] = joblib.load(fr)
            with open(os.path.join(working_dir, prefix, selected_refinements), 'rb') as fr:
                [outlier_indices, label_filename] = joblib.load(fr)
            try:
                label_df = pd.read_csv(os.path.join(working_dir, prefix, label_filename))
                [_, _, scalar, frames2integ] = load_features(working_dir, prefix)
                refinement = Refine(working_dir, prefix, software, annotation_classes, frames2integ,
                                    scalar, targets_test,
                                    predict, proba, outlier_indices,
                                    iterX_model, iterX_f1_scores,
                                    new_videos, new_pose_list, p_cutoff, num_outliers, config,
                                    label_filename, label_df, iteration)
                refinement.add_labels()
            except:
                label_df = None
                pass
            if "active_learning_running" not in st.session_state:
                st.session_state["active_learning_running"] = False
            if st.button('Refine behaviors', key="start_active_learning") or \
                    st.session_state["active_learning_running"]:
                if not st.session_state["active_learning_running"]:
                    # this way, we avoid the use of checkboxes
                    st.session_state["active_learning_running"] = True
                [_, _, scalar, frames2integ] = load_features(working_dir, prefix)
                refinement = Refine(working_dir, prefix, software, annotation_classes, frames2integ,
                                    scalar, targets_test,
                                    predict, proba, outlier_indices,
                                    iterX_model, iterX_f1_scores,
                                    new_videos, new_pose_list, p_cutoff, num_outliers, config,
                                    label_filename, label_df, iteration)
                refinement.refine_outliers()
                refinement.info_box()
        else:
            [new_videos, new_pose_list,
             p_cutoff, num_outliers, label_filename] = prompt_setup(software, ftype, threshold)
            if st.button('Predict new behavior'):
                [_, _, scalar, frames2integ] = load_features(working_dir, prefix)
                predict, proba, outlier_indices, label_df = None, None, None, None
                refinement = Refine(working_dir, prefix, software, annotation_classes, frames2integ,
                                    scalar, targets_test,
                                    predict, proba, outlier_indices,
                                    iterX_model, iterX_f1_scores,
                                    new_videos, new_pose_list, p_cutoff, num_outliers, config,
                                    label_filename, label_df, iteration)
                refinement.predict_behavior_proba()
                refinement.subsample_outliers()
                refinement.save_predict_proba()
                st.experimental_rerun()
    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('◀  PRIOR STEP'):
            swap_app('C-baseline-classification')
        if button_col5.button('NEXT STEP ▶'):
            swap_app('E-predict')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
