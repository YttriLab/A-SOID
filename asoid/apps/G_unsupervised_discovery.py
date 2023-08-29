import streamlit as st
import os
import pandas as pd
from pathlib import Path
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from stqdm import stqdm
from utils.extract_features import feature_extraction, \
    bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale
from utils.load_workspace import load_new_pose, load_iterX, save_data, load_features
from datetime import date
from utils.project_utils import create_new_project, update_config, copy_config
from utils.preprocessing import adp_filt, sort_nicely
from config.help_messages import *
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import hdbscan
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


TITLE = "Unsupervised discovery"


def prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                 framerate, videos_dir, project_dir, iter_dir, pose_expander):
    if software == 'CALMS21 (PAPER)':
        try:
            #TODO: deprecate
            ROOT = Path(__file__).parent.parent.parent.resolve()
            new_pose_sav = os.path.join(ROOT.joinpath("new_test"), './new_pose.sav')
            new_pose_list = load_new_pose(new_pose_sav)
        except FileNotFoundError:
            st.error("The CALMS21 data set is not designed to be used with the discovery step.")
            st.stop()

    else:
        new_pose_csvs = pose_expander.file_uploader('Upload Corresponding Pose Files',
                                                    accept_multiple_files=True,
                                                    type=ftype, key='pose')

        # if len(new_pose_csvs) > 0:
        #     new_pose_list = []
        #     pose_names_list = []
        #     for i, f in enumerate(new_pose_csvs):
        #         current_pose = pd.read_csv(f,
        #                                    header=[0, 1, 2], sep=",", index_col=0)
        #         bp_level = 1
        #         bp_index_list = []
        #         for bp in selected_bodyparts:
        #             bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
        #             bp_index_list.append(bp_index)
        #         selected_pose_idx = np.sort(np.array(bp_index_list).flatten())
        #         # get rid of likelihood columns for deeplabcut
        #         idx_llh = selected_pose_idx[2::3]
        #         # the loaded sleap file has them too, so exclude for both
        #         idx_selected = [i for i in selected_pose_idx if i not in idx_llh]
        #         # idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
        #
        #
        #         new_pose_list.append(np.array(current_pose.iloc[:, idx_selected]))
        #         pose_names_list.append(f.name)
        #     st.session_state['uploaded_pose'] = new_pose_list
        #     st.session_state['uploaded_fnames'] = pose_names_list
        # else:
        #     st.session_state['uploaded_pose'] = []
        #     st.session_state['uploaded_fnames'] = []


def get_features_labels(selected_bodyparts, iterX_model, frames2integ, project_dir, iter_folder, placeholder,
                        ):
    features = [None]
    predict_arr = [None]
    new_pose_csvs = st.session_state['pose']
    input_features, input_targets = None, None
    if placeholder.button('preprocess files'):
        st.session_state['disabled'] = True
        pose_names_list = []
        features = []
        # filter here
        for i, f in enumerate(new_pose_csvs):
            current_pose = pd.read_csv(f,
                                       header=[0, 1, 2], sep=",", index_col=0
                                       )
            pose_names_list.append(f.name)
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
            filt_pose, _ = adp_filt(current_pose, idx_selected, idx_llh)
            # using feature scaling from training set
            feats, _ = feature_extraction([filt_pose], 1, frames2integ)
            features.append(feats)
        st.session_state['uploaded_fnames'] = pose_names_list
        predict_arr = []
        for i in stqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
                pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
                predict_arr.append(np.array(predict).flatten())

        input_features = np.vstack(features)
        input_targets = np.hstack(predict_arr)
        # st.write(input_targets.shape, input_features.shape)
        with st.spinner('Saving...'):
            save_data(project_dir, iter_folder, 'embedding_input.sav',
                      [input_features, input_targets])
        st.session_state['input_sav'] = os.path.join(project_dir, iter_folder, 'embedding_input.sav')
        st.success('Done. Type "R" to Refresh.')
    return input_features, input_targets


UMAP_PARAMS = {
    'n_neighbors': 30,
    'min_dist': 0.0,
    'random_state': 42,
}

HDBSCAN_PARAMS = {
    'min_samples': 1
}


def hdbscan_classification(umap_embeddings, cluster_range):
    max_num_clusters = -np.infty
    num_clusters = []
    min_cluster_size = np.linspace(cluster_range[0], cluster_range[1], 20)
    for min_c in min_cluster_size:
        learned_hierarchy = hdbscan.HDBSCAN(
            prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * umap_embeddings.shape[0])),
            **HDBSCAN_PARAMS).fit(umap_embeddings)
        num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
        if num_clusters[-1] > max_num_clusters:
            max_num_clusters = num_clusters[-1]
            retained_hierarchy = learned_hierarchy
    assignments = retained_hierarchy.labels_
    assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
    soft_assignments = np.argmax(assign_prob, axis=1)
    return retained_hierarchy, assignments, assign_prob, soft_assignments


def pca_umap_hdbscan(target_behavior, annotation_classes, input_sav, cluster_range,
                     project_dir, iter_folder, left_col, right_col):
    if input_sav is not None:
        with open(input_sav, 'rb') as fr:
            [features, predictions] = joblib.load(fr)
        umap_embeddings = {key: [] for key in target_behavior}
        retained_hierarchy = {key: [] for key in target_behavior}
        assignments = {key: [] for key in target_behavior}
        assign_prob = {key: [] for key in target_behavior}
        soft_assignments = {key: [] for key in target_behavior}

        if right_col.button('Embed and Cluster Targeted Behavior'):
            # for each target behavior, scale the features
            for target_behav in target_behavior:
                with st.spinner(f'working on splitting {target_behav}...'):
                    target_beh_id = annotation_classes.index(target_behav)
                    selected_features = features[predictions == target_beh_id]
                    scalar = StandardScaler()
                    selected_feats_scaled = scalar.fit_transform(selected_features)
                    pca = PCA(random_state=42)
                    # define manifold dim to variance explained at 70%
                    pca.fit(selected_feats_scaled)
                    n_dim = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0]
                    # st.info(f'{n_dim} latent dimension achieves 70% variacne')
                    reducer = umap.UMAP(**UMAP_PARAMS, n_components=n_dim)

                    with st.spinner(f'Embedding into {n_dim} dimensions...'):
                        umap_embeddings[target_behav] = reducer.fit_transform(selected_feats_scaled)

                    with st.spinner('Clustering...'):
                        retained_hierarchy[target_behav], \
                            assignments[target_behav], assign_prob[target_behav], \
                            soft_assignments[target_behav] = hdbscan_classification(
                            umap_embeddings[target_behav],
                            cluster_range)

            save_data(project_dir, iter_folder, 'embedding_output.sav',
                      [umap_embeddings, assignments, soft_assignments])
            st.session_state['output_sav'] = os.path.join(project_dir, iter_folder, 'embedding_output.sav')
            st.success('Done. Type "R" to Refresh.')


def plot_hdbscan_embedding(output_sav):
    if output_sav is not None:
        with open(output_sav, 'rb') as fr:
            [umap_embeddings, assignments, soft_assignments] = joblib.load(fr)
        # some plotting parameters
        behav_figs = {key: [] for key in list(assignments.keys())}
        behav_groups = {key: [] for key in list(assignments.keys())}
        # behav_embeds = {key: [] for key in list(assignments.keys())}
        for behav in list(assignments.keys()):
            assign = assignments[behav]
            embeds = umap_embeddings[behav]
            unique_classes = np.unique(assign)
            group_types = ['Noise']
            group_types.extend(['Group{}'.format(i) for i in unique_classes if i >= 0])

            trace_list = []

            for num, g in enumerate(unique_classes):
                if g < 0:
                    idx = np.where(assign == g)[0]
                    trace_list.append(go.Scatter(x=embeds[idx, 0],
                                                 y=embeds[idx, 1],
                                                 name="Noise",
                                                 mode='markers'
                                                 )
                                      )
                else:
                    idx = np.where(assign == g)[0]
                    trace_list.append(go.Scatter(x=embeds[idx, 0],
                                                 y=embeds[idx, 1],
                                                 name=group_types[num],
                                                 mode='markers'
                                                 ))

            fig = make_subplots()
            for trace in trace_list:
                fig.add_trace(trace)
            fig.update_traces(marker_size=3)
            fig.update_layout(
                autosize=True,
                # width=800,
                # height=400,
                # title=f'{behav}',
                xaxis_title=dict(text=f'{behav.capitalize()} (Dim. 1)', font=dict(size=16, color='#EEEEEE')),
                yaxis_title=dict(text=f'{behav.capitalize()} (Dim. 2)', font=dict(size=16, color='#EEEEEE')),
                xaxis=dict(tickfont=dict(size=14, color='#EEEEEE')),
                yaxis=dict(tickfont=dict(size=14, color='#EEEEEE')),
                legend=dict(x=0.0, y=1.2, orientation='h', font=dict(color='#EEEEEE')),
            )
            behav_figs[behav] = fig
            behav_groups[behav] = group_types

        return behav_figs, behav_groups, umap_embeddings


def save_update_info(config, behavior_names_split):
    input_container = st.container()
    # create new project folder with prefix as name:
    working_dir = config["Project"].get("PROJECT_PATH")
    prefix_old = config["Project"].get("PROJECT_NAME")
    project_folder = os.path.join(working_dir, prefix_old)
    iteration = config["Processing"].getint("ITERATION")
    iter_folder = str.join('', ('iteration-', str(iteration)))
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    prefix_new = input_container.text_input('Enter filename prefix', d4,
                                            help=PREFIX_HELP)
    if prefix_new:
        st.success(f'Entered **{prefix_new}** as the prefix.')
    else:
        st.error('Please enter a prefix.')
    if st.button('Create new project', help= SAVE_NEW_HELP):
        parameters_dict = {
            'Data': dict(
                DATA_INPUT_FILES=sort_nicely(st.session_state['uploaded_fnames']),
                LABEL_INPUT_FILES=None),

            "Project": dict(
                PROJECT_PATH=working_dir,
                PROJECT_NAME=prefix_new,
                CLASSES=behavior_names_split,
            )
        }

        copy_config(project_folder, os.path.join(working_dir, prefix_new), iter_folder,
                    updated_params=parameters_dict)
        st.info(f'Created. Please delete the config, and reupload the new config from: {prefix_new}.')
    return working_dir, iter_folder, prefix_new


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

        if 'disabled' not in st.session_state:
            st.session_state['disabled'] = False
        if 'uploaded_fnames' not in st.session_state:
            st.session_state['uploaded_fnames'] = []

        left_col, right_col = st.columns(2)
        pose_expander = left_col.expander('pose'.upper(), expanded=True)
        prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                     framerate, videos_dir, project_dir, iter_folder, pose_expander)

        [iterX_model, _, _] = load_iterX(project_dir, iter_folder)

        buttonL, buttonR = st.columns(2)
        if 'input_sav' not in st.session_state:
            st.session_state['input_sav'] = None
        if 'output_sav' not in st.session_state:
            st.session_state['output_sav'] = None
        if st.session_state['input_sav'] is None:
            all_feats, all_labels = get_features_labels(selected_bodyparts, iterX_model, frames2integ,
                                                        project_dir, iter_folder,
                                                        left_col
                                                        )

        target_behavior = ri.multiselect('Select Behavior to Split', annotation_classes, annotation_classes[3])
        cluster_range = ri.slider('Minimum % for a cluster', min_value=0.1, max_value=10.0, value=3.0)

        if st.session_state['input_sav'] is not None:
            st.session_state['disabled'] = True

            if buttonL.button(':red[Clear Processed Pose]'):
                st.session_state['input_sav'] = None
                st.success('Cleared. Type "R" to Refresh.')
                st.session_state['disabled'] = False

            if st.session_state['output_sav'] is None:
                pca_umap_hdbscan(target_behavior, annotation_classes, st.session_state['input_sav'],
                                 [cluster_range, cluster_range+0.5],
                                 project_dir, iter_folder, left_col, ri)

            else:
                right_col_top = right_col.container()
                if buttonR.button(':red[Clear Embedding]'):
                    st.session_state['output_sav'] = None
                    st.success('Cleared. Type "R" to Refresh.')
                if st.session_state['output_sav'] is not None:
                    behav_figs, behav_groups, behav_embeds = plot_hdbscan_embedding(st.session_state['output_sav'])
                    for behav_keys in list(behav_figs.keys()):
                        right_col_top.subheader(f'{behav_keys.capitalize()}')
                        fig = behav_figs[behav_keys]
                        right_col_top.plotly_chart(fig, use_container_width=True)
                        right_col_top.info(f'Minimum cluster duration: '
                                           f'{np.round((cluster_range*behav_embeds[behav_keys].shape[0])/framerate, 1)} '
                                           f'total seconds')
                    with open(st.session_state['input_sav'], 'rb') as fr:
                        [all_feats, all_labels] = joblib.load(fr)
                    with open(st.session_state['output_sav'], 'rb') as fr:
                        [_, assignments, soft_assignments] = joblib.load(fr)
                    # reorder and integrate new labels
                    annotation_classes_ex = annotation_classes.copy()
                    if exclude_other:
                        other_id = annotation_classes.index('other')
                        annotation_classes_ex.pop(other_id)
                        idx_other = np.where(all_labels == other_id)[0]
                    left_col.subheader('Splitting')
                    for target_behav in target_behavior:
                        if target_behav == target_behavior[0]:
                            selected_subgroup = left_col.multiselect(f'Select the {target_behav} sub groups',
                                                 behav_groups[target_behav][1:], behav_groups[target_behav][1:],
                                                                     key=target_behav
                                                                     ,help = SUBCLASS_SELECT_HELP)
                            target_beh_id = annotation_classes_ex.index(target_behav)
                            # find where these target behavior is
                            idx_target_beh = np.where(all_labels == target_beh_id)[0]
                            # create a copy to perturb
                            all_labels_split = all_labels.copy()

                            group_id = [behav_groups[target_behav][1:].index(sel) for sel in selected_subgroup]
                            new_assigns = soft_assignments[target_behav].copy()
                            # for each behavior being split, change the index into last->last+n
                            count = 1
                            for id_ in [behav_groups[target_behav][1:].index(sel)
                                        for sel in behav_groups[target_behav][1:]]:
                                if id_ in group_id:
                                    max_id = np.max(all_labels_split) + 1
                                    new_assigns[soft_assignments[target_behav]==id_] = max_id + count
                                    count += 1
                                else:
                                    max_id = np.max(all_labels_split) + 1
                                    new_assigns[soft_assignments[target_behav] == id_] = max_id
                            all_labels_split[idx_target_beh] = new_assigns

                            # put other in last if exclude, for active learning to ignore
                            # has to put prior to label encoder
                            if exclude_other:
                                all_labels_split[idx_other] = np.max(all_labels_split[idx_target_beh]) + 1

                            # reorder using labelencoder
                            encoder = LabelEncoder()
                            all_labels_split_reorg = encoder.fit_transform(all_labels_split)
                            annot_array = np.array(annotation_classes_ex)
                            # rename these groups into label_n
                            behavior_names_split = list(annot_array[annot_array != target_behav])
                            behavior_names_split.extend([f'{target_behav}_{i}' if i > 0 else f'{target_behav}'
                                                         for i in
                                                         range(len(np.unique(new_assigns)))])

                            # has to be placed after new split names to maintain last order
                            if exclude_other:
                                behavior_names_split.extend(['other'])

                        else:
                            selected_subgroup = left_col.multiselect(f'Select the {target_behav} sub groups',
                                                 behav_groups[target_behav][1:], behav_groups[target_behav][1:],
                                                                     key=target_behav,
                                                                     help = SUBCLASS_SELECT_HELP)
                            target_beh_id = annotation_classes_ex.index(target_behav)
                            # find where these target behavior is
                            idx_target_beh = np.where(all_labels == target_beh_id)[0]
                            # create a copy to perturb
                            all_labels_split = all_labels_split_reorg.copy()

                            group_id = [behav_groups[target_behav][1:].index(sel) for sel in selected_subgroup]
                            new_assigns = soft_assignments[target_behav].copy()
                            # for each behavior being split, change the index into last->last+n
                            count = 1
                            for id_ in [behav_groups[target_behav][1:].index(sel)
                                        for sel in behav_groups[target_behav][1:]]:
                                if id_ in group_id:
                                    max_id = np.max(all_labels_split) + 1
                                    new_assigns[soft_assignments[target_behav]==id_] = max_id + count
                                    count += 1
                                else:
                                    max_id = np.max(all_labels_split) + 1
                                    new_assigns[soft_assignments[target_behav] == id_] = max_id
                            all_labels_split[idx_target_beh] = new_assigns
                            # put other in last if exclude, for active learning to ignore
                            # has to put prior to label encoder
                            if exclude_other:
                                all_labels_split[idx_other] = np.max(all_labels_split[idx_target_beh]) + 1
                                other_id = behavior_names_split.index('other')
                                behavior_names_split.pop(other_id)
                            # reorder using labelencoder
                            encoder = LabelEncoder()
                            all_labels_split_reorg = encoder.fit_transform(all_labels_split)
                            annot_array = np.array(behavior_names_split)
                            # rename these groups into label_n
                            behavior_names_split = list(annot_array[annot_array != target_behav])
                            behavior_names_split.extend([f'{target_behav}_{i}' if i > 0 else f'{target_behav}'
                                                         for i in
                                                         range(len(np.unique(new_assigns)))])
                            # has to be placed after new split names to maintain last order
                            if exclude_other:
                                behavior_names_split.extend(['other'])
                    working_dir, iter_folder, prefix_new = save_update_info(config, behavior_names_split)
                    if os.path.isdir(os.path.join(working_dir, prefix_new, iter_folder)):
                        save_data(os.path.join(working_dir, prefix_new), iter_folder,
                                  'feats_targets.sav',
                                  [
                                      all_feats,
                                      all_labels_split_reorg,
                                      frames2integ
                                  ])

        else:
            st.session_state['disabled'] = False




    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
