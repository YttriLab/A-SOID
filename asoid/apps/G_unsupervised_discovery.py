import streamlit as st
import os
import numpy as np
import pandas as pd
from pathlib import Path
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.unsupervised_discovery import Explorer
from stqdm import stqdm
from utils.extract_features import feature_extraction, \
    bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale
from utils.load_workspace import load_new_pose, load_iterX, save_data
from utils.view_results import Viewer
from sklearn.preprocessing import LabelEncoder
from utils.import_data import load_labels_auto
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import hdbscan
import joblib
import pickle
from tqdm import notebook
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns


TITLE = "Unsupervised discovery"


def prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                 framerate, videos_dir, project_dir, iter_dir, pose_expander):
    # pose_exapnder = st.expander('pose'.upper(), expanded=True)
    if software == 'CALMS21 (PAPER)':
        ROOT = Path(__file__).parent.parent.parent.resolve()
        new_pose_sav = os.path.join(ROOT.joinpath("new_test"), './new_pose.sav')
        new_pose_list = load_new_pose(new_pose_sav)
    else:

        new_pose_csvs = pose_expander.file_uploader('Upload Corresponding Pose Files',
                                                    accept_multiple_files=True,
                                                    type=ftype, key='pose')

        # st.write(new_pose_csvs, new_videos)
        if new_pose_csvs is not None:
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



        else:
            st.session_state['uploaded_pose'] = []
            # st.session_state['uploaded_vid'] = None


def get_features_labels(iterX_model, frames2integ, project_dir, iter_folder, placeholder,
                        ):
    features = [None]
    predict_arr = [None]
    processed_input_data = st.session_state['uploaded_pose']

    if placeholder.button('preprocess files'):
        st.session_state['disabled'] = True
        # extract features, bin them
        features = []
        for i, data in enumerate(processed_input_data):
            # using feature scaling from training set
            feats, _ = feature_extraction([data], 1, frames2integ)
            features.append(feats)
        predict_arr = []
        for i in stqdm(range(len(features)), desc="Behavior prediction from spatiotemporal features"):
            with st.spinner('Predicting behavior from features...'):
                predict = bsoid_predict_numba_noscale([features[i]], iterX_model)
                pred_proba = bsoid_predict_proba_numba_noscale([features[i]], iterX_model)
                predict_arr.append(np.array(predict).flatten())
        with st.spinner('Saving...'):
            save_data(project_dir, iter_folder, 'embedding_input.sav',
                  [np.vstack(features), np.hstack(predict_arr)])
        st.session_state['input_sav'] = os.path.join(project_dir, iter_folder, 'embedding_input.sav')
        st.success('Done. Type "R" to Refresh.')


UMAP_PARAMS = {
    'n_neighbors': 30,
    'min_dist': 0.0,
    'random_state': 42,
}


HDBSCAN_PARAMS = {
    'min_samples': 1
}


def hdbscan_classification(umap_embeddings,cluster_range):
    max_num_clusters = -np.infty
    num_clusters = []
    min_cluster_size = np.linspace(cluster_range[0], cluster_range[1], 20)
    for min_c in min_cluster_size:
        learned_hierarchy = hdbscan.HDBSCAN(
            prediction_data=True,min_cluster_size=int(round(min_c * 0.01 * umap_embeddings.shape[0])),
            **HDBSCAN_PARAMS).fit(umap_embeddings)
        num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
        if num_clusters[-1] > max_num_clusters:
            max_num_clusters = num_clusters[-1]
            retained_hierarchy = learned_hierarchy
    assignments = retained_hierarchy.labels_
    assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
    soft_assignments = np.argmax(assign_prob,axis=1)
    return retained_hierarchy,assignments,assign_prob,soft_assignments


def pca_umap_hdbscan(target_behavior, annotation_classes, input_sav, cluster_range,
                     project_dir, iter_folder, placeholder):
    with open(input_sav, 'rb') as fr:
        [features, predictions] = joblib.load(fr)

    if placeholder.button('embed'):
        for target_behav in target_behavior:
            target_beh_id = annotation_classes.index(target_behav)
            selected_features = features[predictions==target_beh_id]
            st.write(selected_features.shape)
            scalar = StandardScaler()
            selected_feats_scaled = scalar.fit_transform(selected_features)
            pca = PCA(random_state=42)
            pca.fit(selected_feats_scaled)
            n_dim = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0]
            st.info(f'{n_dim} latent dimension achieves 70% variacne')
            reducer = umap.UMAP(**UMAP_PARAMS, n_components=n_dim)

            with st.spinner(f'Embedding into {n_dim} dimensions...'):
                umap_embeddings = reducer.fit_transform(selected_feats_scaled)
            with st.spinner('Clustering...'):
                retained_hierarchy, assignments, assign_prob, soft_assignments = hdbscan_classification(umap_embeddings,
                                                                                                        cluster_range)
            save_data(project_dir, iter_folder, 'embedding_output.sav',
                      [umap_embeddings, assignments, soft_assignments])
            st.session_state['output_sav'] = os.path.join(project_dir, iter_folder, 'embedding_output.sav')
            st.success('Done. Type "R" to Refresh.')


def plot_targeted_discovery(target_behavior, output_sav):
    with open(output_sav, 'rb') as fr:
        [umap_embeddings, assignments, soft_assignments] = joblib.load(fr)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    colors = sns.color_palette('husl', n_colors=len(np.unique(assignments)))
    for i, g in enumerate(np.unique(soft_assignments)):
        # if g >= 0:

        idx_g = np.where(soft_assignments == g)[0]
        ax.scatter(umap_embeddings[idx_g, 0], umap_embeddings[idx_g, 1],
                   s=0.5,
                   c=colors[i]
                   # cmap=plt.cm.Spectral,
                   # label=[f'groom sub{i}' for i in range(len(np.unique(assignments)))]
                   )

    ax.set_title(f'{target_behavior[0]} subtypes')
    # ax.legend([f'groom sub{i}' for i in range(len(np.unique(assignments)))])
    st.pyplot(fig)

def plot_embedding(class_name, output_sav):
    with open(output_sav, 'rb') as fr:
        [umap_embeddings, assignments, soft_assignments] = joblib.load(fr)
    number_to_class = {i: s for i, s in enumerate(class_name)}
    trace1 = go.Scatter(x=umap_embeddings[:,0],
                        y=umap_embeddings[:,1],
                        # name=number_to_class,
                        mode='markers'
                        )

    fig = make_subplots()
    fig.add_trace(trace1)

    fig.update_xaxes(title_text="UMAP Dim 1",row=1,col=1,showticklabels=False)
    fig.update_yaxes(title_text="UMAP Dim 2",row=1,col=1,showticklabels=False)
    fig.update_layout(title_text="Unsupervised Embedding",
                      )

    return fig


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
        if 'uploaded_pose' not in st.session_state:
            st.session_state['uploaded_pose'] = []

        st.session_state['disabled'] = False
        left_col, right_col = st.columns(2)
        pose_expander = left_col.expander('pose'.upper(), expanded=True)
        # split_behav_exapnder = right_col.expander('Select Behavior to Split', expanded=True)

        prompt_setup(software, ftype, selected_bodyparts, annotation_classes,
                     framerate, videos_dir, project_dir, iter_folder, pose_expander)
        [iterX_model, _, _] = load_iterX(project_dir, iter_folder)
        st.info(f'loaded {iter_folder} model')
        buttonL, buttonR = st.columns(2)
        if 'input_sav' not in st.session_state:
            st.session_state['input_sav'] = None
        if 'output_sav' not in st.session_state:
            st.session_state['output_sav'] = None
        if st.session_state['input_sav'] is None:
            get_features_labels(iterX_model, frames2integ, project_dir, iter_folder, buttonL
                            )

        # st.write(st.session_state)
        target_behavior = right_col.multiselect('Select Behavior to Split', annotation_classes, annotation_classes[3])
        if st.session_state['input_sav'] is not None:

            if buttonR.button(':red[Clear Processed Pose.]'):
                st.session_state['input_sav'] = None
                st.success('Cleared. Type "R" to Refresh.')
            if st.session_state['output_sav'] is None:

                pca_umap_hdbscan(target_behavior, annotation_classes, st.session_state['input_sav'], [5, 5.5],
                                 project_dir, iter_folder, buttonL)

            else:
                if buttonR.button(':red[Clear Embedding.]'):
                    st.session_state['output_sav'] = None
                    st.success('Cleared. Type "R" to Refresh.')
                # plot_targeted_discovery(target_behavior, st.session_state['output_sav'])
                fig = plot_embedding(annotation_classes, st.session_state['output_sav'])
                st.plotly_chart(fig)
        # st.subheader("Directed discovery")
        # explorer = Explorer(config)
        # explorer.main()

    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        # button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        # if button_col1.button('◀  PRIOR STEP'):
        #     swap_app('F-view')
        # if button_col5.button('NEXT STEP ▶'):
        #     swap_app('A-data-preprocess')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
