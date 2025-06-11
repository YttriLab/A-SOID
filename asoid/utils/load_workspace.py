import streamlit as st
import os
import joblib
from asoid.utils.project_utils import load_config


def _load_sav(path, name, filename):
    """just a simplification for all those load functions"""
    with open(os.path.join(path, name, filename), 'rb') as fr:
        data = joblib.load(fr)
    return data


def save_data(path, name, filename, data):
    """just a simplification for all those save functions"""
    with open(os.path.join(path, name, filename), 'wb') as f:
        joblib.dump(data, f)


# @st.cache_data
def load_data(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, "data.sav")
    config, _ = load_config(os.path.join(path, name))

    return [i for i in data], config


# @st.cache_data
def load_test(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, "test.sav")
    return [i for i in data]


# @st.cache_data
def load_test_targets(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, "test_targets.sav")
    return [i for i in data]


# @st.cache_data
def load_new_pose(filename):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    with open(filename, 'rb') as fr:
        data = joblib.load(fr)
    return data


# @st.cache_data
def load_features(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'feats_targets.sav')
    # config, _ = load_config(os.path.join(path, name))

    return [i for i in data]


# @st.cache_data
def load_predict_proba(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'predict_proba.sav')
    return [i for i in data]


# @st.cache_data
def load_newest_model(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'newest_model.sav')
    return [i for i in data]


# @st.cache_data
def load_test_performance(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'test_performance.sav')
    return [i for i in data]


# @st.cache_data
def load_all_train(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'all_train.sav')
    return [i for i in data]



# @st.cache_data
def load_iter0(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'iter0.sav')
    return [i for i in data]


# @st.cache_data
def load_iterX(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'iterX.sav')
    return [i for i in data]


# @st.cache_data
def load_refinement(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'refinements.sav')
    # st.write(data)
    return [i for i in data]


# @st.cache_data
def load_refine_params(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'refine_params.sav')
    return [i for i in data]

@st.cache_data
def load_feats(path, name):
    # with open(os.path.join(path, str.join('', (name, 'feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "feats.sav")
    return [i for i in data]
@st.cache_data
def load_full_feats_targets(path, name):
    # with open(os.path.join(path, str.join('', (name, 'feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "full_feats_targets.sav")
    return [i for i in data]


@st.cache_data
def load_embeddings(path, name):
    # with open(os.path.join(path, name, 'target_embeddings.sav'), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "target_embeddings.sav")
    return [i for i in data]


@st.cache_data
def load_batch_predictions(path, name):
    data = _load_sav(path, name, "batch_predictions.sav")
    return [i for i in data]


@st.cache_data
def load_test_feats(path, name):
    # with open(os.path.join(path, str.join('', (name, '_test_feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "test_feats.sav")
    return [i for i in data]


@st.cache_data
def load_heldout(path, name):
    # with open(os.path.join(path, str.join('', (name, '_test_feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "heldout_feats_targets.sav")
    return [i for i in data]


@st.cache_data
def load_test_embeddings(path, name):
    # with open(os.path.join(path, str.join('', (name, '_test_embeddings.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "test_embeddings.sav")
    return [i for i in data]

@st.cache_data
def load_class_embeddings(path, name, class_name):
    # with open(os.path.join(path, str.join('', (name, '_test_embeddings.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, f"{class_name}_embeddings.sav")
    return [i for i in data]

@st.cache_data
def load_class_clusters(path, name,class_name):
    # with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, f"{class_name}_clusters.sav")
    return [i for i in data]


@st.cache_data
def load_clusters(path, name):
    # with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "clusters.sav")
    return [i for i in data]


@st.cache_resource
def load_classifier(path, name):
    # with open(os.path.join(path, str.join('', (name, '_randomforest.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "randomforest.sav")
    return [i for i in data]


@st.cache_data
def load_predictions(path, name):
    # with open(os.path.join(path, str.join('', (name, '_predictions.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "predictions.sav")
    return [i for i in data]


def load_new_feats(path, name):
    # with open(os.path.join(path, str.join('', (name, '_new_feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "new_feats.sav")
    return [i for i in data]

@st.cache_data
def load_motion_energy(path, name):
    file_path_motion = os.path.join(path, name, 'motionenergy.sav')
    with open(file_path_motion, 'rb') as fr:
        motion_energy = joblib.load(fr)
    return motion_energy
