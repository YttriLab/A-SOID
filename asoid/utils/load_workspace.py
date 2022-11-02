import streamlit as st
import os
import joblib
from utils.project_utils import load_config


def _load_sav(path, name, filename):
    """just a simplification for all those load functions"""
    with open(os.path.join(path, name, filename), 'rb') as fr:
        data = joblib.load(fr)
    return data


def save_data(path, name, filename, data):
    """just a simplification for all those save functions"""
    with open(os.path.join(path, name, filename), 'wb') as f:
        joblib.dump(data, f)


@st.cache
def load_data(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, "data.sav")
    config, _ = load_config(os.path.join(path, name))

    return [i for i in data], config


@st.cache
def load_test(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, "test.sav")
    return [i for i in data]


@st.cache
def load_test_targets(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, "test_targets.sav")
    return [i for i in data]


@st.cache
def load_new_pose(filename):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    with open(filename, 'rb') as fr:
        data = joblib.load(fr)
    return data


@st.cache
def load_features(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'feats_targets.sav')
    # config, _ = load_config(os.path.join(path, name))

    return [i for i in data]


@st.cache
def load_predict_proba(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'predict_proba.sav')
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_newest_model(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'newest_model.sav')
    return [i for i in data]


@st.cache
def load_test_performance(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'test_performance.sav')
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_all_train(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'all_train.sav')
    return [i for i in data]



@st.cache(allow_output_mutation=True)
def load_iter0(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'iter0.sav')
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_iterX(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'iterX.sav')
    return [i for i in data]


# TODO: deprecate
def query_workspace():
    working_dir = st.sidebar.text_input('Enter the prior A-SOiD working directory:')
    try:
        os.listdir(working_dir)
        st.markdown(
            'You have selected **{}** for prior working directory.'.format(working_dir))
    except FileNotFoundError:
        st.error('No such directory')
    # check for project folders containing a data.sav file and config.ini

    asoid_variables = [i for i in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, i))
                       and 'data.sav' in os.listdir(os.path.join(working_dir, i))
                       and 'config.ini' in os.listdir(os.path.join(working_dir, i))
                       ]
    # check if prefix is already folder itself

    if not asoid_variables:
        if 'data.sav' in os.listdir(working_dir) and 'config.ini' in os.listdir(working_dir):
            asoid_variables = [os.path.basename(working_dir)]
            # rename working_dir to upper level directory for easier handling later
            working_dir = os.path.dirname(working_dir)

    asoid_prefix = []
    for var in asoid_variables:
        if var not in asoid_prefix:
            asoid_prefix.append(var)
    prefix = st.selectbox('Select prior A-SOiD prefix', asoid_prefix)
    try:
        st.markdown('You have selected **{}** for prior prefix.'.format(prefix))
    except TypeError:
        st.error('Please input a prior prefix to load workspace.')
    return working_dir, prefix


def query_workspace_v2(key1, key2):
    working_dir = st.text_input('Enter the prior A-SOiD working directory:', key=key1)
    try:
        if not working_dir.endswith('/'):
            working_dir = working_dir + "/"
        os.listdir(working_dir)
        st.markdown(
            'You have selected **{}** for prior working directory.'.format(working_dir))
    except FileNotFoundError:
        st.error('No such directory')
    # check for project folders containing a data.sav file and config.ini

    asoid_variables = [i for i in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, i))
                       and 'data.sav' in os.listdir(os.path.join(working_dir, i))
                       and 'config.ini' in os.listdir(os.path.join(working_dir, i))
                       ]
    # check if prefix is already folder itself
    if not asoid_variables:
        if 'data.sav' in os.listdir(working_dir) and 'config.ini' in os.listdir(working_dir):
            asoid_variables = [os.path.basename(working_dir)]
            # rename working_dir to upper level directory for easier handling later
            working_dir = os.path.dirname(working_dir)

    asoid_prefix = []
    for var in asoid_variables:
        if var not in asoid_prefix:
            asoid_prefix.append(var)
    prefix = st.selectbox('Select prior A-SOiD prefix', asoid_prefix, key=key2)
    try:
        st.markdown('You have selected **{}** for prior prefix.'.format(prefix))
    except TypeError:
        st.error('Please input a prior prefix to load workspace.')
    return working_dir, prefix


@st.cache
def load_feats(path, name):
    # with open(os.path.join(path, str.join('', (name, 'feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "feats.sav")
    return [i for i in data]
@st.cache
def load_full_feats_targets(path, name):
    # with open(os.path.join(path, str.join('', (name, 'feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "full_feats_targets.sav")
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_embeddings(path, name):
    # with open(os.path.join(path, name, 'target_embeddings.sav'), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "target_embeddings.sav")
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_batch_predictions(path, name):
    data = _load_sav(path, name, "batch_predictions.sav")
    return [i for i in data]


@st.cache
def load_test_feats(path, name):
    # with open(os.path.join(path, str.join('', (name, '_test_feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "test_feats.sav")
    return [i for i in data]


@st.cache
def load_heldout(path, name):
    # with open(os.path.join(path, str.join('', (name, '_test_feats.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "heldout_feats_targets.sav")
    return [i for i in data]


@st.cache
def load_test_embeddings(path, name):
    # with open(os.path.join(path, str.join('', (name, '_test_embeddings.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "test_embeddings.sav")
    return [i for i in data]

@st.cache
def load_class_embeddings(path, name, class_name):
    # with open(os.path.join(path, str.join('', (name, '_test_embeddings.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, f"{class_name}_embeddings.sav")
    return [i for i in data]

@st.cache(allow_output_mutation=True)
def load_class_clusters(path, name,class_name):
    # with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, f"{class_name}_clusters.sav")
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_clusters(path, name):
    # with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "clusters.sav")
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_classifier(path, name):
    # with open(os.path.join(path, str.join('', (name, '_randomforest.sav'))), 'rb') as fr:
    #     data = joblib.load(fr)
    data = _load_sav(path, name, "randomforest.sav")
    return [i for i in data]


@st.cache
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
