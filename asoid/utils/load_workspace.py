import streamlit as st
import os
import joblib
from asoid.utils.project_utils import load_config
import asoid.utils.loading_utils as lu


def _load_sav(path, name, filename):
    #to avoid backwards compatibility issues
    return lu._load_sav(path, name, filename)


def save_data(path, name, filename, data):
    """just a simplification for all those save functions"""
    #to avoid backwards compatibility issues since move to utils.loading_utils
    lu.save_data(path, name, filename, data)

"""All loading functions with caching for streamlit use"""

@st.cache_data
def load_data(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_data(path, name)


@st.cache_data
def load_test(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_test(path, name)


@st.cache_data
def load_test_targets(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_test_targets(path, name)


@st.cache_data
def load_new_pose(filename):
    #wrapping the function in a streamlit caching function
    return lu.load_new_pose(filename)


@st.cache_data
def load_features(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_features(path, name)


@st.cache_data
def load_predict_proba(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_predict_proba(path, name)


@st.cache_data
def load_newest_model(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_newest_model(path, name)


@st.cache_data
def load_test_performance(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_test_performance(path, name)


@st.cache_data
def load_all_train(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_all_train(path, name)



@st.cache_data
def load_iter0(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_iter0(path, name)


@st.cache_data
def load_iterX(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_iterX(path, name)



@st.cache_data
def load_feats(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_feats(path, name)
@st.cache_data
def load_full_feats_targets(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_full_feats_targets(path, name)


@st.cache_data
def load_embeddings(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_embeddings(path, name)


@st.cache_data
def load_batch_predictions(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_batch_predictions(path, name)


@st.cache_data
def load_test_feats(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_test_feats(path, name)


@st.cache_data
def load_heldout(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_heldout(path, name)


@st.cache_data
def load_test_embeddings(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_test_embeddings(path, name)

@st.cache_data
def load_class_embeddings(path, name, class_name):
    #wrapping the function in a streamlit caching function
    return lu.load_class_embeddings(path, name)

@st.cache_data
def load_class_clusters(path, name,class_name):
    #wrapping the function in a streamlit caching function
    return lu.load_class_clusters(path, name)


@st.cache_data
def load_clusters(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_clusters(path, name)


@st.cache_resource
def load_classifier(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_classifier(path, name)


@st.cache_data
def load_predictions(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_predictions(path, name)


def load_new_feats(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_new_feats(path, name)

@st.cache_data
def load_motion_energy(path, name):
    #wrapping the function in a streamlit caching function
    return lu.load_motion_energy(path, name)


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


