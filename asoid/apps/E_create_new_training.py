import streamlit as st
import numpy as np
import os
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.load_workspace import load_refinement, load_features, save_data

TITLE = "Create new dataset"


def create_new_training_features_targets(project_dir, selected_iter, new_features, new_targets):
    # load existing training
    iter_folder = str.join('', ('iteration-', str(selected_iter)))
    [features, targets, shuffled_splits, frames2integ] = \
        load_features(project_dir, iter_folder)
    # add iteration number
    new_iter_folder = str.join('', ('iteration-', str(selected_iter+1)))
    os.makedirs(os.path.join(project_dir, new_iter_folder), exist_ok=True)
    # incorporate new features/targets into existing training
    appended_features = np.vstack((features, new_features))
    appended_targets = np.hstack((targets, new_targets))
    # save into new iteration folder
    save_data(project_dir, new_iter_folder, 'feats_targets.sav',
              [appended_features,
               new_targets,
               shuffled_splits,
               frames2integ])


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        iteration = config["Processing"].getint("ITERATION")
        selected_iter = ri.selectbox('select iteration number', np.arange(iteration+1))
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)

        if 'refinements' not in st.session_state:

            [video_name,
             st.session_state['features'],
             st.session_state['predict'],
             st.session_state['examples_idx'], st.session_state['refinements']] = load_refinement(
                project_dir, iter_folder)

        for i, annotation_cls in enumerate(list(st.session_state['examples_idx'].keys())):
            st.write([st.session_state['refinements'][annotation_cls][i]['submitted']
                     for i in range(len(st.session_state['refinements'][annotation_cls]))])
            new_features = st.session_state['features'][st.session_state['examples_idx'][annotation_cls]]
            new_targets = i*np.ones(len(st.session_state['examples_idx'][annotation_cls]))
            st.write(new_features.shape, new_targets)

        # st.write(st.session_state['features'].shape)
        # st.write(st.session_state['predict'].shape)
        # st.write(st.session_state['examples_idx'].keys())
        # st.write(st.session_state['predict'][st.session_state['examples_idx']['other']])
        # st.dataframe(st.session_state['examples_idx'])


        create_button = st.button(f'Create iteration {iteration+1} training dataset'.upper())
        if create_button:
            create_new_training_features_targets(project_dir, selected_iter, new_features, new_targets)




    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
