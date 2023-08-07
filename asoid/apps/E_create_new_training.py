import streamlit as st
import numpy as np
import os
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.load_workspace import load_refinement, load_features, save_data
from utils.project_utils import update_config


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
               appended_targets,
               shuffled_splits,
               frames2integ])
    parameters_dict = {
        "Processing": dict(
            ITERATION = selected_iter+1,
        )
    }
    st.session_state['config'] = update_config(project_dir, updated_params=parameters_dict)
    st.success(f'Included new training data into :orange[ITERATION {selected_iter+1}]. Updated config.')
    st.balloons()


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        iteration = config["Processing"].getint("ITERATION")
        selected_iter = ri.selectbox('select iteration number', np.arange(iteration+1), iteration)
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)
        refined_vid_dirs = [d for d in os.listdir(os.path.join(project_dir, iter_folder))
                            if os.path.isdir(os.path.join(project_dir, iter_folder, d))]

        selected_refine_dir = ri.selectbox('select existing refinement videos', refined_vid_dirs)
        # short_vid_dir = os.path.join(project_dir, iter_folder, selected_refine_dir)

        if 'refinements' not in st.session_state:
            st.write('refinements not found')
            [st.session_state['video_path'],
             st.session_state['features'],
             st.session_state['predict'],
             st.session_state['examples_idx'],
             st.session_state['refinements']] = load_refinement(
                os.path.join(project_dir, iter_folder), selected_refine_dir)
        # st.write(st.session_state)
        new_feats_byclass = []
        new_targets_byclass = []
        for i, annotation_cls in enumerate(list(st.session_state['examples_idx'].keys())):
            submitted_feats = []
            submitted_targets = []
            for j, submitted in enumerate([st.session_state['refinements'][annotation_cls][i]['submitted']
                              for i in range(len(st.session_state['refinements'][annotation_cls]))]):
                if submitted == True:
                    submitted_feats.append(st.session_state['features'][st.session_state['examples_idx'][annotation_cls]][j, :])
                    submitted_targets.append(i*np.ones(1, ))
            try:
                new_feats_byclass.append(np.vstack(submitted_feats))
                new_targets_byclass.append(np.hstack(submitted_targets))
            except:
                pass
        new_features = np.vstack(new_feats_byclass)
        new_targets = np.hstack(new_targets_byclass)
        if selected_refine_dir is not None:
            create_button = st.button(f'Create :orange[ITERATION {selected_iter+1}] training dataset')
            if create_button:
                create_new_training_features_targets(project_dir, selected_iter, new_features, new_targets)
        else:
            st.markdown(f'Move to :orange[Active Learning] to retrain classifier')

    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
