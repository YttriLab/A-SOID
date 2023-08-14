import streamlit as st
import numpy as np
import os
from config.help_messages import IMPRESS_TEXT, NO_CONFIG_HELP
from utils.load_workspace import load_refinement, load_features, load_iterX, save_data
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
    st.success(f'Included new training data ({new_targets.shape[0]} samples) into '
               f':orange[ITERATION {selected_iter+1}]. '
               f'Updated config.')
    st.balloons()


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        iteration = config["Processing"].getint("ITERATION")
        selected_iter = ri.selectbox('Select Iteration #', np.arange(iteration + 1), iteration)
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)
        refined_vid_dirs = [d for d in os.listdir(os.path.join(project_dir, iter_folder))
                            if os.path.isdir(os.path.join(project_dir, iter_folder, d))]
        selected_refine_dirs = ri.multiselect('Select Refinement', refined_vid_dirs, refined_vid_dirs)

        try:
            new_features_dir = []
            new_targets_dir = []
            for selected_refine_dir in selected_refine_dirs:
                [vid_path,
                 features,
                 predict,
                 examples_idx,
                 refinements] = load_refinement(
                    os.path.join(project_dir, iter_folder), selected_refine_dir)
                # st.write(refinements, examples_idx)
                new_feats_byclass = []
                new_targets_byclass = []
                for i, annotation_cls in enumerate(annotation_classes):
                    submitted_feats = []
                    submitted_targets = []
                    for j in range(len(examples_idx[annotation_cls])):
                        refined_behaviors = [refinements[annotation_cls][j]['Behavior'][k]
                                                  for k in range(
                            len(refinements[annotation_cls][j]['Behavior']))]
                        start_idx = examples_idx[annotation_cls][j][0]
                        stop_idx = examples_idx[annotation_cls][j][1]

                        submitted_feats.append(features[start_idx:stop_idx, :])
                        submitted_targets.append([annotation_classes.index(refined_behaviors[k])
                                                  for k in range(len(refined_behaviors))])
                    try:
                        new_feats_byclass.append(np.vstack(submitted_feats))
                        new_targets_byclass.append(np.hstack(submitted_targets))
                    except:
                        pass
                new_features_dir.append(np.vstack(new_feats_byclass))
                new_targets_dir.append(np.hstack(new_targets_byclass))
            new_features = np.vstack(new_features_dir)
            new_targets = np.hstack(new_targets_dir)
        except:
            pass

        if len(selected_refine_dirs) > 0:
            create_button = st.button(f'Create :orange[ITERATION {selected_iter+1}] training dataset')
            if create_button:
                create_new_training_features_targets(project_dir, selected_iter, new_features, new_targets)
        else:
            try:
                [_, _, _, _, _, _] = load_iterX(project_dir, iter_folder)
                if 'refinements' not in st.session_state:
                    st.markdown(f'Refine some behaviors in :orange[Refine Behaviors]')
                else:
                    st.markdown(f'Refine some behaviors in :orange[Refine Behaviors]')
            except:
                st.markdown(f'Move to :orange[Active Learning] to retrain classifier')
                pass

    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)
