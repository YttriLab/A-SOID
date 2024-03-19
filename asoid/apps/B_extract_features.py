import streamlit as st
import numpy as np
import os
from utils.project_utils import update_config
from utils.load_workspace import load_data, load_features
from utils.extract_features import Extract, interactive_durations_dist
from config.help_messages import *

TITLE = "Extract features"

EXTRACT_FEATURES_HELP = ("In this step, you will extract features from the labeled data you uploaded. "
                         "\n\n The features will be used to train the classifier and predict the behavior in the next steps."
                         "\n\n---\n\n"
                            "**Step 1**: Upload your project config file."
                            "\n\n **Step 2**: Set the parameters."
                            "\n\n **Step 3**: Extract the features."
                            "\n\n **Step 4**: Continue with :orange[Active Learning]."
                            "\n\n---\n\n"
                         ":blue[Feature extraction can be repeated but requires new training afterwards.]"
                         )


def prompt_setup(prompt_container, software, framerate, annotation_classes,
                 working_dir, prefix, show_only=False):

    data, config = load_data(working_dir, prefix)
    # get relevant data from data file
    [_, targets] = data
    default_bin_count = np.sqrt(np.hstack(targets).shape[0])
    split = prompt_container.checkbox('Split by annotation group', value=True,
                                      help = SPLIT_CLASSES_HELP)
    n_bins = prompt_container.slider('Number of bins?', 50, 1000, int(default_bin_count),
                                     help = BINS_SLIDER_HELP)
    fig = interactive_durations_dist(targets, annotation_classes, framerate,
                                     prompt_container,
                                     num_bins=n_bins,
                                     split_by_class=split,
                                     )
    frames2integ, num_splits = None, None
    if not show_only:
        # col_left, col_right = prompt_container.columns(2)
        prompt_exp = prompt_container.expander('Minimum Duration', expanded=True)
        # right_exp = col_right.expander('Number of splits', expanded=True)
        if not software == 'CALMS21 (PAPER)':
            duration_min = prompt_exp.number_input('Minimum duration (s) of behavior before transition:',
                                                 min_value=0.05, max_value=None,
                                                 value=0.1, key='fr',
                                                 help = MIN_DURATION_HELP)
            # num_splits = right_exp.number_input('number of shuffled splits:', min_value=1, max_value=20,
            #                                     value=10, key='ns',
            #                                     help = NUM_SPLITS_HELP)
        else:
            duration_min = prompt_exp.number_input('Minimum duration (s) of behavior before transition:',
                                                 min_value=0.05, max_value=None,
                                                 value=0.4, key='fr',
                                                 help = MIN_DURATION_HELP+ CALM_HELP)
            # num_splits = right_exp.number_input('number of shuffled splits:', min_value=1, max_value=20,
            #                                     value=10, key='ns',
            #                                     help = NUM_SPLITS_HELP+ CALM_HELP)
        frames2integ = round(framerate * (duration_min / 0.1))
        parameters_dict = {
            "Processing": dict(
                N_SHUFFLED_SPLIT=num_splits,
                MIN_DURATION=duration_min
            )
        }
        st.session_state['config']= update_config(os.path.join(working_dir, prefix), updated_params=parameters_dict)

    return frames2integ


def main(config=None):
    st.markdown("""---""")

    st.title("Extract Features")
    st.expander("What is this?", expanded=False).markdown(EXTRACT_FEATURES_HELP)

    if config is not None:
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        framerate = config["Project"].getfloat("FRAMERATE")
        iteration = config["Processing"].getint("ITERATION")
        is_3d = config["Project"].getboolean("IS_3D")
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(iteration)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)

        try:
            [_, _, _] = load_features(project_dir, iter_folder)
            prompt_container = st.container()
            message_container = st.container()
            redo_container = st.container()
            if not redo_container.checkbox('Re-extract features', help = RE_EXTRACT_HELP):
                frames2integ = \
                    prompt_setup(prompt_container, software, framerate, annotation_classes,
                                 working_dir, prefix, show_only=True)
                message_container.success(f'This prefix had been extracted.')
            else:
                frames2integ = \
                    prompt_setup(prompt_container, software, framerate, annotation_classes,
                                 working_dir, prefix)
                if st.button('Extract Features', help = EXTRACT_FEATURES_HELP):
                    extractor = Extract(working_dir, prefix, frames2integ, is_3d)
                    extractor.main()
        except FileNotFoundError:
            try:
                prompt_container = st.container()
                frames2integ = \
                    prompt_setup(prompt_container, software, framerate,
                                 annotation_classes, working_dir, prefix)
                if st.button('Extract Features'):
                    extractor = Extract(working_dir, prefix, frames2integ, is_3d)
                    extractor.main()
            except FileNotFoundError:
                st.info(SPLIT_PROJECT_HELP)
        st.session_state['page'] = 'Step 3'

    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
