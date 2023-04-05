import streamlit as st
import numpy as np
from app import swap_app
import os
from utils.project_utils import update_config
from utils.load_workspace import load_data, load_features
from utils.extract_features import Extract, interactive_durations_dist
from config.help_messages import SPLIT_CLASSES_HELP, BINS_SLIDER_HELP, MIN_DURATION_HELP, CALM_HELP, NUM_SPLITS_HELP,\
                                RE_EXTRACT_HELP, EXTRACT_FEATURES_HELP, IMPRESS_TEXT, NO_CONFIG_HELP

import categories

CATEGORY = categories.EXTRACT_FEATURES
TITLE = "Extract Features"


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
        col_left, col_right = prompt_container.columns(2)
        left_exp = col_left.expander('Minimum Duration', expanded=True)
        right_exp = col_right.expander('Number of splits', expanded=True)
        if not software == 'CALMS21 (PAPER)':
            duration_min = left_exp.number_input('Minimum duration (s) of behavior before transition:',
                                                 min_value=0.05, max_value=None,
                                                 value=0.1, key='fr',
                                                 help = MIN_DURATION_HELP)
            num_splits = right_exp.number_input('number of shuffled splits:', min_value=1, max_value=20,
                                                value=10, key='ns',
                                                help = NUM_SPLITS_HELP)
        else:
            duration_min = left_exp.number_input('Minimum duration (s) of behavior before transition:',
                                                 min_value=0.05, max_value=None,
                                                 value=0.4, key='fr',
                                                 help = MIN_DURATION_HELP+ CALM_HELP)
            num_splits = right_exp.number_input('number of shuffled splits:', min_value=1, max_value=20,
                                                value=10, key='ns',
                                                help = NUM_SPLITS_HELP+ CALM_HELP)
        frames2integ = round(framerate * (duration_min / 0.1))

        #update config (only the offline version)

        parameters_dict = {
            "Processing": dict(
                N_SHUFFLED_SPLIT=num_splits,
                MIN_DURATION=duration_min
            )
        }
        update_config(os.path.join(working_dir, prefix),updated_params=parameters_dict)

    return frames2integ, num_splits


def main(config=None):
    st.markdown("""---""")

    if config is not None:
        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        framerate = config["Project"].getfloat("FRAMERATE")
        try:
            [_, _, _, _] = load_features(working_dir, prefix)
            prompt_container = st.container()
            message_container = st.container()
            redo_container = st.container()
            if not redo_container.checkbox('Re-extract features', help = RE_EXTRACT_HELP):
                frames2integ, num_splits = \
                    prompt_setup(prompt_container, software, framerate, annotation_classes,
                                 working_dir, prefix, show_only=True)
                message_container.success(f'This prefix had been extracted.')
            else:
                frames2integ, num_splits = \
                    prompt_setup(prompt_container, software, framerate, annotation_classes,
                                 working_dir, prefix)
                if st.button('EXTRACT FEATURES'.upper(), help = EXTRACT_FEATURES_HELP):
                    extractor = Extract(working_dir, prefix, frames2integ, num_splits)
                    extractor.main()
        except FileNotFoundError:
            prompt_container = st.container()
            frames2integ, num_splits = \
                prompt_setup(prompt_container, software, framerate,
                             annotation_classes, working_dir, prefix)
            if st.button('EXTRACT FEATURES'.upper()):
                extractor = Extract(working_dir, prefix, frames2integ, num_splits)
                extractor.main()

    else:
        st.error(NO_CONFIG_HELP)

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns([3, 3, 1, 1, 1])
        if button_col1.button('◀  PRIOR STEP'):
            swap_app('A-data-preprocess')
        if button_col5.button('NEXT STEP ▶'):
            swap_app('C-auto-active-learning')
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
