import os

import numpy as np
import streamlit as st
from config.help_messages import INIT_RATIO_HELP, MAX_ITER_HELP, MAX_SAMPLES_HELP, \
    SHOW_FINAL_RESULTS_HELP, RE_CLASSIFY_HELP, IMPRESS_TEXT, NO_CONFIG_HELP, \
    NO_FEATURES_HELP
from utils.auto_active_learning import show_classifier_results, RF_Classify
from utils.load_workspace import load_features, load_heldout, \
    load_iter0, load_iterX, load_all_train
from utils.project_utils import update_config

TITLE = "Active learning"


def prompt_setup(software, train_fx, working_dir, prefix, iteration_dir, exclude_other, annotation_classes):
    project_dir = os.path.join(working_dir, prefix)
    [_, targets, _, _] = load_features(project_dir, iteration_dir)
    # if software == 'CALMS21 (PAPER)':
    #     ROOT = Path(__file__).parent.parent.parent.resolve()
    #     targets_test_csv = os.path.join(ROOT.joinpath("test"), './test_labels.csv')
    #     targets_test_df = pd.read_csv(targets_test_csv, header=0)
    #     targets_test = np.array(targets_test_df['annotation'])
    # else:
        #features_heldout, targets_heldout = load_heldout(working_dir, prefix)
        # targets_test = np.hstack(load_test_targets(working_dir, prefix))
    # features_heldout, targets_heldout = load_heldout(working_dir, prefix)
    col1, col2 = st.columns(2)

    if exclude_other:
        #this will fail if other has no labels and exclude the class before that instead (without warning)
        #label_code_other = max(np.unique(np.hstack(targets_runlist)))
        # data_samples_per = [np.mean([len(np.where(targets_runlist[i] == be)[0])
        #                              for i in range(len(targets_runlist))]) for be in np.unique(targets_runlist) if
        #                     be != label_code_other]
        #exclude other by position
        selected_class_num = np.arange((len(annotation_classes)))[:-1]
    else:
        #data_samples_per = [np.mean([len(np.where(targets_runlist[i] == be)[0])
        #                             for i in range(len(targets_runlist))]) for be in np.unique(targets_runlist)]
        selected_class_num = np.arange((len(annotation_classes)))

    data_samples_per = [len(np.where(targets == be)[0]) for be in np.arange((len(selected_class_num)))]
    # if I want 10 samples as minimum, what do I need to put as min_ratio?
    min_samples  = 10
    smallest_class_num = np.min(data_samples_per)
    min_ratio_ = np.round(min_samples/smallest_class_num, 2)
    max_samps_iter = np.ceil(len(data_samples_per)*10).astype(int)

    if not np.all(data_samples_per):
        #if any selected class has no labels in the dataset, throw an error (very rare cases).
        st.error( "Some of selected classes have no available labels! Return back to the data upload and deselect any annotation classes that have no label."\
                  )
        if not exclude_other:
            st.warning("If this is a problem with 'other', you can exclude 'other' in the config or by recreating the project.")

    col1_exp = col1.expander('Initial sampling ratio'.upper(), expanded=True)
    col2_exp = col2.expander('Max number of iterations'.upper(), expanded=True)
    col2_bot_exp = col2.expander('Samples per iteration'.upper(), expanded=True)

    if 'init_ratio' not in st.session_state:
        st.session_state.init_ratio = min_ratio_
        init_ratio = col1_exp.number_input("Select an initial sampling ratio",
            min_value=0.0, max_value=1.0, value=min_ratio_, key='init3', help = INIT_RATIO_HELP)

    else:
        init_ratio = col1_exp.number_input(
            "Select an initial sampling ratio",
            min_value=0.0, max_value=1.0, value=st.session_state.init_ratio, key='init3', help = INIT_RATIO_HELP)
        st.session_state.init_ratio = init_ratio
    # give user info about samples
    info_text = [f"{annotation_classes[i]} [{round(data_samples_per[i] * st.session_state.init_ratio, 2)}]"
                 for i in range(len(data_samples_per))]
    col1_exp.info('On average, samples to train per class: \n\n' + ";  ".join(info_text))
    max_iter = col2_exp.number_input('Max number of self-learning iterations',
                                     min_value=1, max_value=None, value=100, key='maxi3', help = MAX_ITER_HELP)
    max_samples_iter = col2_bot_exp.number_input(f'Max samples amongst the '
                                                 f'{len(data_samples_per)} classes',
                                                 min_value=10, max_value=None, value=max_samps_iter,
                                                 key='maxs3', help = MAX_SAMPLES_HELP)

    parameters_dict = {
        "Processing": dict(
            TRAIN_FRACTION = init_ratio,
            MAX_ITER = max_iter,
            MAX_SAMPLES_ITER=max_samples_iter,
        )
    }
    st.session_state['config'] = update_config(os.path.join(working_dir,prefix),updated_params=parameters_dict)

    # return init_ratio, max_iter, max_samples_iter, targets_test
    return init_ratio, max_iter, max_samples_iter


def main(ri=None, config=None):
    st.markdown("""---""")

    if config is not None:

        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
        train_fx = config["Processing"].getfloat("TRAIN_FRACTION")
        conf_threshold = config["Processing"].getfloat("CONF_THRESHOLD")
        iteration = config["Processing"].getint("ITERATION")
        selected_iter = ri.selectbox('select iteration number', np.arange(iteration+1))
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)

        if conf_threshold is None:
            # backwards compatability
            conf_threshold = 0.5

        try:
            [_, _, _, all_f1_scores, _, _] = \
                load_all_train(project_dir, iter_folder)
            [_, _, iter0_Y_train, iter0_f1_scores, _, _] = \
                load_iter0(project_dir, iter_folder)
            [_, _, iterX_Y_train_list, iterX_f1_scores, _, _] = \
                load_iterX(project_dir, iter_folder)
            if st.checkbox('Show final results', key='sr', value=True, help = SHOW_FINAL_RESULTS_HELP):
                show_classifier_results(annotation_classes, all_f1_scores,
                                        iter0_f1_scores, iter0_Y_train,
                                        iterX_f1_scores, iterX_Y_train_list)
            message_container = st.container()
            redo_container = st.container()
            if not redo_container.checkbox('Re-classify behaviors', help = RE_CLASSIFY_HELP):
                message_container.success(f'This prefix had been classified.')
            else:
                init_ratio, max_iter, max_samples_iter = \
                    prompt_setup(software, train_fx, working_dir, prefix, iter_folder, exclude_other, annotation_classes)
                if st.button('re-classify'.upper()):
                    rf_classifier = RF_Classify(working_dir, prefix, iter_folder, software,
                                                init_ratio, max_iter, max_samples_iter,
                                                annotation_classes, exclude_other, conf_threshold)
                    rf_classifier.main()
        except FileNotFoundError:
            # make sure the features were extracted:
            try:
                init_ratio, max_iter, max_samples_iter = \
                    prompt_setup(software, train_fx, working_dir, prefix, iter_folder, exclude_other, annotation_classes)
                if st.button('classify'.upper()):
                    rf_classifier = RF_Classify(working_dir, prefix, iter_folder, software,
                                                init_ratio, max_iter, max_samples_iter,
                                                annotation_classes, exclude_other, conf_threshold)
                    rf_classifier.main()
            except FileNotFoundError:
                st.error(NO_FEATURES_HELP)
        st.session_state['page'] = 'Step 4'
    else:
        st.error(NO_CONFIG_HELP)
    bottom_cont = st.container()
    with bottom_cont:

        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
