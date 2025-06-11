import os

import numpy as np
import streamlit as st
from asoid.config.help_messages import *
from asoid.utils.auto_active_learning import show_classifier_results, RF_Classify, get_available_conf_options
from asoid.utils.load_workspace import load_features, load_heldout, \
    load_iter0, load_iterX, load_all_train
from asoid.utils.project_utils import update_config

TITLE = "Active learning"
ACTIVE_LEARNING_HELP = ("In this step, you will train a classifier using a small set of labeled data and then small portions of the remaining training data are fed to the classifier for several iterations."
                        "\n\n The samples are selected based on the classifier's confidence in its predictions. This process reaches high performance with limited labeled data."
                        "\n\n :blue[The parameters **Initial sampling ratio**, **Max number of self-learning iterations**, and **Max samples amongst the X classes** can be adjusted to control the active learning process.]"
                        "\n\n---\n\n"
                        "**Step 1**: Select an iteration."
                        "\n\n **Step 2**: Set the parameters."
                        "\n\n **Step 3**: Train the classifier."
                        "\n\n **Step 4**: View the results :blue[(might require a refresh by pressing 'R' on your keyboard)]."
                        "\n\n **Step 5**: Refine the parameters or move to the next step :orange[Predict], :orange[Discover], or :orange[Refine Behaviors]."
                        "\n\n---\n\n"
                        ":blue[This classifier can be directly used in the prediction and discovery steps. Alternatively, you can refine the classifier by adding more unlabeled data in the next step:] :orange[Refine Behaviors]")

def prompt_setup(software, train_fx, conf, conf_type,
                 working_dir, prefix, iteration_dir, exclude_other, annotation_classes):
    project_dir = os.path.join(working_dir, prefix)
    [_, targets, _] = load_features(project_dir, iteration_dir)

    col1, col2 = st.columns(2)
    if exclude_other:
        selected_class_num = np.arange((len(annotation_classes)))[:-1]
    else:
        selected_class_num = np.arange((len(annotation_classes)))
    data_samples_per = [len(np.where(targets == be)[0]) for be in np.arange((len(selected_class_num)))]
    # if I want 10 samples as minimum, what do I need to put as min_ratio?
    min_samples = 10
    smallest_class_num = np.min(data_samples_per)
    min_ratio_ = np.round(min_samples / smallest_class_num, 2)
    max_samps_iter = np.ceil(len(data_samples_per) * 100).astype(int)

    if not np.all(data_samples_per) or smallest_class_num < min_samples:
        # if any selected class has no labels in the dataset, throw an error (very rare cases).
        st.error(
            "Some of selected classes have not enough available labels! Min. samples per class: " + str(min_samples) +". "
            "Return back to the data upload and deselect any annotation classes that have not enough labels."
            )
        st.warning("Samples per class:" + str([*zip(annotation_classes, data_samples_per)]))
        if not exclude_other:
            st.warning(
                "If this is a problem with 'other', you can exclude 'other' in the config or by recreating the project.")
        st.stop()
    col1_exp = col1.expander('Initial sampling ratio'.upper(), expanded=True)
    col2_exp = col2.expander('Max number of iterations'.upper(), expanded=True)
    col2_bot_exp = col2.expander('Samples per iteration'.upper(), expanded=True)

    if 'init_ratio' not in st.session_state:
        st.session_state.init_ratio = min_ratio_
        init_ratio = col1_exp.number_input("Select an initial sampling ratio",
                                               min_value=min_ratio_, max_value=1.0,
                                               value=min_ratio_, key='init3', help=INIT_RATIO_HELP)
    else:
        init_ratio = col1_exp.number_input(
            "Select an initial sampling ratio",
            min_value=0.0, max_value=1.0, value=st.session_state.init_ratio, key='init3', help=INIT_RATIO_HELP)
        st.session_state.init_ratio = init_ratio
    # give user info about samples
    info_text = [f"{annotation_classes[i]} [{round(data_samples_per[i] * st.session_state.init_ratio * 0.8, 2)}]"
                 for i in range(len(data_samples_per))]
    col1_exp.info('Initial samples to train per class: \n\n' + ";  ".join(info_text))
    max_iter = col2_exp.number_input('Max number of self-learning iterations',
                                     min_value=1, max_value=None,
                                     value=50, key='maxi3', help=MAX_ITER_HELP)
    max_samples_iter = col2_bot_exp.number_input(f'Max samples amongst the '
                                                 f'{len(data_samples_per)} classes',
                                                 min_value=max_samps_iter, max_value=None,
                                                 value=max_samps_iter,
                                                 key='maxs3', help=MAX_SAMPLES_HELP)

    
    available_conf_types = get_available_conf_options()
    #turn dict into nice string in markdown
    conf_types_str = "\n".join([f"{k}: {available_conf_types[k]['description']} \n" for k in available_conf_types.keys()])
    CONFIDENCE_TYPE_HELP = ("Confidence type to use for the active learning regime (low-confidence/uncertainty sampling). \n\n" \
                            "The confidence type is used to determine the confidence of the classifier in its predictions. \n" \
                            "The probability ouput of clf.predict_proba(X) is used to calculate the confidence. See \n\n" \
                            "Available types: \n\n" \
                            + conf_types_str)

    if conf_type not in get_available_conf_options():
            col2_bot_exp.error(f"Invalid confidence calculation option found in config: {conf_type}. Defaulting to 'max'.")
            conf_type = 'max'

    
    sel_conf_type = col2_bot_exp.selectbox('Confidence type'
                            , available_conf_types.keys()
                            , index = int(list(available_conf_types.keys()).index(conf_type))
                            , help = CONFIDENCE_TYPE_HELP
                            )
    
    if st.session_state['conf_type'] != sel_conf_type:
        # get the default confidence threshold for the selected confidence type
        conf = available_conf_types[sel_conf_type]["default_thresh"]
    

    st.session_state['conf_threshold'] = col2_bot_exp.number_input('Confidence threshold',
                                               min_value=0.05, max_value=0.95,
                                               value=conf
                                               ,help = CONFIDENCE_THRESHOLD_HELP)
    parameters_dict = {
        "Processing": dict(
            TRAIN_FRACTION=init_ratio,
            MAX_ITER=max_iter,
            MAX_SAMPLES_ITER=max_samples_iter,
            CONF_TYPE=sel_conf_type,
            CONF_THRESHOLD=st.session_state['conf_threshold']
        )
    }
    st.session_state['config'] = update_config(os.path.join(working_dir, prefix), updated_params=parameters_dict)
    st.session_state['conf_type'] = sel_conf_type

    return init_ratio, max_iter, max_samples_iter


def main(ri=None, config=None):
    st.markdown("""---""")
    st.title("Active Learning")
    st.expander("What is this?", expanded=False).markdown(ACTIVE_LEARNING_HELP)


    if config is not None:

        working_dir = config["Project"].get("PROJECT_PATH")
        prefix = config["Project"].get("PROJECT_NAME")
        annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        software = config["Project"].get("PROJECT_TYPE")
        exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
        train_fx = config["Processing"].getfloat("TRAIN_FRACTION")
        #TODO: needs failsave for older projects
        try:
            conf_type = config["Processing"].get("CONF_TYPE")
        except KeyError:
            conf_type = get_available_conf_options().keys()[0]
        conf = config["Processing"].getfloat("CONF_THRESHOLD")
        iteration = config["Processing"].getint("ITERATION")
        selected_iter = ri.selectbox('Select Iteration #', np.arange(iteration + 1), iteration)
        project_dir = os.path.join(working_dir, prefix)
        iter_folder = str.join('', ('iteration-', str(selected_iter)))
        os.makedirs(os.path.join(project_dir, iter_folder), exist_ok=True)

        if conf is None:
        #     # backwards compatability
            conf = 0.5
        if 'conf_threshold' not in st.session_state:
            st.session_state['conf_threshold'] = None
        if 'conf_type' not in st.session_state:
            st.session_state['conf_type'] = conf_type

        try:
            [all_f1_scores] = \
                load_all_train(project_dir, iter_folder)
            [iter0_Y_train, iter0_f1_scores] = \
                load_iter0(project_dir, iter_folder)
            [_, iterX_Y_train_list, iterX_f1_scores] = \
                load_iterX(project_dir, iter_folder)
            if st.checkbox('Show final results', key='sr', value=True, help=SHOW_FINAL_RESULTS_HELP):
                show_classifier_results(annotation_classes, all_f1_scores,
                                        iter0_f1_scores, iter0_Y_train,
                                        iterX_f1_scores, iterX_Y_train_list)
            message_container = st.container()
            redo_container = st.container()
            if not redo_container.checkbox('Re-classify behaviors', help=RE_CLASSIFY_HELP):
                message_container.success(f'This prefix had been classified.')
            else:
                init_ratio, max_iter, max_samples_iter = \
                    prompt_setup(software, train_fx, conf, conf_type, working_dir, prefix, iter_folder, exclude_other,
                                 annotation_classes)
                # st.write(conf_threshold)
                if st.button('Train Classifier'):
                    rf_classifier = RF_Classify(working_dir, prefix, iter_folder, software,
                                                init_ratio
                                                , max_iter
                                                , max_samples_iter
                                                , annotation_classes
                                                , exclude_other
                                                , st.session_state['conf_type']
                                                , st.session_state['conf_threshold'])
                    rf_classifier.main()
                    col_left, _, col_right = st.columns([1, 1, 1])
                    col_right.success("Continue on with next module".upper())
        except FileNotFoundError:
            # make sure the features were extracted:

            try:
                init_ratio, max_iter, max_samples_iter = \
                    prompt_setup(software, train_fx, conf, conf_type, working_dir, prefix, iter_folder, exclude_other,
                                 annotation_classes)
                if st.button('Train Classifier'):
                    rf_classifier = RF_Classify(working_dir, prefix, iter_folder, software
                                                , init_ratio
                                                , max_iter
                                                , max_samples_iter
                                                , annotation_classes
                                                , exclude_other
                                                , st.session_state['conf_type']
                                                , st.session_state['conf_threshold'])
                    rf_classifier.main()
                    col_left, _, col_right = st.columns([1, 1, 1])
                    col_right.success("Continue on with next module".upper())
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
