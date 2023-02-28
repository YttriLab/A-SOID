import glob
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils.import_data import load_pose, get_bodyparts, get_animals, load_labels
from utils.project_utils import create_new_project, update_config
from config.help_messages import POSE_ORIGIN_SELECT_HELP, FPS_HELP, MULTI_ANIMAL_HELP, MULTI_ANIMAL_SELECT_HELP,\
    BODYPART_SELECT, WORKING_DIR_HELP,PREFIX_HELP, DATA_DIR_IMPORT_HELP, POSE_DIR_IMPORT_HELP, POSE_ORIGIN_HELP,\
    POSE_SELECT_HELP, LABEL_DIR_IMPORT_HELP, LABEL_ORIGIN_HELP, LABEL_SELECT_HELP, PREPROCESS_HELP, EXCLUDE_OTHER_HELP,\
    INIT_CLASS_SELECT_HELP



def convert_data_format(data, train=False):
    data_dict = dict(enumerate(data.flatten(), 1))
    pose_estimates = [data_dict[1]['annotator-id_0'][j]['keypoints']
                      for j in list(data_dict[1]['annotator-id_0'].keys())]
    if train:
        targets = [data_dict[1]['annotator-id_0'][j]['annotations']
                   for j in list(data_dict[1]['annotator-id_0'].keys())]
    else:
        targets = None

    keypoint_names = ['nose', 'ear_left', 'ear_right', 'neck', 'hip_left', 'hip_right', 'tail_base']
    keypoints_idx = pd.MultiIndex.from_product([['resident', 'intruder'], keypoint_names, list('xy')],
                                               names=['animal', 'keypoints', 'coords'])
    collection = []
    for i in range(len(pose_estimates)):
        single_sequence = pose_estimates[i]
        single_keypoints_2d = single_sequence.reshape(single_sequence.shape[0],
                                                      single_sequence.shape[1] *
                                                      single_sequence.shape[2] *
                                                      single_sequence.shape[3],
                                                      order='F')
        # convert to dataframe
        df_single_keypoints = pd.DataFrame(single_keypoints_2d, columns=keypoints_idx)
        pose = [0, 2, 12, 14, 16, 18, 20, 22, 24, 26, 1, 3, 13, 15, 17, 19, 21, 23, 25, 27]
        collection.append(np.array(df_single_keypoints.iloc[:, pose]))

    if targets:
        return collection, targets
    else:
        return collection

def select_software():
    """
    Ask for software
    :return:
    """
    software = st.selectbox('Select the type of pose estimation file:',
                                 ('DeepLabCut', 'SLEAP', "CALMS21 (PAPER)"),
                                 help = POSE_ORIGIN_SELECT_HELP)
    if software == 'DeepLabCut':
        # TODO: Add functionality to deal with H5 files from DLC
        # self.ftype = st.selectbox('Select the type of pose estimation file:',
        #                      ('csv', 'h5'))
        ftype = 'csv'

    if software == 'SLEAP':
        ftype = 'h5'

    if software == 'CALMS21 (PAPER)':
        ftype = 'npy'

    return software, ftype



class Preprocess:

    def __init__(self):
        self.input_datafiles = []
        self.input_datafiles_test = []
        self.input_labelfiles = []
        self.input_labelfiles_test = []
        self.processed_input_data = []
        self.processed_input_data_test = []
        self.label_df = []
        self.targets = []
        self.targets_test = []
        self.software = None
        self.ftype = None
        self.label_ftype = "csv"
        self.input_videos = None
        self.label_csvs = None
        self.pose_csvs = None
        self.pose_data_directories = None
        self.label_data_directories = None
        self.data_directories_test = None
        self.selected_animals = []
        self.selected_animal_idx = []
        self.selected_bodyparts = []
        self.selected_pose_idx = []

        self.framerate = None
        self.resolution = None
        self.classes = None
        self.multi_animal = False

        self.working_dir = None
        self.prefix = None

        #for CALMS21 data only
        self.train_data_path = None
        self.test_data_path = None

    def select_software(self):
        self.software, self.ftype = select_software()

    def get_config_params(self):
        """
        Ask for framerate, video resolution, and behavior names
        :return:
        """
        input_container = st.container()
        if not self.software == 'CALMS21 (PAPER)':
            self.framerate = int(
                input_container.number_input('Enter the average video frame-rate of your pose estimation files.',
                                             value=30,
                                             help = FPS_HELP))
            try:
                for i in range(len(self.label_csvs)):
                    # load all label files and upsample them to fit pose estimation
                    # (WARNING: This only works with sample rates that are convertible into each other)
                    # TODO: Adapt to other input type
                    self.label_df.append(load_labels(self.label_csvs[i], origin="BORIS", fps=self.framerate))
                # go through all label files to make sure to catch all optional classes
                # There is probably a faster solution...
                optional_classes = []
                for l_df in self.label_df:
                    # get all optional classes for this file
                    temp_optional_classes = list(l_df.drop(columns=["time"], errors="ignore").columns)
                    # go through all classes
                    for temp_optional_class in temp_optional_classes:
                        # check if it's already in
                        if temp_optional_class not in optional_classes:
                            # if not append
                            optional_classes.append(temp_optional_class)



                self.classes = input_container.multiselect(
                    "Deselect classes that should not be included in training."
                    , optional_classes, optional_classes, help = INIT_CLASS_SELECT_HELP
                )

                #move other to last place
                if "other" in optional_classes:
                    optional_classes.append(optional_classes.pop(optional_classes.index("other")))

                self.exclude_other = input_container.checkbox("Exclude 'other'?", False,
                                                              key= "exclude_other_check",
                                                              help= EXCLUDE_OTHER_HELP)
            except IndexError:
                st.warning('Please select corresponding label files first.')
            except TypeError:
                st.warning('Please select corresponding label files first.')

            self.multi_animal = st.checkbox("Is this a multiple animal project?",
                                            False, key="multi_animal_check",
                                            help = MULTI_ANIMAL_HELP)
        else:
            self.framerate = 30
            self.classes = ['attack', 'investigation', 'mount', 'other']
            self.multi_animal = True
            self.selected_bodyparts = [
                "nose",
                "neck",
                "hip_left",
                "hip_right",
                "tail_base",
            ]
            self.selected_animals = ['resident', 'intruder']
        if self.framerate is not None:
            st.success('You have selected **{}** *frames/second*.'.format(self.framerate))
        if self.classes is not None:
            st.success(
                'Selected classes:* **{}**.'.format(', '.join(self.classes)))
        if self.selected_bodyparts:
            st.success("**Selected keypoints/bodyparts**: " + ", ".join(self.selected_bodyparts))
        if self.selected_animals:
            st.success("**Selected individuals/animals**: " + ", ".join(self.selected_animals))
        try:

            file0_df = load_pose(self.pose_csvs[0], origin=self.software.lower(), multi_animal=self.multi_animal)

            if self.multi_animal:
                if self.software.lower() == "deeplabcut":
                    # if it's multi animal, we take bodyparts from a level below
                    animal_lvl = 1
                    bp_level = 2
                elif self.software.lower() == "sleap":
                    # sleap converted files don't have a scorer level
                    animal_lvl = 0
                    bp_level = 1
                animals = get_animals(file0_df, lvl=animal_lvl)
                self.selected_animals = st.multiselect('Identified animals to include:', animals,
                                                       animals
                                                       ,help = MULTI_ANIMAL_SELECT_HELP)
                # find the indexes where the animal has bps
                an_index_list = []
                for an in self.selected_animals:
                    an_index = np.argwhere(file0_df.columns.get_level_values(animal_lvl) == an)
                    an_index_list.append(an_index)
                self.selected_animal_idx = np.sort(np.array(an_index_list).flatten())
                st.success("**Selected individuals/animals**: " + ", ".join(self.selected_animals))
            else:
                bp_level = 1
            bodyparts = get_bodyparts(file0_df, bp_level)
            # file0_df = pd.read_csv(data_files[0],low_memory=False)
            self.selected_bodyparts = st.multiselect('Identified keypoints/bodyparts to include:', bodyparts,
                                                     bodyparts,
                                                     help = BODYPART_SELECT)
            bp_index_list = []
            for bp in self.selected_bodyparts:
                bp_index = np.argwhere(file0_df.columns.get_level_values(bp_level) == bp)
                # index = [i for i,s in enumerate(file0_array[0,1:]) if a in s]
                if self.multi_animal:
                    # if it's multiple animal project, the user has the option to subselect individual animals
                    # therefore we need to make sure that the bp indexes are only taken if they correspond to the selected animals
                    bp_index = np.array([idx for idx in bp_index if idx in self.selected_animal_idx])
                bp_index_list.append(bp_index)
            self.selected_pose_idx = np.sort(np.array(bp_index_list).flatten())
            st.success("**Selected keypoints/bodyparts**: " + ", ".join(self.selected_bodyparts))
        except:
            pass
        st.info("The parameters can be changed in the config file later on.")

    def select_working_dir(self):
        input_container = st.container()
        # TODO: adapt to run with mac and ubuntu
        self.working_dir = input_container.text_input('Enter a working directory',
                                                      str.join('', (str(Path.home()), '/Desktop/asoid_output')),
                                                      help = WORKING_DIR_HELP)
        try:
            os.listdir(self.working_dir)
            st.success('Entered **{}** as the working directory.'.format("%r" % self.working_dir))
        except FileNotFoundError:
            if st.button('create output folder'):
                os.makedirs(self.working_dir, exist_ok=True)
                st.experimental_rerun()
        today = date.today()
        d4 = today.strftime("%b-%d-%Y")

        self.prefix = input_container.text_input('Enter filename prefix', d4,
                                                 help = PREFIX_HELP)
        if self.prefix:
            st.success(f'Entered **{self.prefix}** as the prefix.')
        else:
            st.error('Please enter a prefix.')

    def setup_project(self):
        col1, col2 = st.columns([1, 1])
        col1_exp = col1.expander('TRAIN', expanded=True)
        col2_exp = col2.expander('CONFIG', expanded=True)
        col3_exp = col1.expander('SAVE', expanded=True)
        with col1_exp:
            self.select_software()
            upload_container = st.container()
            if not self.software == 'CALMS21 (PAPER)':
                #upload_container = st.container()
                if upload_container.checkbox("Use folder import", help = DATA_DIR_IMPORT_HELP):
                    #find our pose and label files automatically
                    #self.select_data_directories()

                    self.pose_data_directories = upload_container.text_input('Select the directory containing the pose estimation files'
                                                                            ,os.getcwd()
                                                                            ,help = POSE_DIR_IMPORT_HELP)
                    try:
                        os.listdir(self.pose_data_directories)
                        upload_container.success(
                            'You have selected **{}** as your _data directory_'.format("%r" % self.pose_data_directories))

                        if self.ftype is not None:
                            found_pose_files = glob.glob(os.path.join(self.pose_data_directories,"*.{}".format(self.ftype)))
                            if len(found_pose_files) == 0:
                                upload_container.warning(
                                    "Make sure that the pose estimation files have the correct file type (*.{}).".format(
                                        self.ftype))
                        # make it a dictionary for later sorting
                        self.pose_files = {os.path.basename(x): x for x in found_pose_files}

                        if self.pose_files:
                            self.pose_csvs = upload_container.multiselect(
                                'Order them to match the sequence of label files below'
                                ,self.pose_files.keys(),self.pose_files.keys()
                                ,help= POSE_SELECT_HELP
                            )
                            # retrieve files in right order:
                            self.pose_csvs = [self.pose_files[x] for x in self.pose_csvs]
                        upload_container.write('---')

                    except FileNotFoundError:
                        upload_container.error('No such directory')


                    self.label_data_directories = upload_container.text_input('Select the directory containing the pose estimation files'
                                                                              , os.getcwd()
                                                                              , help = LABEL_DIR_IMPORT_HELP + LABEL_ORIGIN_HELP)
                    # do the same for labels
                    try:
                        os.listdir(self.label_data_directories)
                        upload_container.success('You have selected **{}** as your _data directory_'.format(
                                "%r" % self.label_data_directories))

                        found_label_files = glob.glob(os.path.join(self.label_data_directories,"*.csv"))
                        if len(found_label_files) == 0:
                            upload_container.warning(
                                "Make sure that the label files have the correct file type (*.csv).")
                        # make it a dictionary for later sorting
                        self.label_files = {os.path.basename(x): x for x in found_label_files}
                        if self.label_files:
                            self.label_csvs = upload_container.multiselect(
                                'Order them to match the sequence of label files below'
                                ,self.label_files.keys(),self.label_files.keys()
                                ,help = LABEL_SELECT_HELP
                                )
                            # retrieve files in right order:
                            self.label_csvs = [self.label_files[x] for x in self.label_csvs]
                        upload_container.write('---')

                    except FileNotFoundError:
                        upload_container.error('No such directory')

                else:
                    #Use file uploader

                    self.pose_files = upload_container.file_uploader('Upload corresponding pose files',
                                                                     accept_multiple_files=True,
                                                                     type=self.ftype, key='pose'
                                                                     ,help = POSE_ORIGIN_HELP)
                    # make it a dictionary for later sorting
                    self.pose_files = {x.name: x for x in self.pose_files}

                    self.pose_csvs = upload_container.multiselect(
                        'Order them to match the sequence of label files below'
                        , self.pose_files.keys()
                        , help = POSE_SELECT_HELP)
                    # retrieve files in right order:
                    self.pose_csvs = [self.pose_files[x] for x in self.pose_csvs]
                    upload_container.write('---')
                    # do the same for labels
                    self.label_files = upload_container.file_uploader(
                        'Upload corresponding annotation files',
                        accept_multiple_files=True,
                        type=self.label_ftype,
                        key='label',
                        help= LABEL_ORIGIN_HELP)
                    self.label_files = {x.name: x for x in self.label_files}
                    self.label_csvs = upload_container.multiselect(
                        'Order them to match the sequence of pose files above'
                        , self.label_files.keys()
                        , help = LABEL_SELECT_HELP)
                    self.label_csvs = [self.label_files[x] for x in self.label_csvs]

            else:
                #for CALMS21 we need the test and train files directly
                self.train_data_path = upload_container.text_input(
                    'Select the full path for the CalMS21 train file (calms21_task1_train.npy)'
                    , os.getcwd())

                self.test_data_path = upload_container.text_input(
                    'Select the full path for the CalMS21 test file (calms21_task1_test.npy)'
                    , os.getcwd())

        with col2_exp:
            self.get_config_params()
        with col3_exp:
            self.select_working_dir()

    def _convert_labels(self, i):
        """converts label file into vector with series of numbered codes representing selected classes.
        Deselected classes are removed automatically and considered as 'other'. Unlabeled data is labeled as 'other'.
        Finally 'other' is sorted to the end of the classes, so that it is always the last class"""

        # get all available classes in this label file
        available_classes = list(self.label_df[i].drop(columns=["time"], errors="ignore").columns)

        # find classes that are not in the initial selection
        deselected_classes = [x for x in available_classes if
                              x not in self.classes]
        # for safety, let's make a copy to not change the original data
        # fixed critical error
        select_label_df = self.label_df[i].drop(columns=["time"], errors="ignore").copy(deep=True)
        # all deselected classes are turned to "0" and will be regarded as unlabeled/ "other"
        select_label_df[deselected_classes] = 0
        # find all unlabeled
        unlabeled_data = select_label_df.sum(axis=1) == 0
        # fixed critical error
        select_label_df.drop(columns=deselected_classes, inplace=True)
        # create a new column called "other" if not existent yet
        if "other" not in select_label_df.columns:
            select_label_df["other"] = 0

        # change all unlabeled to 1 in other
        select_label_df.loc[unlabeled_data, "other"] = 1
        # drop all classes that are deselected
        # select_label_df.drop(columns=deselected_classes, inplace=True)
        # make sure that other is one of the selected classes
        #if "other" not in self.classes and not self.exclude_other:

        if "other" not in self.classes:
            self.classes.append("other")

        # make sure that all classes are in (irrelevant for most cases):
        for selected_class in self.classes:
            if selected_class not in select_label_df.columns:
                # adding a column
                select_label_df[selected_class] = 0

        # rearrange so that "other" is always last and classes are in the order they were selected:
        select_label_df = select_label_df.reindex(
            columns=(list([clm for clm in self.classes if clm != "other"]) + ["other"]))

        #get rid of "other" column if excluded
        # if self.exclude_other:
        #     select_label_df.drop(columns=["other"], inplace=True)

        # convert dummie encoding to numbers but keep also identity of classes not represented in this file
        label_vector = np.argmax(np.array(select_label_df), axis=1)

        return label_vector

    def compile_data(self):
        if st.button("PREPROCESS", help = PREPROCESS_HELP):
            self.input_datafiles = []
            self.input_labelfiles = []
            self.processed_input_data = []
            if self.software == "CALMS21 (PAPER)" and self.ftype == "npy":
                # ROOT = Path(__file__).parent.parent.parent.resolve()
                # filenames = glob.glob(str(ROOT.joinpath("train")) + '/*.npy')[0]
                #train = np.load(os.path.join(ROOT.joinpath("train"), filenames), allow_pickle=True)
                train = np.load(self.train_data_path, allow_pickle=True)
                self.processed_input_data, self.targets = convert_data_format(train, train=True)
                self.input_datafiles.append(self.train_data_path)
                self.input_labelfiles.append(self.train_data_path)
                # filenames_test = glob.glob(str(ROOT.joinpath("test")) + '/*.npy')[0]
                # test = np.load(os.path.join(ROOT.joinpath("test"), filenames_test), allow_pickle=True)
                test = np.load(self.test_data_path, allow_pickle=True)
                self.processed_input_data_test = convert_data_format(test)
                self.input_datafiles_test.append(self.test_data_path)
                self.input_labelfiles_test.append(self.test_data_path)
            # elif self.software == 'DeepLabCut' and self.ftype == 'csv':
            else:
                # if it's deeplabcut or sleap
                # go through all pose files
                for i, f in enumerate(self.pose_csvs):
                    if self.pose_data_directories is None:
                        # because we cannot be sure whether the file was already used by read_csv,
                        # we need to refresh the buffer!
                        f.seek(0)
                        self.input_datafiles.append(f.name)
                        self.input_labelfiles.append(self.label_csvs[i].name)
                    else:
                        #we used the folder import, which results in a string
                        self.input_datafiles.append(os.path.basename(f))
                        self.input_labelfiles.append(os.path.basename(self.label_csvs[i]))

                    # current_pose = pd.read_csv(self.pose_csvs[i],
                    #                            header=[0, 1, 2], sep=",", index_col=0)
                    current_pose = load_pose(f, origin=self.software.lower(), multi_animal=self.multi_animal)
                    # idx_selected = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
                    # take user selected bodyparts
                    idx_selected = self.selected_pose_idx
                    # get rid of likelihood columns for deeplabcut
                    idx_llh = self.selected_pose_idx[2::3]
                    # the loaded sleap file has them too, so exclude for both
                    idx_selected = [i for i in idx_selected if i not in idx_llh]

                    #train_portion = int(np.array(current_pose.shape[0]) * 0.7)
                    self.processed_input_data.append(np.array(current_pose.iloc[:, idx_selected]))

                    # self.processed_input_data.append(np.array(current_pose.iloc[:train_portion, idx_selected]))
                    # self.processed_input_data_test.append(np.array(current_pose.iloc[train_portion:, idx_selected]))

                    # convert dummie encoding to numbers but keep also identify of classes not represented in this file
                    label_vector = self._convert_labels(i)

                    # continue with partioning
                    targets = label_vector[-current_pose.shape[0]:].copy()
                    #train_portion_labels = int(targets.shape[0] * 0.7)
                    self.targets.append(targets)

                    # self.targets.append(targets[:train_portion_labels])
                    # self.targets_test.append(targets[train_portion_labels:])



            self.save_update_info()

    def save_update_info(self):
        # create new project folder with prefix as name:
        project_folder, _ = create_new_project(self.working_dir, self.prefix, overwrite=True)
        with open(os.path.join(project_folder, 'data.sav'), 'wb') as f:
            """Save data as npy file"""
            # data
            joblib.dump(
                [np.array(self.processed_input_data), np.array(self.targets)]
                , f
            )
        # with open(os.path.join(project_folder, 'test.sav'), 'wb') as f:
        #     """Save data as npy file"""
        #     # data
        #     joblib.dump(
        #         np.array(self.processed_input_data_test), f
        #     )
        # with open(os.path.join(project_folder, 'test_targets.sav'), 'wb') as f:
        #     joblib.dump(
        #         np.array(self.targets_test), f
        #     )

        if not self.software == "CALMS21 (PAPER)":
            # update config with new parameters:
            parameters_dict = {'Data': dict(
                ROOT_PATH=None,
                DATA_INPUT_FILES=self.input_datafiles,
                LABEL_INPUT_FILES=self.input_labelfiles
            ),
                "Project": dict(
                    PROJECT_TYPE=self.software,
                    FILE_TYPE=self.ftype,
                    FRAMERATE=self.framerate,
                    INDIVIDUALS_CHOSEN=self.selected_animals if self.selected_animals else ["single animal"],
                    KEYPOINTS_CHOSEN=self.selected_bodyparts,
                    PROJECT_PATH=self.working_dir,
                    CLASSES=self.classes,
                    MULTI_ANIMAL=self.multi_animal,
                    EXCLUDE_OTHER = self.exclude_other
                ),
                "Processing": dict(
                    ITERATION=0,
                )
            }

        else:
            parameters_dict = {'Data': dict(
                DATA_INPUT_FILES=self.input_datafiles,
                LABEL_INPUT_FILES=self.input_labelfiles,
                TEST_DATA_INPUT_FILES=self.input_datafiles_test,
                TEST_LABEL_INPUT_FILES=self.input_labelfiles_test,

            ),
                "Project": dict(
                    PROJECT_TYPE=self.software,
                    FRAMERATE=self.framerate,
                    INDIVIDUALS_CHOSEN=self.selected_animals,
                    KEYPOINTS_CHOSEN=self.selected_bodyparts,
                    PROJECT_PATH=self.working_dir,
                    CLASSES=self.classes,
                    MULTI_ANIMAL=self.multi_animal
                ),
                "Processing": dict(
                    ITERATION=0,
                )
            }

        update_config(project_folder, updated_params=parameters_dict)

        col_left, _, col_right = st.columns([1, 1, 1])
        col_left.info('Processed a total of **{}** .{} files, and compiled into a '
                      '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                 np.array(self.processed_input_data).shape))
        col_right.success("Continue on with next module".upper())

    def main(self):
        self.setup_project()
        self.compile_data()
