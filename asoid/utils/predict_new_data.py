import os
import streamlit as st
from stqdm import stqdm

import numpy as np
import pandas as pd

from utils.load_preprocess import select_software
from utils.load_workspace import load_iterX, load_features
from utils.extract_features import feature_extraction, feature_extraction_with_extr_scaler,frameshift_predict,bsoid_predict_numba,bsoid_predict_numba_noscale
from utils.import_data import load_pose,get_bodyparts,get_animals
from config.help_messages import *



def save_predictions(labels,source_file_name,behavior_classes,sample_rate):
    """takes numerical labels and transforms back into one-hot encoded file (BORIS style). Saves as csv"""

    df = pd.DataFrame(labels,columns=["labels"])
    time_clm = np.round(np.arange(0,df.shape[0]) / sample_rate,2)
    # convert numbers into behavior names
    class_dict = {i: x for i,x in enumerate(behavior_classes)}
    df["classes"] = df["labels"].copy()
    for cl_idx,cl_name in class_dict.items():
        df["classes"].iloc[df["labels"] == cl_idx] = cl_name

    # for simplicity let's convert this back into BORIS type file
    dummy_df = pd.get_dummies(df["classes"])
    #add 0 columns for each class that wasn't predicted in the file
    not_predicted_classes = [x for x in behavior_classes if x not in np.unique(df["classes"].values)]
    for not_predicted_class in not_predicted_classes:
        dummy_df[not_predicted_class] = 0

    dummy_df["time"] = time_clm
    dummy_df = dummy_df.set_index("time")

    # save to csv
    file_name = source_file_name.split(".")[0]
    dummy_df.to_csv(file_name + "_labels.csv")


class Predictor:

    def __init__(self,config):
        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.multi_animal = config["Project"].getboolean("MULTI_ANIMAL")
        self.software = config["Project"].get("PROJECT_TYPE")
        self.ftype = config["Project"].get("FILE_TYPE")
        self.selected_animals = [x.strip() for x in config["Project"].get("INDIVIDUALS_CHOSEN").split(",")]
        self.selected_animal_idx = None
        self.selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        self.selected_pose_idx = None
        self.idx_selected = None
        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")


        [self.iterX_model,_,_,_,_,_] = load_iterX(self.working_dir,self.prefix)
        [_, _, self.scalar, _] = load_features(self.working_dir,self.prefix)

        self.pose_files = None
        self.pose_file_names = []
        self.processed_input_data = []

        self.features = None
        self.scaled_features = None

        self.predictions = None

    def set_pose_idx(self):

        file0_df = load_pose(self.pose_files[0],origin=self.software.lower(),multi_animal=self.multi_animal)
        if self.multi_animal:
            if self.software.lower() == "deeplabcut":
                # if it's multi animal, we take bodyparts from a level below
                animal_lvl = 1
                bp_level = 2
            elif self.software.lower() == "sleap":
                # sleap converted files don't have a scorer level
                animal_lvl = 0
                bp_level = 1

            # find the indexes where the animal has bps
            an_index_list = []
            for an in self.selected_animals:
                an_index = np.argwhere(file0_df.columns.get_level_values(animal_lvl) == an)
                an_index_list.append(an_index)

            self.selected_animal_idx = np.sort(np.array(an_index_list).flatten())
            st.success("**Selected individuals/animals**: " + ", ".join(self.selected_animals))
        else:
            bp_level = 1

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

        # get rid of likelihood columns for deeplabcut
        idx_llh = self.selected_pose_idx[2::3]
        # the loaded sleap file has them too, so exclude for both
        self.idx_selected = [i for i in self.selected_pose_idx if i not in idx_llh]

    def upload_data(self):

        if st.checkbox("Select different pose origin",help=DIFFERENT_POSE_ORIGIN):
            self.software,self.ftype = select_software()

        upload_container = st.container()
        if not self.software == 'CALMS21 (PAPER)':

            self.pose_files = upload_container.file_uploader('Upload pose estimation files',
                                                             accept_multiple_files=True
                                                             ,type=self.ftype
                                                             ,key='pose'
                                                             ,help=POSE_UPLOAD_HELP)
        else:
            upload_container.error(
                "All files from the data set have been used. Select a different pose estimation origin.")

    def compile_data(self):

        # load pose idx
        self.set_pose_idx()
        self.pose_file_names = []
        for i,f in enumerate(self.pose_files):
            # because we cannot be sure whether the file was already used by read_csv,
            # we need to refresh the buffer!
            f.seek(0)
            self.pose_file_names.append(f.name)
            current_pose = load_pose(f,origin=self.software.lower(),multi_animal=self.multi_animal)
            # take selected pose idx from config
            self.processed_input_data.append(np.array(current_pose.iloc[:,self.idx_selected]))

        # feature extraction
        number2train = len(self.processed_input_data)
        frames2integ = round(float(self.framerate) * (self.duration_min / 0.1))

        self.features = []
        self.scaled_features = []
        # extract features, bin them
        for i,data in enumerate(self.processed_input_data):
            # we are doing this to predict on each file seperatly!
            #feature extracting from within each file
            # features,scaled_features = feature_extraction([data]
            #                                               ,1
            #                                               ,frames2integ
            #                                               )
            #using feature scaling from training set
            features,scaled_features = feature_extraction_with_extr_scaler([data]
                                                                            ,1
                                                                            ,frames2integ
                                                                            ,self.scalar
                                                                           )
            self.features.append(features)
            self.scaled_features.append(scaled_features)

    def predict(self):
        prediction_container = st.container()
        if self.scaled_features is not None:
            self.predictions = []
            # TODO: CHECK WITH ALEX IF SCALED OR UNSCALED FEATURES
            for i in stqdm(range(len(self.scaled_features)),desc="Behavior prediction from spatiotemporal features"):
                with st.spinner('Predicting behavior from features...'):
                    predict = bsoid_predict_numba_noscale([self.scaled_features[i]],self.iterX_model)
                    predict_arr = np.array(predict).flatten()

                    self.predictions.append(predict_arr)

        else:
            prediction_container.error("Extract features first.")

    def save_predictions(self):
        prediction_container = st.container()
        with st.spinner("Saving predictions now..."):
            for num,pred_file in enumerate(self.predictions):
                curr_file_name = os.path.join(self.working_dir, self.prefix, self.pose_file_names[num])
                save_predictions(pred_file
                                 ,source_file_name=curr_file_name
                                 ,behavior_classes=self.annotation_classes
                                 ,sample_rate=1 / self.duration_min)

        prediction_container.success("Predicted all files.")

    def main(self):
        self.upload_data()
        # load data and extract features based config
        if st.button("Predict new data"):
            self.compile_data()
            # predict on extracted features
            self.predict()
            self.save_predictions()