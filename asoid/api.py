# exposed functions to run asoid via lines
import os
import numpy as np
from configparser import ConfigParser
import pandas as pd

import streamlit as st
import logging

streamlit_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("streamlit")]
for logger in streamlit_loggers:
    logger.setLevel(logging.ERROR)

from asoid.utils.auto_active_learning import get_available_conf_options, RF_Classify
from asoid.utils.extract_features import Extract

from asoid.utils.predict import bsoid_predict_numba_noscale, bsoid_predict_proba_numba_noscale, weighted_smoothing
from asoid.utils.extract_features_2D import feature_extraction
from asoid.utils.extract_features_3D import feature_extraction_3d
from asoid.utils.load_workspace import load_new_pose, load_iterX
from asoid.utils.import_data import load_pose
from asoid.utils.preprocessing import adp_filt, sort_nicely

from asoid.utils.reporting import extract_descriptors, prep_labels_single

from tqdm import tqdm


def load_config(config_path, verbose = True):
    """
    Load the configuration file.
    :param config_path: Path to the configuration file.
    :return: Configuration object.
    """
    config = ConfigParser()
    config.read(config_path)

    if verbose:
        # report config
        for key, value in config.items():
            print(f"{key}: ")
            for k, v in value.items():
                print(f"  {k}: {v}")


    return config

def _convert_predictions(predictions, annotation_classes, framerate):
    """takes numerical labels and transforms back into one-hot encoded file (BORIS style).
    :param predictions: list of predictions
    :param annotation_classes: list of annotation classes
    :param framerate: framerate of the video
    :return: DataFrame of predictions"""
    # convert to pandas dataframe
    df = pd.DataFrame(predictions, columns=["labels"])
    time_clm = np.round(np.arange(0, df.shape[0]) / framerate, 3)

    # for simplicity let's convert this back into BORIS type file
    dummy_df = pd.get_dummies(df["labels"])

    #rename columns to match the classes
    dummy_df.columns = [annotation_classes[int(x)] for x in dummy_df.columns]
    #remove the first column
    dummy_df = dummy_df.drop(columns=[dummy_df.columns[0]])

    # add 0 columns for each class that wasn't predicted in the file
    not_predicted_classes = [x for x in annotation_classes if x not in np.unique(dummy_df.columns)]
    for not_predicted_class in not_predicted_classes:
        dummy_df[not_predicted_class] = 0

    dummy_df["time"] = time_clm
    dummy_df = dummy_df.set_index("time")
    
    return dummy_df

def _convert_predictions_proba(predictions_proba, annotation_classes, framerate):
    """takes numerical labels and adds timestamps (in BORIS style). However, this is not a one-hot encoded file, but saves the probabilities for each class.
    :param predictions: list of prediction probabilities
    :param annotation_classes: list of annotation classes
    :param framerate: framerate of the video
    :return: DataFrame of predictions"""
    # convert to pandas dataframe
    df = pd.DataFrame(predictions_proba, columns=annotation_classes)
    time_clm = np.round(np.arange(0, df.shape[0]) / framerate, 3)
    # convert numbers into behavior names

    df["time"] = time_clm
    df = df.set_index("time")
    
    return df

def _save_predictions(predictions_df, output_filename):
    """ Save the predictions to a file.
    :param predictions_df: DataFrame of predictions.
    :param output_filename: Name of the output file.
    :return: None
    """
    # save predictions
    predictions_df.to_csv(output_filename)
    print(f"Predictions saved to {output_filename}")

def _report(pred_df, annotation_classes, framerate):
    """
    This function generates a report from the predictions.
    :param pred_df: DataFrame of predictions.
    :return: None
    """
    
    count_df = extract_descriptors(pred_df, annotation_classes, framerate)
    print("Behavior report:")
    print(count_df)


class Trainer:
    """
    This class is used to train the model.
    """

    def __init__(self, config, verbose = True):
        """
        Initialize the Trainer class.
        :param config: Configuration object.
        :param verbose: If True, print the configuration. Default is True.
        """
        self.verbose = verbose
        # load config
        self.config = config
        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.software = config["Project"].get("PROJECT_TYPE")
        self.ftype = [x.strip() for x in config["Project"].get("FILE_TYPE").split(",")]
        self.selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        self.is_3d = config["Project"].getboolean("IS_3D")
        self.multi_animal = config["Project"].getboolean("MULTI_ANIMAL")
        self.llh_value = config["Processing"].getfloat("LLH_VALUE")

        self.exclude_other = config["Project"].getboolean("EXCLUDE_OTHER")
        self.train_fx = config["Processing"].getfloat("TRAIN_FRACTION")
          
        self.framerate = config["Project"].getfloat("FRAMERATE")
    
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")
        self.frames2integ = round(float(self.framerate) * (self.duration_min / 0.1))

        # backwards compatibility
        try:
            self.conf_type = config["Processing"].get("CONF_TYPE")
        except KeyError:
            self.conf_type = get_available_conf_options().keys()[0]
        try:
            self.conf_threshold = config["Processing"].getfloat("CONF_THRESHOLD")
        except KeyError:
            self.conf_threshold = 0.5
    
        try:
            self.iteration = config["Processing"].getint("ITERATION")
        except KeyError:
            self.iteration = 0

        self.project_dir = os.path.join(self.working_dir, self.prefix)
        self.iter_folder = str.join('', ('iteration-', str(self.iteration)))
        


    def extract_features(self, duration_min = None):
        """ Extract features from the data and save them to a file.
        :param duration_min: Duration of the features in seconds. Default is None, which means the default duration from the config file is used.
        """
        if duration_min is not None:
            assert duration_min > 0, "Duration must be greater than 0"
            self.duration_min = duration_min
            self.frames2integ = round(self.framerate * (self.duration_min / 0.1))

        if self.verbose:
            print(f"Frames to integrate to reach {self.duration_min} sec: {self.frames2integ}")
        extractor = Extract(self.working_dir, self.prefix, self.frames2integ, self.is_3d)
        extractor.main()


    def train_model(self
        , init_ratio: float
        , max_iter: int
        , max_samples_iter: int
        , iteration = None
        , conf_type = None
        , conf_threshold = None
        , mode = "notebook"):
        """ Train the model and save it to a file.
        :param init_ratio: Initial ratio of training data.
        :param max_iter: Maximum number of iterations.
        :param max_samples_iter: Maximum number of samples per iteration.
        :param iteration: Iteration number. Default is None, which means the last iteration.
        :param conf_type: Confidence type. Default is None, which means the last confidence type.
        :param conf_threshold: Confidence threshold.
        :param mode: Mode of operation (notebook or cli).
        :return: None
        """

        if init_ratio is not None:
            assert init_ratio > 0 and init_ratio < 1, "Initial ratio must be between 0 and 1"
            self.init_ratio = init_ratio
        if max_iter is not None:
            assert max_iter > 0, "Max iterations must be greater than 0"
            self.max_iter = max_iter

        if max_samples_iter is not None:
            assert max_samples_iter > 0, "Max samples per iteration must be greater than 0"
            self.max_samples_iter = max_samples_iter

        if conf_type is not None:
            assert conf_type in get_available_conf_options().keys(), f"Conf type must be one of {get_available_conf_options().keys()}"
            self.conf_type = conf_type

        if conf_threshold is not None:
            assert conf_threshold > 0 and conf_threshold < 1, "Conf threshold must be between 0 and 1"
            self.conf_threshold = conf_threshold
        
        assert mode in ["notebook", "cli"], "Mode must be either notebook or cli"

        os.makedirs(os.path.join(self.project_dir, self.iter_folder), exist_ok=True)
        # 
        rf_classifier = RF_Classify(self.working_dir
                                , self.prefix, self.iter_folder, self.software
                                , self.init_ratio
                                , self.max_iter
                                , self.max_samples_iter
                                , self.annotation_classes
                                , self.exclude_other
                                , self.conf_type
                                , self.conf_threshold
                                , mode = mode)
        rf_classifier.main()


def extract_features(config, duration_min = None):
    """ Extract features from the data and save them to a file.
    :param config: Configuration object.
    :param duration_min: Duration of the features in seconds. Default is None, which means the default duration from the config file is used.
    """
    trainer = Trainer(config)
    trainer.extract_features(duration_min=duration_min)

def train_model(config, init_ratio: float, max_iter: int, max_samples_iter: int, iteration = None, conf_type = None, conf_threshold = None, mode = "notebook"):
    """ Train the model and save it to a file.
    :param config: Configuration object.
    :param init_ratio: Initial ratio of training data.
    :param max_iter: Maximum number of iterations.
    :param max_samples_iter: Maximum number of samples per iteration.
    :param iteration: Iteration number. Default is None, which means the last iteration.
    :param conf_type: Confidence type. Default is None, which means the last confidence type.
    :param conf_threshold: Confidence threshold.
    :param mode: Mode of operation (notebook or cli).
    :return: None
    """
    trainer = Trainer(config)
    trainer.train_model(init_ratio, max_iter, max_samples_iter, iteration=iteration, conf_type=conf_type, conf_threshold=conf_threshold, mode=mode)


# prediction on new data

class Predictor:
    def __init__(self, config, verbose = False):
        """
        Initialize the Predictor class.
        :param config: Configuration object.
        """
        self.verbose = verbose
        # load config
        self.config = config
        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.software = config["Project"].get("PROJECT_TYPE")
        self.ftype = [x.strip() for x in config["Project"].get("FILE_TYPE").split(",")]
        self.selected_bodyparts = [x.strip() for x in config["Project"].get("KEYPOINTS_CHOSEN").split(",")]
        self.is_3d = config["Project"].getboolean("IS_3D")
        self.multi_animal = config["Project"].getboolean("MULTI_ANIMAL")
        self.llh_value = config["Processing"].getfloat("LLH_VALUE")
        self.iteration = config["Processing"].getint("ITERATION")
        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")
        self.project_dir = os.path.join(self.working_dir, self.prefix)
        self.iter_folder = str.join('', ('iteration-', str(self.iteration)))
        self.frames2integ = round(float(self.framerate) * (self.duration_min / 0.1))

        # load model
        [self.iterX_model, _, _] = load_iterX(self.project_dir, self.iter_folder)

        # setup variables
        self.features = None
        self.scaled_features = None
        self.predictions_raw = None
        self.predictions_proba = None
        self.predictions_match = None

        self.new_pose_csvs = []

        self.total_n_frames = []
        
    def extract_features(self, path_list):
        """ Extract features from the data.
        :param path_list: List of paths to pose files.
        :return: None
        """

        if self.features is None:
            self.features = []
        if self.scaled_features is None:
            self.scaled_features = []

        self.new_pose_csvs = path_list


        # for i, data in enumerate(processed_input_data):
        for i, f in enumerate(tqdm(self.new_pose_csvs, desc="Extracting spatiotemporal features from pose")):
        # for i, f in enumerate(new_pose_csvs):

            current_pose = load_pose(f, self.software, self.multi_animal)
            bp_level = 1
            bp_index_list = []

            if i == 0:
                # check if all bodyparts are in the pose file
                if len(self.selected_bodyparts) > len(current_pose.columns.get_level_values(bp_level).unique()):
                    raise ValueError(f'Not all selected keypoints/bodyparts are in the pose file: {f.name}')

                elif len(self.selected_bodyparts) < len(current_pose.columns.get_level_values(bp_level).unique()):
                    # subselection would take care of this, so we need to make sure that they all exist
                    for bp in sself.elected_bodyparts:
                        if bp not in current_pose.columns.get_level_values(bp_level).unique():
                            raise ValueError(f'At least one keypoint "{bp}" is missing in pose file: {f.name}')
            
                for bp in self.selected_bodyparts:
                    bp_index = np.argwhere(current_pose.columns.get_level_values(bp_level) == bp)
                    bp_index_list.append(bp_index)
                selected_pose_idx = np.sort(np.array(bp_index_list).flatten())

                # get likelihood column idx directly from dataframe columns
                idx_llh = [i for i, s in enumerate(current_pose.columns) if "likelihood" in s and s in current_pose.columns[selected_pose_idx]]

                # the loaded sleap file has them too, so exclude for both
                idx_selected = [i for i in selected_pose_idx if i not in idx_llh]

            # filtering does not work for 3D yet
            # check if there is a z coordinate

            if "z" in current_pose.columns.get_level_values(2):
                if self.is_3d is not True:
                    raise ValueError("3D data detected. But parameter is set to 2D project.")
                print("3D project detected. Skipping likelihood adaptive filtering.")
                # if yes, just drop likelihood columns and pick the selected bodyparts
                filt_pose = current_pose.iloc[:, idx_selected].values
            else:
                filt_pose, _ = adp_filt(current_pose, idx_selected, idx_llh, self.llh_value)

            # using feature scaling from training set
            if not self.is_3d:
                feats, scaled_feats = feature_extraction([filt_pose], 1, self.frames2integ)
            else:
                feats, scaled_feats = feature_extraction_3d([filt_pose], 1, self.frames2integ)

            self.features.append(feats)
            self.scaled_features.append(scaled_feats)
            self.total_n_frames.append(filt_pose.shape[0])


    def predict(self, smooth_size = 0):
        """ Predict the behavior from pose data and returns raw and smoothed output.
        :param path_list: List of paths to pose files.
        :param smooth_size: Size of the smoothing window. If 0, no smoothing is applied. Default is 0.
        :return: predictions_raw, predictions_match: List of predictions per file (raw and smoothed). If export_proba is True, also returns probabilities.
        """
        assert smooth_size >= 0, "Smooth size must be greater than or equal to 0"
        assert self.features is not None, "Features must be extracted before prediction"
        
  
        repeat_n = int(self.frames2integ / 10)

        self.predictions_raw = []
        self.predictions_match = []
        self.predictions_proba = []
        for i in tqdm(range(len(self.features)), desc="Behavior prediction from spatiotemporal features"):

            predict = bsoid_predict_numba_noscale([self.features[i]], self.iterX_model)
           
            proba = bsoid_predict_proba_numba_noscale([self.features[i]], self.iterX_model)
            

            predict_arr = np.array(predict).flatten()
            proba_arr = np.array(proba).reshape(-1, np.array(proba).shape[2])

            predictions_raw = np.pad(predict_arr.repeat(repeat_n), (repeat_n, 0), 'edge')[:self.total_n_frames[i]]
            predictions_proba = np.pad(proba_arr.repeat(repeat_n, axis=0), ((repeat_n, 0), (0, 0)), mode='edge'
                                    )[:self.total_n_frames[i], :]
            
            if smooth_size > 0:
                # smooth predictions
                predictions_match = weighted_smoothing(predictions_raw, size=smooth_size)
            else:
                predictions_match = predictions_raw

            self.predictions_raw.append(predictions_raw)
            self.predictions_match.append(predictions_match)
            self.predictions_proba.append(predictions_proba)

   

    def frameshift_predict(self):
        pass


    def save_predictions(self, output_types = ["smooth"], output_dir = None):
        """ Save the predictions to a file in BORIS style.
        :param output_types: Types of output ("raw", "smooth", "proba"). Default is "smooth". If "raw", save the raw predictions. If "smooth", save the smoothed predictions. If "proba", save the probabilities.
        Can be a list of types. If "all", save all types.
        :param output_dir: Directory or list of directories to save the predictions. Default is None, which means the default directory from the config file is used.
        :return: None
        """
        if output_types == "all":
                output_types = ["raw", "smooth", "proba"]
        
        # check if output_dir is a list
        if output_dir is not None:
            if isinstance(output_dir, str):
                output_dir = [output_dir] * len(self.new_pose_csvs)
            elif len(output_dir) != len(self.new_pose_csvs):
                raise ValueError(f"Output directory list must be the same length as the number of files. {len(output_dir)} != {len(self.new_pose_csvs)}")


        # save predictions
        for i in range(len(self.predictions_raw)):

            # save predictions
            curr_file_name = self.new_pose_csvs[i]
            curr_file_name = os.path.basename(curr_file_name)
            curr_file_name = os.path.splitext(curr_file_name)[0]

            # check if output_types is a list
            if isinstance(output_types, str):
                output_types = [output_types]

            for o_type in output_types:
                if o_type not in ["raw", "smooth", "proba"]:
                    raise ValueError(f"Output type {o_type} is not valid. Must be one of ['raw', 'smooth', 'proba']")

                if o_type == "raw":
                    predictions = self.predictions_raw[i]
                    # convert predictions to pandas dataframe in BORIS style            
                    pred_df = _convert_predictions(predictions, self.annotation_classes, self.framerate)
                    
                elif o_type == "smooth":
                    predictions = self.predictions_match[i]# convert predictions to pandas dataframe in BORIS style            
                    pred_df = _convert_predictions(predictions, self.annotation_classes, self.framerate)
                    
                elif o_type == "proba":
                    predictions = self.predictions_proba[i]
                    pred_df = _convert_predictions_proba(predictions, self.annotation_classes, self.framerate)
                
                if output_dir is None:
                    output_filename = os.path.join(self.project_dir, self.iter_folder, curr_file_name + f"_predictions_{o_type}.csv")
                else:
                    curr_output_dir = output_dir[i]
                    os.makedirs(curr_output_dir, exist_ok=True)
                    # check if curr_output_dir is a valid directory
                    if not os.path.isdir(curr_output_dir):
                        raise ValueError(f"Output directory {curr_output_dir} is not a valid directory.")
                    output_filename = os.path.join(curr_output_dir, curr_file_name + f"_predictions_{o_type}.csv")


                # save predictions
                _save_predictions(pred_df, output_filename)


def predict(config, path_list
            , smooth_size = 0
            , save_predictions = True
            , output_dir = None
            , output_types = "smooth"
            , verbose = False):
    """ Predict the behavior from pose data and returns raw and smoothed output.
    :param config: Configuration object.
    :param path_list: List of paths to pose files.
    :param smooth_size: Size of the smoothing window. If 0, no smoothing is applied. Default is 0.
    :param save_predictions: If True, save the predictions to csv file in BORIS style at location. Default is True.
    :param output_dir: Directory to save the predictions. Default is None, which means the default directory from the config file is used.
    :param output_types: Types of output ("raw", "smooth", "proba"). Default is "smooth". If "raw", save the raw predictions. If "smooth", save the smoothed predictions. If "proba", save the probabilities.
    Can be a list of types. If "all", save all types.
    :param verbose: If True, print the a report for each prediction. Default is False.
    :return: predictions_raw, predictions_match, prediction_proba: List of predictions per file (raw, smoothed, and probabilities). 
    """

    predictor = Predictor(config, verbose=verbose)
    predictor.extract_features(path_list)
    predictor.predict(smooth_size=smooth_size)
    
    if save_predictions:
        predictor.save_predictions(output_dir = output_dir, output_types = output_types)
    
    return predictor.predictions_raw, predictor.predictions_match, predictor.predictions_proba