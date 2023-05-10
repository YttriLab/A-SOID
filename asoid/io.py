# this file serves as a interface for script style usage of asoid functions

import os
import numpy as np
from tqdm import tqdm
import glob

from utils.project_utils import load_config, view_config_str
from utils.predict_new_data import save_predictions
from utils.import_data import load_pose
from utils.extract_features import feature_extraction_with_extr_scaler,bsoid_predict_numba_noscale
import utils.loading_utils as lu

#catch streamlit warning
#TODO: FIX WARNING
# WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager



def load_project(project_path: str, config = None):
    """Loads a project from a path and returns config file"""

    if config is None:
        config, _ = load_config(project_path)

    assert config is not None, "No config file found at {}".format(project_path)

    return config

def show_config(config):
    """Prints the config file"""
    view_config_str(config)



class Predictor:
    """Same as the Predictor class in asoid\predict_new_data.py but without streamlit and made to work with project class"""

    def __init__(self,config, verbose = False):
        self._verbose = verbose
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


        [self.iterX_model,_,_,_,_,_] = lu.load_iterX(self.working_dir,self.prefix)
        [_, _, self.scalar, _] = lu.load_features(self.working_dir,self.prefix)

        self.pose_files = None
        self.pose_file_names = []
        self.processed_input_data = []

        self.features = None
        self.scaled_features = None

        self.predictions = None

    def _set_pose_idx(self):
        """Sets the pose indexes for the selected bodyparts and individuals"""

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
            if self._verbose:
                print("**Selected individuals/animals**: " + ", ".join(self.selected_animals))
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

    def _upload_data(self, pose_file_paths, pose_origin = None):
        """Uploads the pose data from the selected files and stores it in the class instance."""

        if pose_origin is not None:
            self.software = pose_origin

        if not self.software == 'CALMS21 (PAPER)':

            self.pose_files = pose_file_paths
        else:
            raise AssertionError(
                "All files from the data set have been used. Select a different pose estimation origin.")

    def _compile_data(self):
        if self._verbose:
            print("Compiling new files...")
        # load pose idx
        self._set_pose_idx()
        self.pose_file_names = []
        for i,f in enumerate(self.pose_files):
            self.pose_file_names.append(f)
            current_pose = load_pose(f,origin=self.software.lower(),multi_animal=self.multi_animal)
            # take selected pose idx from config
            self.processed_input_data.append(np.array(current_pose.iloc[:,self.idx_selected]))

        # feature extraction
        number2train = len(self.processed_input_data)
        frames2integ = round(float(self.framerate) * (self.duration_min / 0.1))

        if self._verbose:
            print("Extracting features...")

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

    def _predict(self):
        if self._verbose:
            print("Predicting new files...")

        if self.scaled_features is not None:
            self.predictions = []
            # TODO: CHECK WITH ALEX IF SCALED OR UNSCALED FEATURES
            for i in tqdm(range(len(self.scaled_features)),desc="Behavior prediction from spatiotemporal features"):
                predict = bsoid_predict_numba_noscale([self.scaled_features[i]],self.iterX_model)
                predict_arr = np.array(predict).flatten()

                self.predictions.append(predict_arr)

        else:
            raise AssertionError("Extract features first.")

    def _save_predictions(self):

        if self._verbose:
            print("Saving predictions...")

        for num,pred_file in enumerate(self.predictions):
            curr_file_name = os.path.join(self.working_dir, self.prefix, self.pose_file_names[num])
            save_predictions(pred_file
                             ,source_file_name=curr_file_name
                             ,behavior_classes=self.annotation_classes
                             ,sample_rate=1 / self.duration_min)


    def predict(self, pose_file_paths, pose_origin = None, save_predictions = True):
        """Main function to run the prediction on a set of pose files given as list of paths.
        :param pose_file_paths: list of paths to pose files
        :param pose_origin: origin of the pose files, if not specified, the origin from the config file is used
        :param save_predictions: if True, the predictions are saved as .csv files in the same directory as the pose files
        :return: predictions as list of numpy arrays
            """

        assert isinstance(pose_file_paths, list), "Please provide a list of paths to pose files."

        self._upload_data(pose_file_paths, pose_origin = pose_origin)
        self._compile_data()
        # predict on extracted features
        self._predict()
        if save_predictions:
            self._save_predictions()
        else:
            pass

        return self.predictions



class Project:
    """Loads an A-SOiD project from a path and provides methods to access data and models"""
    def __init__(self, project_path, verbose = False):
        self._verbose = verbose
        self.config = load_project(project_path)

        self.project_base_dir = self.config["Project"]["PROJECT_PATH"]
        self.prefix = self.config["Project"]["PROJECT_NAME"]

        #get relevant data
        self._data = self.get_data()

        if self._verbose:
            print("Loading project from {}.".format(project_path))
            print("Project name: {}".format(self.prefix))

        #check if classifier exists and load it
        try:
            self.clf = Predictor(self.config, self._verbose)

        except FileNotFoundError:
            if self._verbose:
                print("No classifier found. Please train a classifier first. Continuing without classifier.")

            self.clf = None

    def get_config(self):
        """Returns the config file. if verbose prints the config file"""
        if self._verbose:
            show_config(self.config)
        return self.config

    def get_data(self):
        """Returns the data file"""
        data, _ = lu.load_data(self.project_base_dir, self.prefix)
        return data

    def get_classifier(self):
        """returns latest iteration of classifier"""
        return self.clf

    def _predict_from_list(self, pose_files, pose_origin = None, save_predictions = True):
        """Predicts on a list of pose files.
        :param pose_files: list of paths to pose files
        :param pose_origin: origin of the pose files, if not specified, the origin from the config file is used
        :param save_predictions: if True, the predictions are saved as .csv files in the same directory as the pose files
        :return: predictions as list of numpy arrays"""
        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.predict(pose_files, pose_origin, save_predictions)


    def _predict1(self, pose_file, pose_origin= None, save_predictions = True):
        """Predicts on a single pose files.
        :param pose_file:  path to pose files
        :param pose_origin: origin of the pose files, if not specified, the origin from the config file is used
        :param save_predictions: if True, the predictions are saved as .csv files in the same directory as the pose files
        :return: predictions as list of numpy arrays"""
        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.predict([pose_file], pose_origin, save_predictions)

    def _predict_from_folder(self, pose_files, pose_origin = None, save_predictions = True):
        """Predicts on a list of pose files.
        :param pose_files: list of paths to pose files
        :param pose_origin: origin of the pose files, if not specified, the origin from the config file is used
        :param save_predictions: if True, the predictions are saved as .csv files in the same directory as the pose files
        :return: predictions as list of numpy arrays"""

        if pose_origin is None:
            #get f_type from config
            f_type =  self.config["Project"].get("FILE_TYPE")

        elif pose_origin.lower() == "sleap":
            f_type = "h5"

        elif pose_origin.lower() == "deeplabcut":
            f_type = "csv"
        else:
            raise AssertionError("Please specify a valid pose origin (sleap or deeplabcut)")

        #find all files from same type in folder
        if os.path.isdir(pose_files):
            pose_files = glob.glob(pose_files + "/*.{}".format(f_type))
        else:
            raise AssertionError("Please provide a valid path to a folder.")

        assert pose_files, "No files found in folder. Please provide a valid path to a folder."

        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.predict(pose_files, pose_origin, save_predictions)


    def predict(self, pose_files, pose_origin = None, save_predictions = True):
        """Predicts new file(s).
        :param pose_files: list of paths to pose files, or path to folder with pose files, or path to single pose file
        :param pose_origin: origin of the pose files, if not specified, the origin from the config file is used
        :param save_predictions: if True, the predictions are saved as .csv files in the same directory as the pose files
        :return: predictions as list of numpy arrays"""

        if pose_files is None:
            raise AssertionError("Please provide a valid path to a pose file or folder.")

        if isinstance(pose_files, list):
            return self._predict_from_list(pose_files, pose_origin, save_predictions)
        elif os.path.isdir(pose_files):
            return self._predict_from_folder(pose_files, pose_origin, save_predictions)
        elif os.path.isfile(pose_files):
            return self._predict1(pose_files, pose_origin, save_predictions)

    def get_predictions(self):
        """Returns predictions from latest prediction run
        :return: predictions"""
        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.predictions

    def get_pose(self):
        """Returns pose and pose file names from latest prediction run
        :return: pose_file_names, processed_input_data"""
        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.pose_file_names, self.clf.processed_input_data


    def get_features(self):
        """Returns features from latest prediction run
        :return: features, scaled_features"""
        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.features, self.clf.scaled_features

