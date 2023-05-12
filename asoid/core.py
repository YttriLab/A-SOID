# this file serves as a interface for script style usage of asoid functions

import os
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
import datetime

from asoid.utils.project_utils import load_config, view_config_str
from asoid.utils.predict_new_data import save_dataframe_to_csv, convert_predictions_to_dataframe, convert_dataframe_to_dummies
from asoid.utils.import_data import load_pose
from asoid.utils.extract_features import feature_extraction_with_extr_scaler,bsoid_predict_numba_noscale
import asoid.utils.loading_utils as lu

from asoid.utils.view_results import count_events

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

        if self._verbose:
            print("Project loaded")
            print("You can now use the predict method to predict new data")

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

    def _save_prediction(self, pred_file,  file_name):

        curr_file_name = os.path.join(self.working_dir, self.prefix, file_name)
        save_dataframe_to_csv(pred_file, curr_file_name)


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
        sample_rate = 1 / self.duration_min
        #convert to dataframe
        for i in range(len(self.predictions)):
            curr_pred = convert_predictions_to_dataframe(self.predictions[i],self.annotation_classes, sample_rate)
            curr_pred = convert_dataframe_to_dummies(curr_pred, self.annotation_classes)
            if save_predictions:
                if self._verbose:
                    print("Saving predictions...")
                curr_file_name = os.path.basename(self.pose_file_names[i])
                self._save_prediction(curr_pred, curr_file_name)
        else:
            pass

        return self.predictions

    def get_predictions(self, format = "dummies"):
        """Returns the predictions as list of pd.DataFrames including the annotation classes as columns.
        :param format: "dummies" or "labels", if "dummies" the predictions are returned as one-hot encoded vectors, if "labels" the predictions are returned as labels
        """
        assert format in ["dummies", "labels"], "Please specify a valid format ('dummies' or 'labels')."
        predictions =  []
        sample_rate = 1 / self.duration_min
        for num, pred in enumerate(self.predictions):

            curr_pred = convert_predictions_to_dataframe(pred, self.annotation_classes, sample_rate)
            if format == "dummies":
                curr_pred = convert_dataframe_to_dummies(curr_pred, self.annotation_classes)
            predictions.append(curr_pred)
        return predictions


class Viewer:

    def __init__(self, config, verbose = False):

        assert config is not None, "Please provide a config file."

        self._verbose = verbose

        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")
        self.class_to_number = {s: i for i, s in enumerate(self.annotation_classes)}
        self.number_to_class = {i: s for i, s in enumerate(self.annotation_classes)}

    def convert_dummies_to_labels(self, labels):
        """
        This function converts dummy variables to labels
        :param labels: pandas dataframe with dummy variables
        :return: pandas dataframe with labels and codes
        """
        if self._verbose:
            print("Converting dummies to labels...")

        conv_labels = pd.from_dummies(labels)
        cat_df = pd.DataFrame(conv_labels.values, columns=["labels"])
        if self.annotation_classes is not None:
            cat_df["labels"] = pd.Categorical(cat_df["labels"] , ordered=True, categories=self.annotation_classes)
        else:
            cat_df["labels"] = pd.Categorical(cat_df["labels"], ordered=True, categories=cat_df["labels"].unique())
        cat_df["codes"] = cat_df["labels"].cat.codes

        return cat_df

    def plot_ethogram_single(self,df_label):
        """ This function plots the labels in a heatmap"""

        if self._verbose:
            print("Plotting ethogram...")

        names = [f"{x}: {y}" for x, y in dict(zip(df_label['labels'].cat.codes, df_label['labels'] )).items()]
        # Count the number of unique values
        unique_values = df_label["labels"].unique()
        num_values = len(unique_values) + 1

        # Define a colormap with one color per unique value
        colors = px.colors.qualitative.Light24[:num_values + 1]
        tick_space = np.linspace(0, 1, num_values)
        tick_space = np.repeat(tick_space, 2)[1:-1]
        color_rep = np.repeat(colors, 2)
        my_colorscale = list(zip(tick_space, color_rep))

        trace = go.Heatmap(z=df_label.values.T,
                           colorscale=my_colorscale,
                           showscale=True,

                           colorbar=dict(
                               tickmode='array',
                               tickvals=np.arange(num_values),
                               ticktext= names,
                               tickfont=dict(size=14),
                               lenmode='fraction',
                               #len=0.5,
                               thicknessmode='fraction',
                               thickness=0.03,
                               outlinewidth=0,
                               bgcolor='rgba(0,0,0,0)'
                           )
                           )

        layout = go.Layout(
                            xaxis=dict(title='Frames'),
                            yaxis=dict(title='Session', range=[0.7,1.2]),
                            title_text="Ethogram"
                            )

        fig = go.Figure(data=[trace], layout=layout)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_xaxes(zeroline=False)

        return fig

    def plot_heatmap_single(self,df_label):
        """ This function plots the labels in a heatmap"""

        if self._verbose:
            print("Plotting heatmap...")

        df_heatmap = df_label.copy()
        df_heatmap["session"] = 1

        fig = px.density_heatmap(df_heatmap,x = "session", y="labels"
                                 , width=500
                                 , height=500
                                 , histnorm="probability"
                                 #, text_auto=True
                                 )
        fig.update_layout(xaxis = dict(title='Session', showticklabels=False, showgrid=False, zeroline=False),
                        yaxis = dict(title='', autorange="reversed"),
                        title_text = "Relative frequency of labels")


        return fig

class Describer:
    """ This class describes the data in a table"""

    def __init__(self, config, verbose = False):
        assert config is not None, "Please provide a config file."

        self._verbose = verbose

        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")
        self.class_to_number = {s: i for i, s in enumerate(self.annotation_classes)}
        self.number_to_class = {i: s for i, s in enumerate(self.annotation_classes)}

    def describe_labels_single(self, df_label):
        """ This function describes the labels in a table"""

        if self._verbose:
            print("Describing labels...")

        event_counter = count_events(df_label)

        count_df = df_label.value_counts().to_frame().reset_index().rename(columns={0: "frame count"})
        # heatmap already shows this information
        #count_df["percentage"] = count_df["frame count"] / count_df["frame count"].sum() *100

        if self.framerate is not None:
            count_df["total duration"] = count_df["frame count"] / self.framerate
            count_df["total duration"] = count_df["total duration"].apply(lambda x: str(datetime.timedelta(seconds=x)))

        count_df["bouts"] = event_counter["events"]

        count_df.set_index("codes", inplace=True)
        count_df.sort_index(inplace=True)
        #rename all columns to include their units
        count_df.rename(columns={"frame count": "frame count [-]",
                                 "percentage": "percentage [%]",
                                 "total duration": "total duration [hh:mm:ss]",
                                 "bouts": "bouts [-]"},
                        inplace=True)
        #TODO: autosize columns with newer streamlit versions (e.g., using use_container_width=True)
        return count_df


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

        # init Viewer
        self.viewer = Viewer(self.config, self._verbose)
        #init Describer
        self.describer = Describer(self.config, self._verbose)

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

    def get_annotation_classes(self):
        """returns the annotation classes"""
        annotation_classes = [x.strip() for x in self.config["Project"].get("CLASSES").split(",")]
        return annotation_classes

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

    def get_predictions(self, format = "labels"):
        """Returns predictions from latest prediction run
        :param format: "dummies" or "labels", if "dummies" the predictions are returned as one-hot encoded vectors, if "labels" the predictions are returned as labels
       :return: list of predictions as pd.DataFrame in the same order as the pose files were provided to the predict function"""
        if self.clf is None:
            raise AssertionError("No classifier found. Please train a classifier first.")
        else:
            return self.clf.get_predictions(format= format)

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

    def get_summary(self, df_dummies:pd.DataFrame = None, pred_idx: int = None):
        """ Reports from latest prediction run by idx or from a given dataframe in chosen format.
        :param df_dummies: dataframe with predictions in one-hot encoded format (e.g. from get_predictions(format = "dummies"))
        :param pred_idx: index of prediction
        :return: pd.DataFrame with summary of predictions"""

        if df_dummies is None and pred_idx is not None:
            assert pred_idx < len(self.clf.predictions), "Please provide a valid index."
            df_dummies = self.get_predictions(format = "dummies")[pred_idx]

        elif df_dummies is None and pred_idx is None:
            raise AssertionError("Please provide a valid index or dataframe.")

        elif df_dummies is not None and pred_idx is not None:
            raise AssertionError("Please provide only an index or a dataframe.")

        df_labels = self.viewer.convert_dummies_to_labels(df_dummies)

        return self.describer.describe_labels_single(df_labels)


    def plot_predictions(self, df_dummies: pd.DataFrame = None, pred_idx: int = None, format = "ethogram"):
        """ Plots predictions from latest prediction run by idx or from a given dataframe in chosen format.
        :param df_dummies: dataframe with predictions in one-hot encoded format (e.g. from get_predictions(format = "dummies"))
        :param pred_idx: index of prediction to plot
        :param format: "ethogram" or "heatmap", if "ethogram" the predictions are plotted as an ethogram, if "heatmap" the predictions are plotted as a heatmap
        :return: plotly fig object"""

        if df_dummies is None and pred_idx is not None:
            assert pred_idx < len(self.clf.predictions), "Please provide a valid index."
            df_dummies = self.get_predictions(format = "dummies")[pred_idx]

        elif df_dummies is None and pred_idx is None:
            raise AssertionError("Please provide a valid index or dataframe.")

        elif df_dummies is not None and pred_idx is not None:
            raise AssertionError("Please provide only an index or a dataframe.")

        assert format in ["ethogram", "heatmap"], "Please provide a valid format (ethogram or heatmap)."

        df_labels = self.viewer.convert_dummies_to_labels(df_dummies)

        if format == "ethogram":
            return self.viewer.plot_ethogram_single(df_labels)

        elif format == "heatmap":
            return self.viewer.plot_heatmap_single(df_labels)





