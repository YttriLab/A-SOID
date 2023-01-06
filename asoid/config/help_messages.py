
"""General"""

CALM_HELP = " Keep this parameter if you want to replicate our findings."
NO_CONFIG_HELP = "Upload a config file to access this step. If you have not created a project yet," \
                 " return to 'Menu', or go directly to 'Upload Data' to begin your project."

NO_FEATURES_HELP = "Make sure that you extracted the features in the previous step."

"""Menu"""

UPLOAD_CONFIG_HELP = "Upload config files from previous A-SOiD projects to continue working on them."

"""Upload Data/ Data preprocess"""

POSE_ORIGIN_SELECT_HELP = "If you want to replicate the published results, select CalMS21 (Paper)." \
                          " Otherwise, we currently support DeepLabCut and SLEAP pose estimation."

FPS_HELP = "This information can usually be found in the meta data of the original videos used for pose estimation."

INIT_CLASS_SELECT_HELP =  "This will move all available data from the excluded classes to be included in 'other'."

EXCLUDE_OTHER_HELP = "WARNING: This option excludes the collective 'other' class from training. This is a valid option if you have only a FEW unlabeled frames (e.g., noise) which will be collected automatically into 'other'." \
                     " However, keep in mind, that if you are planning to predict on novel data, occurences of 'other' - i.e., when 'other'" \
                     " is used to collect a LARGE number of undefined frames (e.g., irrelevant behavior), will result in worse performance as" \
                     " the classifier will be forced to decide between the remaining classes for every input frame. "

MULTI_ANIMAL_HELP = "Only select this option if you are using a multi animal model for pose estimation, such as maDLC or SLEAP Top-Down or Bottom-Up." \
                    " If you have multiple animals, but are using a single instance network," \
                    " then select all body parts/individuals that you want to include during body part selection."

MULTI_ANIMAL_SELECT_HELP = "Select all animals that you want to include in the feature extraction." \
                           " Keep in mind that some behaviors require features of multiple animals to be represented correctly."

BODYPART_SELECT = "Select all body parts that you want included in the feature extraction." \
                  " Keep in mind that while some body parts will not give relevant information for the annotated behaviors," \
                  " some are essential for the separation between behaviors."

WORKING_DIR_HELP = "This working directory will be used to create a new A-SOiD project under the chosen prefix."

PREFIX_HELP = "This prefix will be used to create a parent folder for the current project." \
              " Keep in mind that picking the same prefix as an already existing project might lead to problems (including partial overwrites)."

DATA_DIR_IMPORT_HELP = "Use this if your files are over 2 GB or when you have a lot."

POSE_DIR_IMPORT_HELP = "Select the folder that includes the pose estimation files for this project." \
                       " Multi-folder import is not supported through this." \
                       " However you can use the normal uploader for this."

POSE_ORIGIN_HELP = "This will only accept files previously selected by their origin. DeepLabCut: csv files; SLEAP: h5 files."

POSE_SELECT_HELP = "Order files to match them with an annotation file. Make sure that this order is the same as the annotation file order. Default is the order in the folder and should be fine if the files are named in the same way."

LABEL_DIR_IMPORT_HELP = "Select the folder that includes the annotation files for this project." \
                       " Multi-folder import is not supported through this." \
                       " However you can use the normal uploader for this." \

LABEL_ORIGIN_HELP =   "Note that A-SOiD currently requires BORIS files in which annotations are 'one-hot encoded'. " \
                        "For this export your BORIS observation as a 'binary file' with the lowest sample rate."

LABEL_SELECT_HELP = "Order files to match them with a pose file. Make sure that this order is the same as the pose file order. Default is the order in the folder and should be fine if the files are named in the same way."

PREPROCESS_HELP = "Press this to create a new project with the above configurations. Overwrites prefixes with the same name!"

"""Feauture extraction"""

BEHAVIOR_COLOR_SELECT_HELP = "Rearrange and select colors for behavioral classes to help with visualization."

SPLIT_CLASSES_HELP = "Deselecting this will pool all behaviors into the same histogram."

BINS_SLIDER_HELP = "Select the number of bins to use. A higher number results in a better temporal resolution when deciding which minimum duration to pick."

MIN_DURATION_HELP = "The minimum duration a behavioral bout is happening at. This input will be translated into a time window for feature extraction." \
                    " It is recommended to select a duration that also incorporates short bouts. For example the 10% quantile."

RE_EXTRACT_HELP = "Select to enable to re-extract features based on the updated configuration."
EXTRACT_FEATURES_HELP = "Press to extract features based on the configuration above."

NUM_SPLITS_HELP = "Number of shuffled splits to use for cross-validation. "

"""Baseline classification"""

INIT_RATIO_HELP = "The initial sampling ratio is used to select a small training set for the initial classification (iteration 0)." \
                  " A small ratio can help to decrease the overall required samples. However, make sure that all classes have a minimum amount of samples."

MAX_ITER_HELP = "The maximum number of iterations during active learning. Note that active learning stops automatically if no new low-confidence predictions are found. "

MAX_SAMPLES_HELP = "The maximum number of samples across all classes that are taken from the pool of low-confidence predictions each iteration during active learning."

SHOW_FINAL_RESULTS_HELP = "Check to reveal results of the latest active learning cycle."

RE_CLASSIFY_HELP = "Check to re-classify your data with a new set of configurations. This will overwrite previous active learning cycles!"

"""Prediction"""
DIFFERENT_POSE_ORIGIN = "Check if the designated pose estimation files do not originate from the same provider (DLC, SLEAP)."
POSE_UPLOAD_HELP = "Upload pose estimation files to predict with the trained classifier. Default provider and file type is based on the project config."

"""View"""
VIEW_LOADER_HELP = "You can upload BORIS type files, including A-SOiD output from predictions to view as ethograms."

"""Unsupervised discovery"""
PREPARE_DATA_HELP = "Before we can run discovery, we need to extract the features from the full data set. This might take a while, but will only run once."
CLASS_SELECT_HELP = "Select a class for unsupervised clustering (UMAP + HDBscan). If available, the clustering from previous attemps will be visualized."
CLUSTER_RANGE_HELP = "The minimum cluster size that is found by HDBscan clustering. This range allows A-SOiD to look for the best min cluster size."
START_DISCOVERY_HELP = "Press to start the unsupervised clustering step. This includes rescaling, embedding, and clustering of the data, which can take a while..."
SUBCLASS_SELECT_HELP = "Select one or more sub-classes from the plot above to include in your new annotations."
SAVE_NEW_HELP = "Press to save your previous selection in the project folder."

"""Final"""

IMPRESS_TEXT = "A-SOiD was developed by Alexander Hsu and Jens Schweihoff. For support, visit our [GitHub Repository](https://github.com/YttriLab/A-SOID)."