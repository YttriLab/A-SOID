import pandas as pd
import numpy as np
import h5py
import streamlit as st


def get_animals(df: pd.DataFrame, lvl=1):
    """Returns animal list from df in internal multiindex format. Excludes Label column"""

    animals = list(df.columns.get_level_values(lvl).unique())
    # remove label if there
    if "Label" in animals:
        animals.remove("Label")

    return animals


def get_bodyparts(df: pd.DataFrame, lvl=1):
    """Returns body part list from df in internal multiindex format. Excludes any empty columns"""
    bodyparts = list(df.columns.get_level_values(lvl).unique())
    # remove any empty columns: mainly to remove MultiIndexes that do have no first level in MultiIndex
    while "" in bodyparts:
        bodyparts.remove("")
    return bodyparts


def convert_bodyparts_to_columnname(bodyparts):
    """Takes list of str as input and converts it to column name in style bodypart_x/y"""
    bp_columns = []
    for bp in bodyparts:
        bp_columns.append(bp + "_x")
        bp_columns.append(bp + "_y")
        bp_columns.append(bp + "_likelihood")

    return bp_columns


def load_sleap_data(path, multi_animal=False):
    """loads sleap data h5 format from file path and returns it as pd.DataFrame
    As sleap body parts (nodes) are not ordered in a particular way, we sort them alphabetically.
    As sleap tracks do not export a score/likelihood but cut off automatically (nan), we are simulating a likelihood
     value for compatability with feature extraction functions"""
    # TODO: discriminate with multiple animal from single instance model output
    with h5py.File(path, "r") as f:
        # occupancy_matrix = f['track_occupancy'][:]
        # track_names = f['track_names'][:]
        tracks_matrix = f["tracks"][:].T
        animals = []
        temp_tracks = None
        for an in np.arange(tracks_matrix.shape[3]):
            animals.append('Animal{}'.format(an))
            if temp_tracks is None:
                temp_tracks = tracks_matrix[:, :, :, an]
            else:
                temp_tracks = np.concatenate([temp_tracks, tracks_matrix[:, :, :, an]], axis=1)

        bodyparts = f["node_names"][:].astype("U13")
    # preallocate array including dummy likelihood for similarity between dlc and sleap import
    # can be changed later if sleap files incorporate scores at any time in the future
    likelihood_matrix = np.ones((temp_tracks.shape[0], temp_tracks.shape[1], 1))
    sleap_matrix = np.concatenate((temp_tracks, likelihood_matrix), axis=2)
    # # reshape tracks into 2d format (x,y, likelihood)*len(bps)
    sleap_array = np.reshape(sleap_matrix, (sleap_matrix.shape[0], sleap_matrix.shape[1] * sleap_matrix.shape[2]))
    # create beautiful multiindex columns to easily sort by bodypart

    column_names = pd.MultiIndex.from_product([animals, bodyparts, ('x', 'y', 'likelihood')],
                                              names=['Animal', 'bodypart', 'coords'])

    # create dataframe from both
    df = pd.DataFrame(sleap_array, columns=column_names)
    # #change any nan values (lost tracking) to 0
    # TODO: confirm that this is the way to go
    for i, ani in enumerate(animals):
        for bp in bodyparts:
            # find any nan either in x or y coordinates. Usually they are paired, but you never know!
            any_nan = np.any(pd.isna(df[[(ani, bp, 'x'), (ani, bp, 'x')]]).values, axis=1)
            # get the indexes of the boolean mask
            nan_indexes = np.argwhere(any_nan).flatten()
            # set all nan index to 0
            df[(ani, bp, 'likelihood')][nan_indexes] = 0

    # SLEAP comes with NaN values, so we need to get rid of them!
    # first linear interpolation
    clean_df = df.interpolate(method="linear", limit_direction='forward')
    # backward fill with first valid entry to take care of NaNs in the beginning
    clean_df.fillna(method="bfill", inplace=True)
    # sort bodyparts by alphabet
    clean_df = clean_df.reindex(columns=sorted(bodyparts), level=1)
    return clean_df


def load_dlc_data(path, multi_animal=False):
    """loads dlc csv file format and returns it as pd.Dataframe"""
    if not multi_animal:
        df = pd.read_csv(path, header=[0, 1, 2], sep=",", index_col=0)
    else:
        df = pd.read_csv(path, header=[0, 1, 2, 3], sep=",", index_col=0)
    return df


def load_pose(path, origin, multi_animal=False):
    """General loading function. loads pose estimation file based on origin.
    :param multi_animal:
    :param path: str, full path to pose estimation file
    :param origin: str, origin of pose estimation file. 'DeepLabCut' or 'Sleap'"""

    if origin.lower() == 'deeplabcut':
        # file_j_df = pd.read_csv(filename, low_memory=False)
        df = load_dlc_data(path, multi_animal)
    elif origin.lower() == 'sleap':
        df = load_sleap_data(path, multi_animal)
    else:
        raise ValueError(f'Pose estimation file origin {origin} is not supported.')

    return df

def load_pose_ftype(path, ftype, multi_animal=False):
    """General loading function. loads pose estimation file based on ftype.
    Limited to csv for dlc and h5 for sleap
    :param multi_animal:
    :param path: str, full path to pose estimation file
    :param origin: str, origin of pose estimation file. 'DeepLabCut' or 'Sleap'"""

    if ftype.lower() == 'csv':
        # file_j_df = pd.read_csv(filename, low_memory=False)
        df = load_dlc_data(path, multi_animal)
    elif ftype.lower() == 'h5':
        df = load_sleap_data(path, multi_animal)
    else:
        raise ValueError(f'Pose estimation file type {ftype} is not supported.')

    return df


"""LABEL SECTION"""

def _load_boris_raw(file):
    """loads raw BORIS binary behavior table. Takes file from st.uploader as input"""
    if file.name.endswith("csv"):
        labels = pd.read_csv(file)
    elif file.name.endswith("tsv"):
        labels = pd.read_csv(file, sep='\t')
    else:
        raise ValueError(f"Label file {file.name} is not a csv or tsv file.")

    # all unlabeled are placed into "other"
    # create a new column called "other" if not existent yet
    if "other" not in labels.columns:
        labels["other"] = 0
    # find all unlabeled
    unlabeled_data = labels.drop(columns=["time"]).sum(axis=1) == 0
    # change all unlabeled to 1 in other
    labels.loc[unlabeled_data, "other"] = 1

    return labels

def resample_labels(labels: pd.DataFrame, fps: int, sample_rate:int):
    # upsample labels to fit with pose estimation info
    pose_sample_rate = 1 / fps
    # fixed critical error, BORIS does not always reset time if you label two observations at once
    label_sample_rate = 1/sample_rate
    resample_rate = label_sample_rate / pose_sample_rate
    #check if resample rate is divisible by fps
    if sample_rate % fps != 0 and fps % sample_rate != 0:
        raise st.error(f"Annotations cannot be resampled to fit pose estimation sampling. The annotation sample rate {sample_rate} is not divisible by FPS {fps} or vice versa.")

    if resample_rate > 1:
        # take the upsample rate and apply it to the samples
        re_labels = pd.DataFrame(
            np.repeat(labels.values, int(resample_rate), axis=0), columns=labels.columns
        )
    elif resample_rate < 1:
        #downsample
        downsample_rate = 1/resample_rate
        re_labels = labels.iloc[::int(downsample_rate),:]

    else:
        warnings.warn("Sample rate is equal to frame rate. No resampling necessary.")
        return labels
    # correct the time column
    re_labels["time"] = np.arange(0, re_labels.shape[0]) / fps
    # st.write(re_labels)
    return re_labels



def load_labels_boris_auto(path: str, fps: int = None):
    #TODO: Deprecate this function
    """loads labels from BORIS binary behavior table
    the labels file is generated by BORIS, which works in timesteps measured in increments of seconds, but not in frames
    this results in an alignment issue, beginning with the fact that we cannot seperate the output closer to fps then 3 decimal points.
    with 30 Hz aka 33.333333333333 ms , BORIS can do 0.033 s/step (33 ms) at best. This inaccuracy results in a mismatch of frames downstream.
    so we save the BORIS lables as 100 ms time steps
    Optional: upsample each step into 3 resulting in 30Hz (or any other desired framerate) samples."""
    #TODO: CHANGE TO USER INPUT ASKING FOR SAMPLING RATE; THEN throw error if not divisible by framerate
    if path.endswith("csv"):
        labels = pd.read_csv(path)
    elif path.endswith("tsv"):
        labels = pd.read_csv(path, sep='\t')
    else:
        raise ValueError(f"Label file {path} is not a csv or tsv file.")
    if fps is not None:
        # upsample labels to fit with pose estimation info
        sample_rate = 1 / fps
        # fixed critical error, BORIS does not always reset time if you label two observations at once
        time_step = labels["time"][1] - labels["time"][0]
        upsample_rate = time_step / sample_rate
    else:
        upsample_rate = 1

    if upsample_rate > 1:
        # take the upsample rate and apply it to the samples
        up_labels = pd.DataFrame(
            np.repeat(labels.values, int(upsample_rate), axis=0), columns=labels.columns
        )
        # correct the time column
        up_labels["time"] = np.arange(0, up_labels.shape[0]) / fps

    elif upsample_rate < 0.99:
        raise ValueError(f"Sample rate {time_step} of label file is to high with {fps} as video fps. Downsampling is not available at this point.")

    else:
        # nothing to do here
        up_labels = labels

    # all unlabeled are placed into "other"
    # create a new column called "other" if not existent yet
    if "other" not in up_labels.columns:
        up_labels["other"] = 0
    # find all unlabeled
    unlabeled_data = up_labels.drop(columns=["time"]).sum(axis=1) == 0
    # change all unlabeled to 1 in other
    up_labels.loc[unlabeled_data, "other"] = 1

    return up_labels


def load_labels(file, origin: str, fps: int, sample_rate: int):
    """Collective function to load from multiple origins.
    Right now, only includes BORIS labels

    :param file: File object from st.file_uploader
    """

    if origin.lower() == "boris":
        labels = _load_boris_raw(file)
    else:
        raise ValueError(f"Label origin {origin} is not supported.")
    #testing
    # upsample labels to test downsampling
    # st.write(labels.shape)
    # labels = pd.DataFrame(
    #         np.repeat(labels.values, 6, axis=0), columns=labels.columns
    #     )
    # st.write(labels)
    if sample_rate != fps:
        re_labels = resample_labels(labels, fps, sample_rate)
        return re_labels
    else:
        return labels



def load_labels_auto(path: str, origin: str, fps: int):
    #TODO: Deprecate this function
    """Collective function to load from multiple origins.
    Right now, only includes BORIS labels"""

    if origin.lower() == "boris":
        labels = load_labels_boris_auto(path, fps)
    else:
        raise ValueError(f"Label origin {origin} is not supported.")

    return labels

