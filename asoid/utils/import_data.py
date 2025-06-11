import pandas as pd
import numpy as np
import h5py
from scipy.io import loadmat
import warnings

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

    # likelihood filtering based on b-soid approach

    return df


def load_opm_data(path: str):
    """ Loads openmonkeypose data from mat or txt file and returns it as pd.DataFrame"""
    if path.name.endswith(".mat"):
        # mat file
        # might throw an error if matlab version is to high
        mat = loadmat(path)
        monkey_pose = pd.DataFrame(mat['coords'], columns=["idx", "x", "y", "z"])
    elif path.name.endswith(".txt"):
        # txt file
        monkey_pose = pd.read_csv(path, sep=" ", header=None, names=["idx", "x", "y", "z"])
    else:
        raise ValueError(f"OpenMonkeyPose file {path.name} is not a .mat or .txt file.")
    # each body part is represented by 3 columns (x, y, z), however the index is not unique because rows are repeated for each frame per bodypart
    # we need to reshape the dataframe to have a unique index for each body part
    n_frames, n_bodyparts = np.unique(monkey_pose["idx"], return_counts=True)
    assert np.all(n_bodyparts == n_bodyparts[0])
    # bodyparts are fixed in OpenMonkeyPose
    bodyparts_opm = "Nose, Head, Neck, RShoulder, RHand, LShoulder, LHand, Hip, RKnee, RFoot, LKnee, LFoot, Tail".split(", ")
    n_bodyparts = n_bodyparts[0]
    assert n_bodyparts == len(bodyparts_opm)
    n_frames = len(n_frames)
    # drop index
    monkey_pose["likelihood"] = 1
    monkey_pose = monkey_pose.drop(columns="idx", errors="ignore")

    monkey_pose = monkey_pose.values.reshape(
        (n_frames, n_bodyparts * 4))  # reshape to have a unique index for each body part x,y, z, likelihood

    monkey_pose = pd.DataFrame(monkey_pose, columns=pd.MultiIndex.from_product(
        [["OpenMonkeyStudio"]
            #, [f"bp_{num}" for num in range(n_bodyparts)]
            , bodyparts_opm
            , ["x", "y", "z", "likelihood"]]
        , names=["Animal", "bodypart", "coords"]))

    return monkey_pose

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
    elif origin.lower() == 'openmonkeystudio':
        if multi_animal:
            raise ValueError("A-SOiD + OpenMonkeyStudio does not support multi animal pose estimation ATM.")
        df = load_opm_data(path)
    else:
        raise ValueError(f'Pose estimation file origin {origin} is not supported.')

    return df

def load_pose_ftype(path, ftype, multi_animal=False):
    #TODO: Deprecate this function
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
        raise ValueError(f"Annotations cannot be resampled to fit pose estimation sampling. The annotation sample rate {sample_rate} is not divisible by FPS {fps} or vice versa.")

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

    return re_labels


def load_labels(file, origin: str, fps: int, sample_rate: int):
    """Collective function to load from multiple origins.
    Right now, only includes BORIS labels

    :param file: File object from st.file_uploader
    """

    if origin.lower() == "boris":
        labels = _load_boris_raw(file)
    else:
        raise ValueError(f"Label origin {origin} is not supported.")

    if sample_rate != fps:
        re_labels = resample_labels(labels, fps, sample_rate)
        return re_labels
    else:
        return labels

