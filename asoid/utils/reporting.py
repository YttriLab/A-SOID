import pandas as pd
import numpy as np
import datetime

pd.options.mode.copy_on_write = True # only for pandas< 3.0

#TODO: fix this, count df only shows nans

def convert_dummies_to_labels(labels_df, annotation_classes):
    """
    This function converts dummy variables to labels
    :param labels: pandas dataframe with dummy variables
    :return: pandas dataframe with labels and codes
    """
    labels = labels_df.copy()
    conv_labels = pd.from_dummies(labels)
    cat_df = pd.DataFrame(conv_labels.values, columns=["labels"])
    if annotation_classes is not None:
        cat_df["labels"] = pd.Categorical(cat_df["labels"], ordered=True, categories=annotation_classes)
    else:
        cat_df["labels"] = pd.Categorical(cat_df["labels"], ordered=True, categories=cat_df["labels"].unique())
    cat_df["codes"] = cat_df["labels"].cat.codes

    return cat_df

def prep_labels_single(label_df, annotation_classes):
    """
    This function loads the labels from a single file and prepares them for plotting
    :param labels: pandas dataframe with labels
    :return: pandas dataframe with labels
    """
    labels = label_df.copy()
    labels = labels.drop(columns=["time"], errors="ignore")
    labels = convert_dummies_to_labels(labels, annotation_classes)

    return labels

def count_events(df_label, annotation_classes):
    """ This function counts the number of events for each label in a dataframe"""
    df_label_cp = df_label.copy()
    # prepare event counter
    # event_counter = pd.DataFrame(df_label_cp["labels"].unique(), columns=["labels"])
    event_counter = pd.DataFrame(annotation_classes, columns=["labels"])
    event_counter["events"] = 0

    # Count the number of isolated blocks of labels for each unique label
    # go through each unique label and create a binary column
    for label in annotation_classes:

        df_label_cp[label] = (df_label_cp["labels"] == label)
        # df_label_cp[label].iloc[df_label_cp[label] == False] = np.NaN
        df_label_cp[label].iloc[df_label_cp[label] == False] = np.NaN
        # go through each unique label and count the number of isolated blocks
        df_label_cp[f"{label}_block"] = np.where(df_label_cp[label].notnull(),
                                                 (df_label_cp[label].notnull() & (df_label_cp[label] != df_label_cp[
                                                     label].shift())).cumsum(),
                                                 np.nan)
        event_counter["events"].iloc[event_counter["labels"] == label] = df_label_cp[f"{label}_block"].max()

    return event_counter
def extract_descriptors(label_df, annotation_classes, framerate):
    """ This function extracts the descriptors from the labels
    :param label_df: pandas dataframe with labels
    :param annotation_classes: list of annotation classes
    :param framerate: framerate of the video
    :return: pandas dataframe with descriptors
    """
    df_label = label_df.copy()
    print(df_label.head())
    # df_label = prep_labels_single(df_label, annotation_classes)
    event_counter = count_events(df_label, annotation_classes)
    count_df = df_label.value_counts().to_frame().reset_index()
    #in some cases the column is called "count" in others it is called 0
    count_df.rename(columns={"count": "frame count"}, inplace=True, errors="ignore")
    count_df.rename(columns={0: "frame count"}, inplace=True, errors="ignore")
    # heatmap already shows this information
    # count_df["percentage"] = count_df["frame count"] / count_df["frame count"].sum() *100
    if framerate is not None:
        count_df["total duration"] = count_df["frame count"] / framerate
        count_df["total duration"] = count_df["total duration"].apply(lambda x: str(datetime.timedelta(seconds=x)))
    # event counter goes sequential order, but frame count is sorted already...
    count_df.set_index("codes", inplace=True)
    count_df.sort_index(inplace=True)
    count_df["bouts"] = event_counter["events"]

    # rename all columns to include their units
    count_df.rename(columns={"bouts": "bouts [-]",
                             "frame count": "frame count [-]",
                             "total duration": "total duration [hh:mm:ss]",
                             "percentage": "percentage [%]",

                             },
                    inplace=True)

    return count_df


