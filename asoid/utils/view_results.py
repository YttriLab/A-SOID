import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime

from config.help_messages import VIEW_LOADER_HELP
from utils.import_data import load_labels


def label_blocks(df, clm_block):

    df_labeled = df.copy()
    df_labeled["block"] = (df_labeled[clm_block].shift(1) != df_labeled[clm_block]).astype(int).cumsum()

    return df_labeled


def get_events(df: pd.DataFrame, event_clm, label_clm, event):
    """
    This function returns a series of indexes (frames) where an event started (first TRUE value) and ended (first FALSE value)
    :param df: pd.Dataframe containing behavior labels
    :param event_clm: column name to use as status (TRUE/FALSE) in df
    :param label_clm: column name to use as block labels (1,2,3,4...) in df, will be used to identify start of each event
    :return: onset and offset index of events defined in event_clm and label_clm
    """
    event_df = df.copy()
    event_df[event] = event_df[event_clm] == event
    st.write(event_df)
    block_df = label_blocks(event_df, event)
    """Take trial labels to find start of every labeled block (Events and NonEvents) and drop all else"""
    unique_df = block_df.drop_duplicates(subset= ["block"])
    """Sort only for Event OnSets (True) and skip Event Ends (False)"""
    event_true = unique_df[event] == True
    event_false = ~event_true
    st.write(unique_df)
    """only take index"""
    onset_idx = unique_df[event_true].index
    offset_idx = unique_df[event_false].index
    return onset_idx, offset_idx

def get_block_boundaries(df, label_clm, cat_clm):
    """
    this function returns the start and stop of label block
    :param df: dataframe
    :param label_clm: column created by label_block, is searched for onset and offset of each block
    :param cat_clm: column used by label_block to look for blocks, will be used as key in block_dict
    :return: block_dict in style "block" = list(tuple(onset, offset), ...)
    """
    """Take trial labels to find start of every labeled block (Trial and NonTrials) and drop all else"""
    df_descend = df.sort_index(ascending= False) # flip to find last entry
    unique_df = df.drop_duplicates(subset= [label_clm]) # finds first entry
    unique_df_desc = df_descend.drop_duplicates(subset= [label_clm]) # finds last entry
    unique_df_desc = unique_df_desc.sort_index(ascending= True) # flip again
    block_dict = {}
    for block in unique_df[cat_clm].values:

        cluster_start = unique_df[cat_clm] == block
        cluster_stop = unique_df_desc[cat_clm] == block
        onset_idx = list(unique_df[cluster_start].index)
        offset_idx = list(unique_df_desc[cluster_stop].index)
        block_list = []
        for i in range(len(onset_idx)):
            block_list.append((onset_idx[i], offset_idx[i]))

        block_dict[block] = block_list

    return block_dict

def count_events(df_label):
    """ This function counts the number of events for each label in a dataframe"""
    df_label_cp = df_label.copy()
    # prepare event counter
    event_counter = pd.DataFrame(df_label_cp["labels"].unique(), columns=["labels"])
    event_counter["events"] = 0

    # Count the number of isolated blocks of labels for each unique label
    # go through each unique label and create a binary column
    for label in df_label_cp["labels"].unique():
        df_label_cp[label] = (df_label_cp["labels"] == label)
        df_label_cp[label].iloc[df_label_cp[label] == False] = np.NaN
        # go through each unique label and count the number of isolated blocks
        df_label_cp[f"{label}_block"] = np.where(df_label_cp[label].notnull(),
                                                 (df_label_cp[label].notnull() & (df_label_cp[label] != df_label_cp[
                                                     label].shift())).cumsum(),
                                                 np.nan)
        event_counter["events"].iloc[event_counter["labels"] == label] = df_label_cp[f"{label}_block"].max()

    return event_counter



class Viewer:

    def __init__(self, config = None):
        self.label_files = None
        self.label_csvs = None

        if config is not None:
            self.working_dir = config["Project"].get("PROJECT_PATH")
            self.prefix = config["Project"].get("PROJECT_NAME")
            self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
            self.framerate = config["Project"].getint("FRAMERATE")
            self.duration_min = config["Processing"].getfloat("MIN_DURATION")

            self.class_to_number = {s: i for i, s in enumerate(self.annotation_classes)}
            self.number_to_class = {i: s for i, s in enumerate(self.annotation_classes)}


        else:
            self.working_dir = None
            self.prefix = None
            self.annotation_classes = None
            self.framerate = None
            self.duration_min = None

            self.class_to_number = None
            self.number_to_class = None


        pass

    def upload_labels(self):

        upload_container = st.container()

        self.label_files = upload_container.file_uploader('Upload annotation or classification files that you want to view',
                                                             accept_multiple_files=True
                                                             ,type="csv"
                                                             ,key='label'
                                                             ,help=VIEW_LOADER_HELP)

    def convert_dummies_to_labels(self, labels):
        """
        This function converts dummy variables to labels
        :param labels: pandas dataframe with dummy variables
        :return: pandas dataframe with labels and codes
        """
        conv_labels = pd.from_dummies(labels)
        cat_df = pd.DataFrame(conv_labels.values, columns=["labels"])
        if self.annotation_classes is not None:
            cat_df["labels"] = pd.Categorical(cat_df["labels"] , ordered=True, categories=self.annotation_classes)
        else:
            cat_df["labels"] = pd.Categorical(cat_df["labels"], ordered=True, categories=cat_df["labels"].unique())
        cat_df["codes"] = cat_df["labels"].cat.codes

        return cat_df

    def prep_labels_single(self, labels):
        """
        This function loads the labels from a single file and prepares them for plotting
        :param labels: pandas dataframe with labels
        :return: pandas dataframe with labels
        """
        labels = labels.drop(columns=["time"], errors="ignore")
        labels = self.convert_dummies_to_labels(labels)

        return labels


    def plot_ethogram_single(self,df_label):
        """ This function plots the labels in a heatmap"""

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
        st.plotly_chart(fig, use_container_width=True)

    def plot_heatmap_single(self,df_label):
        """ This function plots the labels in a heatmap"""
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


        st.plotly_chart(fig, use_container_width=True)



    def describe_labels_single(self, df_label):
        """ This function describes the labels in a table"""

        event_counter = count_events(df_label)

        count_df = df_label.value_counts().to_frame().reset_index().rename(columns={0: "frame count"})
        # heatmap already shows this information
        #count_df["percentage"] = count_df["frame count"] / count_df["frame count"].sum() *100
        self.framerate = 30
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
        st.dataframe(count_df)

    def main(self):

        self.upload_labels()
        self.label_csvs = {}
        if self.label_files:
            for file in self.label_files:
                file.seek(0)
                temp_name = file.name
                labels = load_labels(file,origin = "BORIS", fps = self.framerate)
                self.label_csvs[temp_name] = labels

            for num, f_name in enumerate(self.label_csvs.keys()):

                with st.expander(label = f_name ):
                    try:
                        single_label = self.prep_labels_single(self.label_csvs[f_name])
                        st.subheader("Ethogram")
                        self.plot_ethogram_single(single_label)
                        st.subheader("Label statistics")
                        heatmap_clm, desc_clm = st.columns([0.4, 0.7])
                        with heatmap_clm:
                            self.plot_heatmap_single(single_label)
                        with desc_clm:
                            st.write("") #empty line to align the two columns
                            st.write("Statistics")
                            self.describe_labels_single(single_label)

                    except ValueError as e:
                        st.error(e)
                        st.warning("This error is likely due to the labels not being in the correct format."
                                         " Please check that the labels are in the correct format and are exclusive"
                                         " - i.e., each row has only one label.")

