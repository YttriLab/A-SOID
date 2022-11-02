import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.colors as mcolors


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

        else:
            self.working_dir = None
            self.prefix = None
            self.annotation_classes = None
            self.framerate = None
            self.duration_min = None

        pass

    def upload_labels(self):

        upload_container = st.container()

        self.label_files = upload_container.file_uploader('Upload annotation or classification files that you want to view',
                                                             accept_multiple_files=True
                                                             ,type="csv"
                                                             ,key='label'
                                                             ,help=VIEW_LOADER_HELP)


    def plot_labels_matplotlib(self, labels):
        params = {"ytick.color": "w",
                  "xtick.color": "w",
                  "axes.labelcolor": "w",
                  "axes.edgecolor": "w"}
        plt.rcParams.update(params)

        #time_delta = labels["time"].iloc[1]
        labels = labels.drop(columns=["time"], errors="ignore")
        classes = list(labels.columns)
        label_names = np.argmax(labels.values, axis=1)
        # dictionary for code and corresponding labels
        assigned_labels = dict(zip(label_names,classes))
        # plot ethogram
        # plot them for comparison
        fig,ax = plt.subplots(1,figsize=(9,3))

        cmap = cm.get_cmap('tab10',len(classes))
        ethogram = plt.imshow(label_names[None, :]
                              ,aspect="auto"
                              ,cmap=cmap,interpolation="nearest"
                            )
        ethogram.set_clim(0,len(classes))
        plt.xlabel("Frames")
        plt.yticks([])
        cbar = plt.colorbar(ethogram)

        cbar.set_ticks(np.arange(0,len(classes)) + 0.5)
        cbar.set_ticklabels(classes)

        plt.tight_layout()
        st.pyplot(fig, facecolor= "black", transparent=True)

    def plot_labels_plotly(self,labels):

        time_delta = labels["time"].iloc[1]
        labels = labels.drop(columns=["time"],errors="ignore")

        cat_df = pd.from_dummies(labels)
        cat_df["label"] = pd.Categorical(cat_df[cat_df.columns[0]])
        cat_df["codes"] = cat_df["label"].cat.codes

        classes = list(labels.columns)
        test_view = labels.values * cat_df["codes"].values[:, None]
        #test_view[test_view == 0] = -1
        #fig = make_subplots(1,2)
        fig = px.imshow(test_view.T
                        , aspect= "auto"
                        , color_continuous_scale='Edge'
                        #,contrast_rescaling='infer'
                        ,y = classes
                        ,x = np.arange(labels.shape[0])*time_delta /60
                        #, zmin = 1
                        #,binary_string=True
                         )
        #fig.add_trace(go.Image(z = test_view),1,1)
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(fig,use_container_width=False)

    # def plot_labels(self,labels):
    #     plot_cont = st.container()
    #     time_delta = labels["time"].iloc[1]
    #     st.write(time_delta)
    #     labels = labels.drop(columns=["time"],errors="ignore")
    #     classes = list(labels.columns)
    #     cat_df = pd.from_dummies(labels)
    #     #time_line = pd.date_range("00:00:00", periods= cat_df.shape[0],freq=f"{time_delta}L").time
    #     # plot histogram
    #     #cat_df.index = time_line
    #
    #     cat_df["label"] = pd.Categorical(cat_df[cat_df.columns[0]])
    #     cat_df["codes"] = cat_df["label"].cat.codes
    #     cmap = cm.get_cmap('tab10',len(classes))
    #     #st.write(cat_df["label"].cat.codes)
    #     #extract blocks of same labels
    #     #block_df = label_blocks(cat_df, "label")
    #     #st.write(block_df)
    #     behavior_dict = {}
    #     for behavior in classes:
    #         onset, offset = get_events(cat_df, "label", "block", "Walk")
    #         behavior_dict[behavior] = [onset, offset]
    #
    #     st.write(behavior_dict)
    #
    #     block_dict = get_block_boundaries()
    #
    #
    #     #find start of each, stop is the index before the next start
    #
    #     # figures = [
    #     #     px.histogram(cat_df, histnorm='percent'),
    #     #     px.histogram(cat_df, histnorm='percent')
    #     # ]
    #     #
    #     #
    #     # fig = make_subplots(rows=1,cols=len(figures), shared_yaxes=True,)
    #     #
    #     # for i,figure in enumerate(figures):
    #     #
    #     #     for trace in range(len(figure["data"])):
    #     #         fig.add_trace(trace= figure["data"][trace],row= 1,col=i+1)
    #     #         fig.update_xaxes(title_text="Behavior Classes",row=1,col=i+1)
    #     #
    #     # fig.update_layout(yaxis_title="Percentage [%]", height=400, width=600)
    #     st.write(cat_df.shape)
    #     fig = px.pie(cat_df,values='codes',names='label',color='label')
    #     #fig = px.imshow(labels.values,binary_string=True, aspect= "auto")
    #     #fig = px.imshow(cat_df["codes"][None,:], aspect= "auto")
    #     #fig.update_yaxes(showticklabels = False)
    #
    #     plot_cont.plotly_chart(fig,use_container_width=False)

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
                    self.plot_labels_matplotlib(self.label_csvs[f_name])


