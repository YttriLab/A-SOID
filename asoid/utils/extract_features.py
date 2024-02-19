import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats



from utils.extract_features_2D import feature_extraction
from utils.extract_features_3D import feature_extraction_3d

from utils.load_workspace import load_data, save_data
from config.help_messages import *


def interactive_durations_dist(targets, behavior_classes, framerate, plot_container,
                               num_bins, split_by_class=True):
    # Add histogram data
    plot_col, option_col = plot_container.columns([3, 1])
    option_col.write('')
    option_col.write('')
    option_col.write('')
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    with option_col:
        option_expander = st.expander("Configure plot")
        if split_by_class:
            behavior_colors = {k: [] for k in behavior_classes}
            if len(behavior_classes) == 4:
                default_colors = ["red", "darkorange", "dodgerblue", "gray"]
            else:
                np.random.seed(42)
                selected_idx = np.random.choice(np.arange(len(all_c_options)), len(behavior_classes), replace=False)
                default_colors = [all_c_options[s] for s in selected_idx]

            for i, class_id in enumerate(behavior_classes):
                behavior_colors[class_id] = option_expander.selectbox(f'Color for {behavior_classes[i]}',
                                                                      all_c_options,
                                                                      index=all_c_options.index(default_colors[i]),
                                                                      key=f'color_option{i}',
                                                                      help=BEHAVIOR_COLOR_SELECT_HELP)
            colors = [behavior_colors[class_id] for class_id in behavior_classes]
        else:
            behavior_colors = {k: [] for k in ['All']}
            default_colors = ['dodgerblue']
            for i, class_id in enumerate(['All']):
                behavior_colors[class_id] = option_expander.selectbox(f'Color for all',
                                                                      all_c_options,
                                                                      index=all_c_options.index(default_colors[i]),
                                                                      key=f'color_option{i}',
                                                                      help=BEHAVIOR_COLOR_SELECT_HELP)
            colors = [behavior_colors[class_id] for class_id in ['All']]

    duration_dict = {k: [] for k in behavior_classes}
    durations = []
    corr_targets = []
    for seq in range(len(targets)):
        durations.append(np.diff(np.hstack((0, np.where(np.diff(targets[seq]) != 0)[0] + 1))))
        corr_targets.append(targets[seq][
                                np.hstack((0, np.where(np.diff(targets[seq]) != 0)[0] + 1))][:durations[seq].shape[0]])
    for seq in range(len(durations)):
        current_seq_durs = durations[seq]
        for unique_beh in np.unique(np.hstack(corr_targets)):
            # make sure it's an int
            unique_beh = int(unique_beh)
            idx_behavior = np.where(corr_targets[seq] == unique_beh)[0]
            curr_annot = behavior_classes[unique_beh]
            if len(idx_behavior) > 0:
                duration_dict[curr_annot].append(current_seq_durs[np.where(corr_targets[seq] == unique_beh)[0]])
    keys = ['Sequence', 'Annotation', 'Duration (seconds)']
    data_dict = {k: [] for k in keys}
    for curr_annot in behavior_classes:
        for seq in range(len(duration_dict[curr_annot])):
            for bout, duration in enumerate(duration_dict[curr_annot][seq]):
                data_dict['Sequence'].append(seq)
                data_dict['Annotation'].append(curr_annot)
                data_dict['Duration (seconds)'].append(duration / framerate)

    df = pd.DataFrame(data_dict)
    if split_by_class:
        fig = px.histogram(df, x="Duration (seconds)", color='Annotation',
                           opacity=0.7,
                           nbins=num_bins,
                           marginal="box",
                           barmode='relative',
                           color_discrete_sequence=colors,
                           range_x=[0, np.percentile(np.hstack(data_dict['Duration (seconds)']), 99)],
                           hover_data=df.columns)
    else:
        fig = px.histogram(df, x="Duration (seconds)",
                           opacity=0.8,
                           nbins=num_bins,
                           marginal="box",
                           barmode='relative',
                           histnorm='probability',
                           color_discrete_sequence=colors,
                           range_x=[0, np.percentile(np.hstack(durations) / framerate, 99)],
                           hover_data=df.columns)
    fig.update_yaxes(linecolor='dimgray', gridcolor='dimgray')

    fig.update_layout(
        title="",
        xaxis_title=keys[2],
        legend_title=keys[1],
        font=dict(
            family="Arial",
            size=14,
            color="white"
        )
    )
    plot_col.plotly_chart(fig, use_container_width=True)
    return fig


class Extract:

    def __init__(self, working_dir, prefix, frames2integ, is_3d):

        self.working_dir = working_dir
        self.prefix = prefix
        self.project_dir = os.path.join(working_dir, prefix)
        self.iteration_0 = 'iteration-0'
        self.frames2integ = frames2integ
        self.is_3d = is_3d

        self.processed_input_data = None
        self.targets = None
        self.features = None
        self.scaled_features = None
        self.targets_mode = None
        self.scalar = None

        self.features_train = []
        self.targets_train = []
        self.features_heldout = []
        self.targets_heldout = []

    def extract_features(self):
        data, config = load_data(self.working_dir,
                                 self.prefix)
        # get relevant data from data file
        [self.processed_input_data,
         self.targets] = data
        # grab all 70 sequences
        number2train = len(self.processed_input_data)
        # extract features, bin them
        if self.is_3d:
            print('3D feature extraction')
            #3D feature extraction
            self.features, self.scaled_features = feature_extraction_3d(self.processed_input_data,
                                                                 number2train,
                                                                 self.frames2integ)
        else:
            print('2D feature extraction')
            #2D feature extraction
            self.features, self.scaled_features = feature_extraction(self.processed_input_data,
                                                                 number2train,
                                                                 self.frames2integ)

    def downsample_labels(self):
        num2skip = int(self.frames2integ / 10)  # 12
        targets_ls = []
        for i in range(len(self.targets)):
            targets_not_matching = np.hstack(
                [stats.mode(self.targets[i][(num2skip - 1) + num2skip * n:(num2skip - 1) + num2skip * n + num2skip])[0]
                 for n in range(len(self.targets[i]))])
            # features are skipped so if it's not multiple of 12, we discard the final few targets
            targets_matching_features = self.targets[i][(num2skip - 1):-1:num2skip]
            targets_ls.append(targets_not_matching[:targets_matching_features.shape[0]])
        self.targets_mode = np.hstack(targets_ls)
        if self.features.shape[0] > self.targets_mode.shape[0]:
            self.features = self.features[:self.targets_mode.shape[0]]
            # y = self.targets_mode.copy()
        elif self.features.shape[0] < self.features.shape[0]:
            # X = self.features.copy()
            self.targets_mode = self.targets_mode[:self.features.shape[0]]
        # else:
        #     X = self.features.copy()
        #     y = self.targets_mode.copy()

    def save_features_targets(self):
        save_data(self.project_dir, self.iteration_0, 'feats_targets.sav',
                  [
                      self.features,
                      self.targets_mode,
                      self.frames2integ
                  ])

    def main(self):
        self.extract_features()
        self.downsample_labels()
        self.save_features_targets()
        col_left, _, col_right = st.columns([1, 1, 1])
        col_right.success("Continue on with next module".upper())
