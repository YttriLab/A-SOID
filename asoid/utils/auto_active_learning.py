import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score
import matplotlib.colors as mcolors
from utils.predict import frameshift_predict, bsoid_predict_numba, bsoid_predict_numba_noscale
from utils.load_workspace import load_features, load_test, save_data


def show_classifier_results(behavior_classes, all_score,
                            base_score, base_annot,
                            learn_score, learn_annot):
    plot_col, option_col = st.columns([3, 1])
    option_col.write('')
    option_col.write('')
    option_col.write('')
    option_col.write('')
    option_col.write('')
    option_col.write('')
    option_expander = option_col.expander("Configure Plot")
    behavior_colors = {k: [] for k in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())

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
                                                              key=f'color_option{i}')
    keys = ['Behavior', 'Performance %', 'Iteration #']
    perf_by_class = {k: [] for k in behavior_classes}
    # st.write(base_score, learn_score)
    scores = np.vstack((np.hstack(base_score), np.vstack(learn_score)))
    # st.write(scores)
    mean_scores = [100 * round(np.mean(scores[j], axis=0), 2) for j in range(len(scores))]
    mean_scores2beat = np.mean(all_score, axis=0)
    scores2beat_byclass = all_score.copy()
    for c, c_name in enumerate(behavior_classes):
        if c_name != behavior_classes[-1]:
            for it in range(scores.shape[0]):
                perf_by_class[c_name].append(100 * round(scores[it][c], 2))
    fig = make_subplots(rows=2, cols=1, row_width=[0.2, 0.6]
                        )
    fig.add_scatter(y=np.repeat(100 * round(mean_scores2beat, 2), scores.shape[0]),
                    mode='lines',
                    marker=dict(color='white', opacity=0.1),
                    name='average (full data)',
                    row=1, col=1
                    )

    for c, c_name in enumerate(behavior_classes):
        if c_name != behavior_classes[-1]:
            fig.add_scatter(y=perf_by_class[c_name], mode='lines+markers',
                            marker=dict(color=behavior_colors[c_name]), name=c_name,
                            row=1, col=1
                            )
            fig.add_scatter(y=np.repeat(100 * round(scores2beat_byclass[c], 2), scores.shape[0]), mode='lines',
                            marker=dict(color=behavior_colors[c_name]), name=str.join('', (c_name, ' (full data)')),
                            row=1, col=1
                            )
    fig.add_scatter(y=mean_scores, mode='lines+markers',
                    marker=dict(color='gray', opacity=0.8),
                    name='average',
                    row=1, col=1
                    )

    fig.update_xaxes(range=[-.5, len(scores) - .5],
                     linecolor='dimgray', gridcolor='dimgray')
    fig.for_each_trace(
        lambda trace: trace.update(line=dict(width=2, dash="dot"))
        if trace.name.endswith('with full data')
        else (trace.update(line=dict(width=2))),
    )

    # counts
    base_counts = np.hstack([len(np.where(base_annot == b)[0]) for b in np.unique(base_annot)])
    learn_counts = np.vstack([np.hstack([len(np.where(learn_annot[it] == b)[0])
                                         for b in np.unique(learn_annot[it])])
                              for it in range(len(learn_annot))])
    train_counts = np.vstack((base_counts, learn_counts))
    stackData = {
        c_name: train_counts[:, c] for c, c_name in enumerate(behavior_classes) if c_name != behavior_classes[-1]
    }

    for c, c_name in enumerate(behavior_classes):
        if c_name != behavior_classes[-1]:
            fig.add_trace(go.Bar(x=np.arange(len(train_counts)),
                                 y=stackData[c_name], name=c_name,
                                 marker=dict(color=behavior_colors[c_name])), row=2, col=1,
                          )

    fig.update_layout(barmode='stack')
    fig.update_yaxes(linecolor='dimgray', gridcolor='dimgray')
    fig.update_layout(
        title="",
        xaxis_title=keys[2],
        yaxis_title=keys[1],
        legend_title=keys[0],
        autosize=False,
        width=800,
        height=500,
        font=dict(
            family="Arial",
            size=14,
            color="white"
        )
    )

    plot_col.plotly_chart(fig, use_container_width=False)
    return fig


class RF_Classify:

    def __init__(self, working_dir, prefix, iteration_dir, software,
                 init_ratio, max_iter, max_samples_iter,
                 annotation_classes, exclude_other, conf_threshold: float = 0.5):
        self.container = st.container()
        self.placeholder = self.container.empty()
        self.working_dir = working_dir
        self.prefix = prefix
        self.project_dir = os.path.join(working_dir, prefix)
        self.iter_dir = iteration_dir
        self.software = software
        self.conf_threshold = conf_threshold
        self.init_ratio = init_ratio
        self.max_iter = max_iter
        self.max_samples_iter = max_samples_iter
        self.annotation_classes = annotation_classes
        # self.targets_test = targets_test
        self.features_train = []
        self.targets_train = []
        self.features_heldout = []
        self.targets_heldout = []

        # get label code for last class ('other') to exclude later on if applicable
        self.exclude_other = exclude_other
        # self.label_code_other = max(np.unique(np.hstack(self.targets_heldout)))
        self.label_code_other = max(np.arange(len(annotation_classes)))
        self.frames2integ = None

        self.all_model = None
        self.all_X_train = None
        self.all_Y_train = None
        self.all_f1_scores = None
        self.all_macro_scores = None
        self.all_predict_prob = None

        self.iter0_model = None
        self.iter0_X_train = None
        self.iter0_Y_train = None
        self.iter0_f1_scores = None
        self.iter0_macro_scores = None
        self.iter0_predict_prob = None

        self.iterX_model = None
        self.iterX_X_train_list = []
        self.iterX_Y_train_list = []
        self.iterX_f1_scores_list = []
        self.iterX_macro_scores_list = []
        self.iterX_predict_prob_list = []
        self.sampled_idx_list = []

        self.keys = ['Behavior', 'Performance %', 'Iteration #']
        self.perf_by_class = {k: [] for k in annotation_classes}
        self.perf2beat_by_class = {k: [] for k in annotation_classes}

    def split_data(self):
        [X, y, self.frames2integ] = \
            load_features(self.project_dir, self.iter_dir)
        # st.write(features.shape, targets.shape)
        # partitioning into N randomly selected train/test splits

        self.features_train, self.features_heldout, \
            self.targets_train, self.targets_heldout = train_test_split(X, y, test_size=0.20, random_state=42)

        # seeds = np.arange(shuffled_splits)
        # # st.write(seeds)
        # if features.shape[0] > targets.shape[0]:
        #     X = features[:targets.shape[0]].copy()
        #     y = targets.copy()
        # elif features.shape[0] < targets.shape[0]:
        #     X = features.copy()
        #     y = targets[:features.shape[0]].copy()
        # else:
        #     X = features.copy()
        #     y = targets.copy()
        # for seed in stqdm(seeds, desc="Randomly partitioning into 80/20..."):
        #     X_train, X_test, y_train, y_test = train_test_split(X, y,
        #                                                         test_size=0.20, random_state=seed)
        #     self.features_train.append(X_train)
        #     self.targets_train.append(y_train)
        #     self.features_heldout.append(X_test)
        #     self.targets_heldout.append(y_test)

    def subsampled_classify(self):
        self.split_data()

        unique_classes = np.unique(np.hstack([np.hstack(self.targets_train), np.hstack(self.targets_heldout)]))
        # remove other if exclude other
        if self.exclude_other:
            unique_classes = unique_classes[unique_classes != self.label_code_other]
        X_all = []
        Y_all = []
        # go through each class and select the all samples from the features and targets
        for sample_label in unique_classes:
            X_all.append(self.features_train[self.targets_train == sample_label][:])
            Y_all.append(self.targets_train[self.targets_train == sample_label][:])
        X_all_train = np.vstack(X_all)
        Y_all_train = np.hstack(Y_all)
        self.all_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                criterion='gini',
                                                class_weight='balanced_subsample'
                                                )
        self.all_model.fit(X_all_train, Y_all_train)
        predict = bsoid_predict_numba_noscale([self.features_heldout], self.all_model)
        predict = np.hstack(predict)

        # check f1 scores per class, always exclude other (unlabeled data)
        self.all_f1_scores = f1_score(
            self.targets_heldout[self.targets_heldout != self.label_code_other],
            predict[self.targets_heldout != self.label_code_other],
            average=None)
        # check f1 scores overall
        self.all_macro_scores = f1_score(
            self.targets_heldout[self.targets_heldout != self.label_code_other],
            predict[self.targets_heldout != self.label_code_other],
            average='macro')
        self.all_predict_prob = self.all_model.predict_proba(
            self.features_train[self.targets_train != self.label_code_other])
        self.all_X_train = X_all_train
        self.all_Y_train = Y_all_train

        X = []
        Y = []

        # find the available amount of samples in the trainset,
        # take only the initial ratio and only classes that are in test
        # this returns 0 for samples that are not available
        samples2train = [int(np.sum(self.targets_train == b) * self.init_ratio)
                         for b in unique_classes]

        # go through each class and select the number of samples from the features and targets
        for n_samples, sample_label in zip(samples2train, unique_classes):
            # if there are samples in the train
            if n_samples > 0:
                X.append(self.features_train[self.targets_train == sample_label][:n_samples])
                Y.append(self.targets_train[self.targets_train == sample_label][:n_samples])

        X_train = np.vstack(X)
        Y_train = np.hstack(Y)
        self.iter0_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                  criterion='gini',
                                                  class_weight='balanced_subsample'
                                                  )
        self.iter0_model.fit(X_train, Y_train)
        predict = bsoid_predict_numba_noscale([self.features_heldout], self.iter0_model)
        predict = np.hstack(predict)

        # check f1 scores per class
        self.iter0_f1_scores = f1_score(
            self.targets_heldout[self.targets_heldout != self.label_code_other],
            predict[self.targets_heldout != self.label_code_other],
            average=None)

        # check f1 scores overall
        self.iter0_macro_scores = f1_score(
            self.targets_heldout[self.targets_heldout != self.label_code_other],
            predict[self.targets_heldout != self.label_code_other],
            average='macro')
        self.iter0_predict_prob = self.iter0_model.predict_proba(
            self.features_train[self.targets_train != self.label_code_other])
        self.iter0_X_train = X_train
        self.iter0_Y_train = Y_train

        # for i in range(len(self.features_train)):
        #     X_all = []
        #     Y_all = []
        #
        #     unique_classes = np.unique(np.hstack([np.hstack(self.targets_train), np.hstack(self.targets_heldout)]))
        #     # remove other if exclude other
        #     if self.exclude_other:
        #         unique_classes = unique_classes[unique_classes != self.label_code_other]
        #
        #     # go through each class and select the all samples from the features and targets
        #     for sample_label in unique_classes:
        #             X_all.append(self.features_train[i][self.targets_train[i] == sample_label][:])
        #             Y_all.append(self.targets_train[i][self.targets_train[i] == sample_label][:])
        #
        #     X_all_train = np.vstack(X_all)
        #     Y_all_train = np.hstack(Y_all)
        #     self.all_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
        #                                             criterion='gini',
        #                                             class_weight='balanced_subsample'
        #                                             )
        #     self.all_model.fit(X_all_train, Y_all_train)
        #     # if not self.software == 'CALMS21 (PAPER)':
        #     #     # predict = frameshift_predict(data_test, len(data_test), scalar,
        #     #     #                              self.all_model, framerate=self.frames2integ)
        #     #     predict = bsoid_predict_numba_noscale([self.features_heldout[i]], self.all_model)
        #     #     predict = np.hstack(predict)
        #     # else:
        #     #     predict = frameshift_predict(data_test, len(data_test), scalar,
        #     #                                  self.all_model, framerate=120)
        #     #
        #     predict = bsoid_predict_numba_noscale([self.features_heldout[i]], self.all_model)
        #     predict = np.hstack(predict)
        #
        #     # check f1 scores per class, always exclude other (unlabeled data)
        #     self.all_f1_scores.append(f1_score(
        #         self.targets_heldout[i][self.targets_heldout[i] != self.label_code_other],
        #         predict[self.targets_heldout[i] != self.label_code_other],
        #         average=None))
        #     # check f1 scores overall
        #     self.all_macro_scores.append(f1_score(
        #         self.targets_heldout[i][self.targets_heldout[i] != self.label_code_other],
        #         predict[self.targets_heldout[i] != self.label_code_other],
        #         average='macro'))
        #     self.all_predict_prob.append(
        #         self.all_model.predict_proba(self.features_train[i][self.targets_train[i] != self.label_code_other]
        #                                      ))
        #     self.all_X_train.append(X_all_train)
        #     self.all_Y_train.append(Y_all_train)

        # for i in range(len(self.features_train)):
        # X = []
        # Y = []
        #
        # # find the available amount of samples in the trainset,
        # # take only the initial ratio and only classes that are in test
        # # this returns 0 for samples that are not available
        # samples2train = [int(np.sum(self.targets_train[i] == b) * self.init_ratio)
        #                  for b in unique_classes]
        #
        # # go through each class and select the number of samples from the features and targets
        # for n_samples, sample_label in zip(samples2train, unique_classes):
        #     # if there are samples in the train
        #     if n_samples > 0:
        #         X.append(self.features_train[i][self.targets_train[i] == sample_label][:n_samples])
        #         Y.append(self.targets_train[i][self.targets_train[i] == sample_label][:n_samples])
        #
        # X_train = np.vstack(X)
        # Y_train = np.hstack(Y)
        # self.iter0_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
        #                                           criterion='gini',
        #                                           class_weight='balanced_subsample'
        #                                           )
        # self.iter0_model.fit(X_train, Y_train)
        # # test on remaining held out data
        # # if not self.software == 'CALMS21 (PAPER)':
        # #     # predict = frameshift_predict(data_test, len(data_test), scalar,
        # #     #                              self.iter0_model, framerate=self.frames2integ)
        # #     predict = bsoid_predict_numba_noscale([self.features_heldout[i]], self.iter0_model)
        # #     predict = np.hstack(predict)
        # # else:
        # #     predict = frameshift_predict(data_test, len(data_test), scalar,
        # #                                  self.iter0_model, framerate=120)
        #
        # predict = bsoid_predict_numba_noscale([self.features_heldout[i]], self.iter0_model)
        # predict = np.hstack(predict)
        #
        # # check f1 scores per class
        # self.iter0_f1_scores.append(f1_score(
        #     self.targets_heldout[i][self.targets_heldout[i] != self.label_code_other],
        #     predict[self.targets_heldout[i] != self.label_code_other],
        #     average=None))
        #
        # # check f1 scores overall
        # self.iter0_macro_scores.append(f1_score(
        #     self.targets_heldout[i][self.targets_heldout[i] != self.label_code_other],
        #     predict[self.targets_heldout[i] != self.label_code_other],
        #     average='macro'))
        # self.iter0_predict_prob.append(self.iter0_model.predict_proba(
        #     self.features_train[i][self.targets_train[i] != self.label_code_other]
        #     ))
        # self.iter0_X_train.append(X_train)
        # self.iter0_Y_train.append(Y_train)

    def show_subsampled_performance(self):
        behavior_classes = self.annotation_classes
        all_c_options = list(mcolors.CSS4_COLORS.keys())
        if len(behavior_classes) == 4:
            default_colors = ["red", "darkorange", "dodgerblue", "gray"]
        else:
            np.random.seed(42)
            selected_idx = np.random.choice(np.arange(len(all_c_options)), len(behavior_classes), replace=False)
            default_colors = [all_c_options[s] for s in selected_idx]
        mean_scores2beat = np.mean(self.all_f1_scores, axis=0)
        for c, c_name in enumerate(behavior_classes):
            if c_name != behavior_classes[-1]:
                self.perf_by_class[c_name].append(int(100 * round(self.iter0_f1_scores[c], 2)))
                self.perf2beat_by_class[c_name].append(int(100 * round(self.all_f1_scores[c], 2)))
        fig = make_subplots(rows=1, cols=1)
        for c, c_name in enumerate(behavior_classes):
            if c_name != behavior_classes[-1]:
                fig.add_scatter(y=self.perf_by_class[c_name], mode='markers',
                                marker=dict(color=default_colors[c]), name=c_name,
                                row=1, col=1
                                )
        fig.add_scatter(y=np.repeat(100 * round(mean_scores2beat, 2), self.max_iter + 1),
                        mode='lines',
                        marker=dict(color='white', opacity=0.1),
                        name='average (full data)',
                        row=1, col=1
                        )
        fig.update_xaxes(range=[-.5, self.max_iter + .5],
                         linecolor='dimgray', gridcolor='dimgray')
        fig.update_yaxes(ticksuffix="%", linecolor='dimgray', gridcolor='dimgray')
        fig.for_each_trace(
            lambda trace: trace.update(line=dict(width=2, dash="dot"))
            if trace.name == "average (full data)"
            else (trace.update(line=dict(width=2))),
        )
        fig.update_layout(
            title="",
            xaxis_title=self.keys[2],
            yaxis_title=self.keys[1],
            legend_title=self.keys[0],
            font=dict(
                family="Arial",
                size=14,
                color="white"
            )
        )
        self.placeholder.plotly_chart(fig, use_container_width=True)

    def base_classification(self):
        with st.spinner("Subsampled classfication..."):
            # print("Subsampled classfication...")
            self.subsampled_classify()

        # print("Showing subsampled performance...")
        with st.spinner("Preparing plot..."):
            self.show_subsampled_performance()
        with st.spinner("Saving training data..."):
            self.save_all_train_info()
        with st.spinner("Saving subsampled data..."):
            self.save_subsampled_info()
        # print("All done.")

    def self_learn(self):
        # X_train = dict()
        # Y_train = dict()
        # iterX_predict_prob = dict()
        # iterX_macro_scores = dict()
        # iterX_f1_scores = dict()
        for it in range(self.max_iter):
            with st.spinner(f'Training iteration {it + 1}...'):
                # st.info(f'Training iteration {it + 1}...'.upper())
                # X_train_it = []
                # Y_train_it = []
                # iterX_predict_prob_it = []
                # iterX_macro_scores_it = []
                # iterX_f1_scores_it = []
                sampled_idx = []
                if it == 0:
                    # start with iter0 data (retrieve from above)
                    X_train = self.iter0_X_train
                    Y_train = self.iter0_Y_train
                    # retrieve iteration 0 predict probability
                    idx_lowconf = np.where(self.iter0_predict_prob.max(1) < self.conf_threshold)[0]
                    # identify all features/targets that were low predict prob
                    new_X_human = self.features_train[self.targets_train != self.label_code_other][
                                  idx_lowconf, :]
                    new_Y_human = self.targets_train[self.targets_train != self.label_code_other][
                        idx_lowconf]
                else:
                    idx_lowconf = np.where(self.iterX_predict_prob_list[it - 1].max(1) < self.conf_threshold)[0]
                    new_X_human = self.features_train[self.targets_train != self.label_code_other][
                                  idx_lowconf, :]
                    new_Y_human = self.targets_train[self.targets_train != self.label_code_other][
                        idx_lowconf]
                np.random.seed(42)
                try:
                    # attempt sampling up to max_samples_per iteration
                    idx_sampled = np.random.choice(np.arange(idx_lowconf.shape[0]),
                                                   self.max_samples_iter, replace=False)
                except:
                    # otherwise just grab all
                    try:
                        idx_sampled = np.random.choice(np.arange(idx_lowconf.shape[0]),
                                                       idx_lowconf.shape[0], replace=False)
                    except:
                        break

                new_X_sampled = new_X_human[idx_sampled, :]
                new_Y_sampled = new_Y_human[idx_sampled]
                sampled_idx.append(idx_lowconf[idx_sampled])

                if it == 0:
                    # if iteration 1, we use iteration 0 as base, and append new samples
                    X_train = np.vstack(
                        (X_train, new_X_sampled))
                    Y_train = np.hstack(
                        (Y_train, new_Y_sampled))
                else:
                    # if iteration >1, we use previous iteration as base, and append new samples
                    X_train = np.vstack(
                        (self.iterX_X_train_list[it - 1], new_X_sampled))
                    Y_train = np.hstack(
                        (self.iterX_Y_train_list[it - 1], new_Y_sampled))
                # model initialization
                # X_train_it.append(X_train[it])
                # Y_train_it.append(Y_train[it])
                self.iterX_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                          criterion='gini',
                                                          class_weight='balanced_subsample'
                                                          )
                self.iterX_model.fit(X_train, Y_train)
                predict = bsoid_predict_numba_noscale([self.features_heldout], self.iterX_model)
                predict = np.hstack(predict)

                self.iterX_X_train_list.append(X_train)
                self.iterX_Y_train_list.append(Y_train)
                self.iterX_f1_scores_list.append(f1_score(
                    self.targets_heldout[self.targets_heldout != self.label_code_other],
                    predict[self.targets_heldout != self.label_code_other],
                    average=None))
                self.iterX_macro_scores_list.append(f1_score(
                    self.targets_heldout[self.targets_heldout != self.label_code_other],
                    predict[self.targets_heldout != self.label_code_other],
                    average='macro'))
                self.iterX_predict_prob_list.append(self.iterX_model.predict_proba(
                    self.features_train[self.targets_train != self.label_code_other]))
                self.sampled_idx_list.append(sampled_idx)
                len_low_conf = len(np.where(self.iterX_predict_prob_list[-1].max(1) < self.conf_threshold)[0])
                if np.min(len_low_conf) > 0:
                    self.show_training_performance(it + 1)



                # iterX_f1_scores[it] = f1_score(
                #     self.targets_heldout[self.targets_heldout != self.label_code_other],
                #     predict[self.targets_heldout != self.label_code_other],
                #     average=None)
                # self.iterX_f1_scores_list.append(f1_score(
                #     self.targets_heldout[self.targets_heldout != self.label_code_other],
                #     predict[self.targets_heldout != self.label_code_other],
                #     average=None))
                # iterX_macro_scores[it] = f1_score(
                #     self.targets_heldout[self.targets_heldout != self.label_code_other],
                #     predict[self.targets_heldout != self.label_code_other],
                #     average='macro')
                # # iterX_macro_scores_it.append(iterX_macro_scores[it])
                # iterX_predict_prob[it] = self.iterX_model.predict_proba(
                #     self.features_train[self.targets_train != self.label_code_other]
                # )

                # iterX_predict_prob_it.append(iterX_predict_prob[it])

                # len_low_conf = [len(np.where(iterX_predict_prob[ii].max(1) < self.conf_threshold)[0])
                #                 for ii in range(len(iterX_predict_prob))]
                # if len(iterX_f1_scores) != 0 and np.min(len_low_conf) > 0:
                #     # self.iterX_X_train_list.append(X_train[it])
                #     # self.iterX_Y_train_list.append(Y_train[it])
                #     # self.iterX_f1_scores_list.append(iterX_f1_scores)
                #     # self.iterX_macro_scores_list.append(iterX_macro_scores_it)
                #     # self.iterX_predict_prob_list.append(iterX_predict_prob_it)
                #     # self.sampled_idx_list.append(sampled_idx)
                #     self.show_training_performance(it + 1)

                else:
                    st.success('The model did the best it could, no more confusing samples. Saving your progress...')
                    self.save_final_model_info()
                    break
                if it == self.max_iter - 1:
                    st.success("All iterations have been refined. Saving your progress...")
                    # save the data on last time
                    self.save_final_model_info()


    def show_training_performance(self, it):

        all_c_options = list(mcolors.CSS4_COLORS.keys())
        if len(self.annotation_classes) == 4:
            default_colors = ["red", "darkorange", "dodgerblue", "gray"]
        else:
            np.random.seed(42)
            selected_idx = np.random.choice(np.arange(len(all_c_options)), len(self.annotation_classes), replace=False)
            default_colors = [all_c_options[s] for s in selected_idx]

        mean_scores = np.hstack([100 * round(np.mean(self.iter0_f1_scores), 2),
                                 np.hstack([100 * round(np.mean(self.iterX_f1_scores_list[j], axis=0), 2)
                                            for j in range(len(self.iterX_f1_scores_list))])])
        mean_scores2beat = np.mean(self.all_f1_scores, axis=0)
        for c, c_name in enumerate(self.annotation_classes):
            if c_name != self.annotation_classes[-1]:
                self.perf_by_class[c_name].append(int(100 * round(self.iterX_f1_scores_list[-1][c], 2)))

        fig = make_subplots(rows=1, cols=1)
        fig.add_scatter(y=np.repeat(100 * round(mean_scores2beat, 2), self.max_iter + 1),
                        mode='lines',
                        marker=dict(color='white', opacity=0.1),
                        name='average (full data)',
                        row=1, col=1
                        )
        for c, c_name in enumerate(self.annotation_classes):
            if c_name != self.annotation_classes[-1]:
                fig.add_scatter(y=self.perf_by_class[c_name], mode='lines+markers',
                                marker=dict(color=default_colors[c]), name=c_name,
                                row=1, col=1
                                )
                fig.add_scatter(y=np.repeat(100 * round(self.all_f1_scores[c], 2),
                                            self.max_iter + 1), mode='lines',
                                marker=dict(color=default_colors[c]), name=str.join('', (c_name, ' (full data)')),
                                row=1, col=1
                                )
        fig.add_scatter(y=mean_scores, mode='lines+markers',
                        marker=dict(color='gray', opacity=0.8),
                        name='average',
                        row=1, col=1
                        )

        fig.update_xaxes(range=[-.5, self.max_iter + .5],
                         linecolor='dimgray', gridcolor='dimgray')
        fig.update_yaxes(ticksuffix="%", linecolor='dimgray', gridcolor='dimgray')
        fig.for_each_trace(
            lambda trace: trace.update(line=dict(width=2, dash="dot"))
            if trace.name.endswith('(full data)')
            else (trace.update(line=dict(width=2))),
        )
        # fig.update_traces(line=dict(width=2))
        fig.update_layout(
            title="",
            xaxis_title=self.keys[2],
            yaxis_title=self.keys[1],
            legend_title=self.keys[0],
            font=dict(
                family="Arial",
                size=14,
                color="white"
            )
        )
        self.placeholder.plotly_chart(fig, use_container_width=True)

    def save_subsampled_info(self):
        save_data(self.project_dir, self.iter_dir, 'iter0.sav',
                  [
                      self.iter0_Y_train,
                      self.iter0_f1_scores,
                  ])

    def save_all_train_info(self):
        save_data(self.project_dir, self.iter_dir, 'all_train.sav',
                  [
                      self.all_f1_scores,
                  ])

    def save_final_model_info(self):
        save_data(self.project_dir, self.iter_dir, 'iterX.sav',
                  [
                      self.iterX_model,
                      self.iterX_Y_train_list,
                      self.iterX_f1_scores_list,
                  ])

    def main(self):
        self.base_classification()
        self.self_learn()
        col_left, _, col_right = st.columns([1, 1, 1])
        col_right.success("Continue on with next module".upper())
