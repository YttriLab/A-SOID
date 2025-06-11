import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import f1_score
import matplotlib.colors as mcolors

from asoid.utils.predict import frameshift_predict, bsoid_predict_numba, bsoid_predict_numba_noscale
from asoid.utils.load_workspace import load_features, load_test, save_data

from stqdm import stqdm
from tqdm import tqdm



def get_confidence_calc(k):
    """ Returns function to calculate confidence based on the selected option.
    param k: str, selected option
    return: function to calculate confidence
    """
    confidence_lot = {
    'max': lambda predictions: np.max(predictions, axis=1),
    'max - min': lambda predictions: np.max(predictions, axis=1) - np.min(predictions, axis=1),
    'max - mean': lambda predictions: np.max(predictions, axis=1) - np.mean(predictions, axis=1),
    'max - median': lambda predictions: np.max(predictions, axis=1) - np.median(predictions, axis=1),
    'max - 25p': lambda predictions: np.max(predictions, axis=1) - np.percentile(predictions, 25, axis=1),
    'max - 75p': lambda predictions: np.max(predictions, axis=1) - np.percentile(predictions, 75, axis=1),
    }

    assert k in confidence_lot, f"Invalid option {k}. Available options are: {list(confidence_lot.keys())}"
    
    return confidence_lot[k]

def get_available_conf_options():
    """ Returns available confidence calculation options.
    return: list of available options
    """
    #TODO: Add comprehensive description
    #TODO: change default threshold to real default threshold
    confidence_lot = {
    'max': dict(description = 'Maximum probability', default_thresh = 0.5),
    'max - min': dict(description = 'Maximum - minimum', default_thresh = 0.5),
    'max - mean': dict(description = 'Maximum - mean', default_thresh = 0.5),
    'max - median': dict(description = 'Maximum - median', default_thresh = 0.5),
    'max - 25p': dict(description = 'Maximum - 25th percentile', default_thresh = 0.5),
    'max - 75p': dict(description = 'Maximum - 75th percentile', default_thresh = 0.5)
    }
    
    return confidence_lot


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
    scores = np.vstack((np.hstack(base_score), np.vstack(learn_score)))
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


class RF_Classify:

    def __init__(self, working_dir, prefix, iteration_dir, software
                 , init_ratio
                 , max_iter
                 , max_samples_iter
                 , annotation_classes
                 , exclude_other
                 , conf_type: str = 'max'
                 , conf_threshold: float = 0.5
                 , mode: str = 'app'):

        self.mode = mode
        if mode == 'app':
            self.container = st.container()
            self.placeholder = self.container.empty()
        else:
            self.container = None
            self.placeholder = None
        if self.mode == 'notebook':
            init_notebook_mode(connected=True)
        self.working_dir = working_dir
        self.prefix = prefix
        self.project_dir = os.path.join(working_dir, prefix)
        self.iter_dir = iteration_dir
        self.software = software
        self.conf_type = conf_type
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

    def with_spinner(self, message, func, *args, **kwargs):
        """
        Wrapper for st.spinner to handle different modes.
        :param message: Message to display in the spinner.
        :param func: Function to execute.
        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        :return: Result of the function execution.
        """
        if self.mode == 'app':
            with st.spinner(message):
                return func(*args, **kwargs)
        else:
            print(message)  # Print the message in notebook or other modes
            return func(*args, **kwargs)

    def show_success(self, message):
        """
        Show success message in the app mode.
        :param message: Message to display.
        """
        if self.mode == 'app':
            st.success(message)
        else:
            print(message)

    def init_cl(self):
        cl = RandomForestClassifier(n_estimators=200
                                    , random_state=42
                                    , n_jobs=-1
                                    , criterion='gini'
                                    , class_weight='balanced_subsample'
                                    )
        return cl

    def split_data(self):
        [X, y, self.frames2integ] = \
            load_features(self.project_dir, self.iter_dir)

        self.features_train, self.features_heldout, \
            self.targets_train, self.targets_heldout = train_test_split(X, y
                                                                        , test_size=0.20
                                                                        , random_state=42
                                                                        )


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
        self.all_model = self.init_cl()
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
        self.iter0_model = self.init_cl()
        
        self.iter0_model.fit(X_train, Y_train)
        predict = bsoid_predict_numba_noscale([self.features_heldout], self.iter0_model)
        predict = np.hstack(predict)

        flt_predict = predict[self.targets_heldout != self.label_code_other]
        curr_targets = self.targets_heldout[self.targets_heldout != self.label_code_other]

        # check f1 scores per class
        self.iter0_f1_scores = f1_score(
            curr_targets,
            flt_predict,
            average=None
            , labels=np.unique(curr_targets))

        # check f1 scores overall
        self.iter0_macro_scores = f1_score(
            curr_targets,
            flt_predict,
            average='macro'
            , labels=np.unique(curr_targets))

        self.iter0_predict_prob = self.iter0_model.predict_proba(
            self.features_train[self.targets_train != self.label_code_other])
        self.iter0_X_train = X_train
        self.iter0_Y_train = Y_train


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

        if self.mode == 'notebook':
            print("Baseline")
            print("Subsampled performance: ", self.iter0_f1_scores)
            print("Full data performance: ", self.all_f1_scores)
            
        elif self.mode == 'app':

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
        self.with_spinner("Subsampled classification...", self.subsampled_classify)
        self.with_spinner("Preparing plot...", self.show_subsampled_performance)
        self.with_spinner("Saving training data...", self.save_all_train_info)
        self.with_spinner("Saving subsampled data...", self.save_subsampled_info)
        
    def _train_iteration(self, it, compute_confidence):
        sampled_idx = []
        if it == 0:
            # Start with iter0 data
            X_train = self.iter0_X_train
            Y_train = self.iter0_Y_train
            idx_lowconf = np.where(compute_confidence(self.iter0_predict_prob) < self.conf_threshold)[0]
        else:
            idx_lowconf = np.where(compute_confidence(self.iterX_predict_prob_list[it - 1]) < self.conf_threshold)[0]

        # Identify low-confidence samples
        new_X_human = self.features_train[self.targets_train != self.label_code_other][idx_lowconf, :]
        new_Y_human = self.targets_train[self.targets_train != self.label_code_other][idx_lowconf]

        np.random.seed(42)
        try:
            idx_sampled = np.random.choice(np.arange(idx_lowconf.shape[0]),
                                        min(self.max_samples_iter, idx_lowconf.shape[0]),
                                        replace=False)
        except:
            return

        new_X_sampled = new_X_human[idx_sampled, :]
        new_Y_sampled = new_Y_human[idx_sampled]
        sampled_idx.append(idx_lowconf[idx_sampled])

        if it == 0:
            X_train = np.vstack((X_train, new_X_sampled))
            Y_train = np.hstack((Y_train, new_Y_sampled))
        else:
            X_train = np.vstack((self.iterX_X_train_list[it - 1], new_X_sampled))
            Y_train = np.hstack((self.iterX_Y_train_list[it - 1], new_Y_sampled))

        # Train the model
        self.iterX_model = self.init_cl()
        self.iterX_model.fit(X_train, Y_train)
        predict = bsoid_predict_numba_noscale([self.features_heldout], self.iterX_model)
        predict = np.hstack(predict)

        self.iterX_X_train_list.append(X_train)
        self.iterX_Y_train_list.append(Y_train)

        # Evaluate performance
        flt_predict = predict[self.targets_heldout != self.label_code_other]
        curr_targets = self.targets_heldout[self.targets_heldout != self.label_code_other]

        curr_f1_scores = f1_score(curr_targets, flt_predict, average=None, labels=np.unique(curr_targets))
        curr_macro_scores = f1_score(curr_targets, flt_predict, average='macro', labels=np.unique(curr_targets))

        self.iterX_f1_scores_list.append(curr_f1_scores)
        self.iterX_macro_scores_list.append(curr_macro_scores)
        self.iterX_predict_prob_list.append(self.iterX_model.predict_proba(
            self.features_train[self.targets_train != self.label_code_other]))
        self.sampled_idx_list.append(sampled_idx)

    def self_learn(self):
        compute_confidence = get_confidence_calc(self.conf_type)

        for it in range(self.max_iter):
            self.with_spinner(f'Training iteration {it + 1}...', self._train_iteration, it, compute_confidence)

            len_low_conf = len(np.where(compute_confidence(self.iterX_predict_prob_list[-1]) < self.conf_threshold)[0])
        
            if np.min(len_low_conf) > 0:
                self.show_training_performance(it + 1)
            else:
                self.show_success('The model did the best it could, no more confusing samples. Saving your progress...')
                self.save_final_model_info()
                break
            if it == self.max_iter - 1:
                self.show_success("All iterations have been refined. Saving your progress...")
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

        if self.mode == 'notebook':
            print("Iteration ", it)
            print("Subsampled performance: ", self.iterX_f1_scores_list[-1])
        
        if self.mode == "app":

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
