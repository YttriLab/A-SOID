import numpy as np
import pandas as pd
from datetime import date
from psutil import virtual_memory
import streamlit as st
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import umap
import hdbscan

from utils.extract_features import Extract
from utils.project_utils import create_new_project, update_config
from utils.load_workspace import load_features,save_data,load_data, load_class_embeddings,load_class_clusters, load_full_feats_targets
from config.help_messages import CLASS_SELECT_HELP,CLUSTER_RANGE_HELP,START_DISCOVERY_HELP, \
    SUBCLASS_SELECT_HELP,SAVE_NEW_HELP, PREPARE_DATA_HELP, PREFIX_HELP
from config.global_config import HDBSCAN_PARAMS,UMAP_PARAMS


def reset_checkbox(check_box_key:str):
    if check_box_key in st.session_state:
        st.session_state[check_box_key] = False


def plot_embedding(plt_reducer_embeddings,class_name):
    trace1 = go.Scatter(x=plt_reducer_embeddings[:,0],
                        y=plt_reducer_embeddings[:,1],
                        name=class_name,
                        mode='markers'
                        )

    fig = make_subplots()
    fig.add_trace(trace1)

    fig.update_xaxes(title_text="UMAP Dim 1",row=1,col=1,showticklabels=False)
    fig.update_yaxes(title_text="UMAP Dim 2",row=1,col=1,showticklabels=False)
    fig.update_layout(title_text="Unsupervised Embedding",
                      )

    return fig


def plot_hdbscan_embedding(plt_assignments,plt_reducer_embeddings):
    # some plotting parameters
    NOISE_COLOR = 'lightgray'
    unique_classes = np.unique(plt_assignments)
    group_types = ['Noise']
    group_types.extend(['Group{}'.format(i) for i in unique_classes if i >= 0])

    trace_list = []

    for num,g in enumerate(unique_classes):

        if g < 0:
            idx = np.where(plt_assignments == g)[0]
            trace_list.append(go.Scatter(x=plt_reducer_embeddings[idx,0],
                                         y=plt_reducer_embeddings[idx,1],
                                         name="Noise",
                                         mode='markers'
                                         )
                              )
        else:
            idx = np.where(plt_assignments == g)[0]

            trace_list.append(go.Scatter(x=plt_reducer_embeddings[idx,0],
                                         y=plt_reducer_embeddings[idx,1],
                                         name=group_types[num],
                                         mode='markers'

                                         ))

    fig = make_subplots()
    for trace in trace_list:
        fig.add_trace(trace)

    fig.update_xaxes(title_text="UMAP Dim 1",row=1,col=1,showticklabels=False)
    fig.update_yaxes(title_text="UMAP Dim 2",row=1,col=1,showticklabels=False)
    fig.update_layout(title_text="Unsupervised Clustering",
                      )

    return fig,group_types


def hdbscan_classification(umap_embeddings,cluster_range):
    max_num_clusters = -np.infty
    num_clusters = []
    min_cluster_size = np.linspace(cluster_range[0],cluster_range[1],5)
    for min_c in min_cluster_size:
        learned_hierarchy = hdbscan.HDBSCAN(
            prediction_data=True,min_cluster_size=int(round(min_c * 0.01 * umap_embeddings.shape[0])),
            **HDBSCAN_PARAMS).fit(umap_embeddings)
        num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
        if num_clusters[-1] > max_num_clusters:
            max_num_clusters = num_clusters[-1]
            retained_hierarchy = learned_hierarchy
    assignments = retained_hierarchy.labels_
    assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
    soft_assignments = np.argmax(assign_prob,axis=1)
    return retained_hierarchy,assignments,assign_prob,soft_assignments

def expand_assignments(assignments, features, model=None):
    """ Train a classifier on the given features and assignments to assign noise to clusters. Then use the classifier to
    assign the noise to clusters.
    :param features: the features to train on
    :param assignments: the assignments to train on from hdbscan
    :param model: the model to use for training (optional). If not provided, a default random forest classifier is used.
    :return: the expanded assignments
    """
    # todo: have config file for standard model parameters shared across all modules
    if model is None:
        noise_assignment_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                        criterion='gini',
                                                        class_weight='balanced_subsample'
                                                        )
    else:
        noise_assignment_model = model

    # split features into noise and clusters
    noise_idx = np.where(assignments == -1)[0]
    cluster_idx = np.where(assignments >= 0)[0]
    # features
    noise_features = features[noise_idx]
    cluster_features = features[cluster_idx]
    # assignments
    cluster_assignments = assignments[cluster_idx]

    # train on clusters
    noise_assignment_model.fit(cluster_features, cluster_assignments)
    # predict on noise
    noise_predictions = noise_assignment_model.predict(noise_features)
    # combine predictions with clusters but keep positions of noise
    expanded_assignments = np.zeros(features.shape[0])
    expanded_assignments[noise_idx] = noise_predictions
    expanded_assignments[cluster_idx] = cluster_assignments

    return expanded_assignments, noise_assignment_model


class Explorer:

    def __init__(self,config):
        self.config = config
        self.random_state = 42
        self.selected_class_num = None
        self.reducer_embeddings = None
        self.pca = PCA(random_state=self.random_state)
        self.scaled_feature_set_just_class = None
        # hdbscan
        self.cluster_range = [1.5,2.5]
        self.hdbscan_assignments = None

        self.ran_embedding = False
        self.ran_clustering = False

        self.working_dir = config["Project"].get("PROJECT_PATH")
        self.prefix = config["Project"].get("PROJECT_NAME")
        self.annotation_classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.framerate = config["Project"].getint("FRAMERATE")
        self.duration_min = config["Processing"].getfloat("MIN_DURATION")
        self.frames2integ = round(float(self.framerate) * (self.duration_min / 0.1))
        # [self.features,self.targets,self.scalar,self.frames2integ] = load_features(self.working_dir,self.prefix)
        # self.feature_set = self.features[0]
        # self.target_set = self.targets[0]
        self.feature_set = None
        self.target_set = None
        # # get classes for later
        self.classes = [x.strip() for x in config["Project"].get("CLASSES").split(",")]
        self.class_to_number = {s: i for i,s in enumerate(self.classes)}
        self.number_to_class = {i: s for i,s in enumerate(self.classes)}
        # prep for later reassignment
        self.new_annotations = None
        self.new_number_to_class = self.number_to_class.copy()
        #prep assignment_model
        self.noise_assignment_model = None
        self.expanded_assignments = None

        self.feature_extractor = Extract(self.working_dir, self.prefix, self.frames2integ, 1)
        self.data, _ = load_data(self.working_dir, self.prefix)
        # get relevant data from data file
        [self.processed_input_data,
         self.targets] = self.data

    def run_pca(self):
        self.pca.fit(self.feature_set)

    def select_class(self):

        selected_class = st.selectbox(
            'Select Class to use for directed discovery'
            ,self.classes
            ,help=CLASS_SELECT_HELP)

        self.selected_class_num = self.class_to_number[selected_class]

        return selected_class

    def scale_selected_class(self):
        features_train_just_class = self.feature_set[self.target_set == self.selected_class_num,:]

        scaler = StandardScaler()
        scaler.fit(features_train_just_class)
        self.scaled_feature_set_just_class = scaler.transform(features_train_just_class)

    def umap_embedd(self,low_memory):
        num_dimensions = np.argwhere(np.cumsum(self.pca.explained_variance_ratio_) >= 0.7)[0][0] + 1
        self.reducer_embeddings = umap.UMAP(n_components=num_dimensions if num_dimensions >1 else 2,
                                            low_memory=low_memory,
                                            **UMAP_PARAMS).fit_transform(self.scaled_feature_set_just_class)

        save_data(self.working_dir,self.prefix,f'{self.number_to_class[self.selected_class_num]}_embeddings.sav',
                  [self.scaled_feature_set_just_class,
                   self.reducer_embeddings])

    def show_embedding(self):

        fig = plot_embedding(self.reducer_embeddings,class_name=self.number_to_class[self.selected_class_num])
        st.plotly_chart(fig)

    def hdbscan_clustering(self):

        retained_hierarchy,self.hdbscan_assignments,assign_prob,soft_assignments = hdbscan_classification(
            self.reducer_embeddings,
            self.cluster_range)
        save_data(self.working_dir,self.prefix,f'{self.number_to_class[self.selected_class_num]}_clusters.sav',
                  [retained_hierarchy,
                   self.hdbscan_assignments,
                   assign_prob,
                   soft_assignments])

    def show_clustering(self):

        fig, group_types = plot_hdbscan_embedding(self.hdbscan_assignments,self.reducer_embeddings)
        st.plotly_chart(fig)
        return group_types

    def export_to_csv(self, new_targets):
        """Export the new expanded assignments to individual csv files for external use"""
        #get input label file names
        label_file_names = [x.strip() for x in self.config["Data"].get("LABEL_INPUT_FILES").split(",")]
        #get the new class names
        new_class_list = [x for x in list(self.new_number_to_class.values()) if x != "other"]
        #and that other is last!
        new_class_list.append("other")
        new_num_to_class = {n: x for n,x in enumerate(new_class_list)}
        #iter through the expanded assignments (already split to correct size) and save as csv#
        for num, j in enumerate(new_targets):
            df = pd.Series(j)
            for n_class, k_class in new_num_to_class.items():
                df[df == n_class] = k_class

            #TODO: add time column?
            # add empty columns even if class not present!

            # transform into binary
            df_dummies = pd.get_dummies(df)
            #make sure all classes are present (even if not present in this file)
            for k_class in new_class_list:
                if k_class not in df_dummies.columns:
                    df_dummies[k_class] = 0
            #add time column based on frame rate
            df_dummies["time"] = df_dummies.index / self.framerate
            #make time column new index
            df_dummies = df_dummies.set_index("time")
            # save as csv with new filename
            fname = label_file_names[num].split(".")[0] + "_discovered.csv"
            #create a new folder for this
            os.makedirs(os.path.join(self.working_dir, self.prefix, "export"), exist_ok=True)
            #save as csv
            df_dummies.to_csv(os.path.join(self.working_dir, self.prefix, "export", fname))



    def export_to_new_project(self, new_targets, new_prefix = None):
        # create new project folder with prefix as name:
        if new_prefix is None:
            new_prefix = self.prefix + f"_{self.number_to_class[self.selected_class_num]}_disc"
        project_folder,_ = create_new_project(self.working_dir,new_prefix,overwrite=True)
        with open(os.path.join(project_folder,'data.sav'),'wb') as f:
            """Save data as npy file"""
            # data
            joblib.dump(
                [np.array(self.processed_input_data), np.array(new_targets)]
                ,f
            )
        #update empty config with previous project
        update_config(project_folder,updated_params=self.config)
        #make sure to add the new classes
        new_class_list = [x for x in list(self.new_number_to_class.values()) if x != "other"]
        #and that other is last!
        new_class_list.append("other")
        # update config with new parameters:
        parameters_dict = {
            "Project": dict(PROJECT_NAME = new_prefix,
                CLASSES= new_class_list
            )
        }
        update_config(project_folder, updated_params=parameters_dict)
        st.success(f"Upload newly created project {new_prefix} to continue training with the new classes".upper())

    def save_subclasses(self, selected_subclasses, new_prefix, export_to_csv = False):
        """Save the selected subclasses to a new project
        :param selected_subclasses: list of selected subclasses
        :param new_prefix: prefix for new project
        :param export_to_csv: if True, export to csv in current project dir, default False
        :return:"""

        idx_class = np.argwhere(self.target_set == self.selected_class_num)
        #original number of classes, ignoring other
        num_other = len(self.classes)
        org_num_classes = num_other -1


        class_name = self.number_to_class[self.selected_class_num]
        self.new_annotations = self.target_set.copy()

        #train a classifier to assign noise to clusters
        with st.spinner("Training classifier to assign noise to clusters"):
            self.expanded_assignments, self.noise_assignment_model = expand_assignments(self.hdbscan_assignments
                                                                                        ,self.scaled_feature_set_just_class)

        #add new classes to new annotations and number to class
        for num, new_class in enumerate(selected_subclasses):

            idx_sub_class = np.argwhere(self.expanded_assignments == new_class)
            idx_candidates = idx_class[idx_sub_class]
            # create new class
            new_class_num = org_num_classes + num
            self.new_annotations[idx_candidates] = new_class_num
            self.new_number_to_class[new_class_num] = 'Sub-{} group {}'.format(class_name,new_class)


        #now that all new classes have been added, reset other to the last position
        idx_other = np.argwhere(self.target_set == num_other)
        new_num_other = new_class_num + 1
        self.new_annotations[idx_other] = new_num_other
        self.new_number_to_class[new_num_other] = "other"

        #TODO: Add frameshift prediction?
        # Upsample targets to original framerate, then create new project
        # upsample labels to fit with pose estimation info

        sample_rate = 1 / self.framerate
        time_step = self.duration_min
        upsample_rate = time_step / sample_rate

        new_targets = np.repeat(self.new_annotations, int(upsample_rate),axis=0)
        #turn into int to avoid indexing issues later
        new_targets = new_targets.astype(int)
        #reshape/split into targets shape
        # List of lengths of the sub-arrays (original files)
        sub_array_lengths = [sub_array.shape[0] for sub_array in self.targets]
        # cumsum to get relative position of each file, we can ignore the last
        sub_array_idx = np.cumsum(sub_array_lengths[:-1])
        # Split the array using `split`
        split_targets = np.split(new_targets, sub_array_idx)
        if export_to_csv:
            self.export_to_csv(split_targets)
        else:
            pass
        self.export_to_new_project(split_targets, new_prefix)

    def run_discovery(self):
        print("Running discovery")

        self.cluster_range = st.slider('Select a cluster range',
                                       0.1,10.0,(1.5,2.5),
                                       help=CLUSTER_RANGE_HELP,
                                       key='cluster_range')

        if st.button("Start discovery",help=START_DISCOVERY_HELP):

            self.run_pca()
            with st.spinner('Rescaling features...'):
                self.scale_selected_class()

            mem = virtual_memory()
            available_mb = mem.available >> 20
            st.info('You have {} MB RAM 🐏 available'.format(available_mb))
            if available_mb > (
                    self.feature_set.shape[0] * self.feature_set.shape[1] * 32 * 60) / 1024 ** 2 + 64:
                st.info('RAM 🐏 available is sufficient')
                low_memory = False

            else:
                st.info(
                    'Detecting that you are running low on available memory for this computation, '
                    'setting low_memory so will take longer.')
                low_memory = True

            with st.spinner('Embedding features...'):
                self.umap_embedd(low_memory)

            with st.spinner('Clustering embedding...'):
                self.hdbscan_clustering()

            st.success("Done. Uncheck 'Redo discovery' to view results and continue")

    def load_results(self):
        [self.scaled_feature_set_just_class,
         self.reducer_embeddings] = load_class_embeddings(self.working_dir,self.prefix,
                                                          self.number_to_class[self.selected_class_num])

        [_,
         self.hdbscan_assignments,
         _,
         _] = load_class_clusters(self.working_dir,self.prefix,self.number_to_class[self.selected_class_num])


    def show_results(self):
        #self.show_embedding()
        optional_subclasses = self.show_clustering()

        #remove noise as option:
        valid_subclasses = [x for x in optional_subclasses if x != "Noise"]
        valid_subclasses_to_number = {s: i for i,s in enumerate(valid_subclasses)}

        st.subheader("Export discovered classes to new project")
        st.write("A new project will be created with the new classes added. You can then use this project to train a new classifier.")

        with st.form(key="new_class_submit_form"):
            selected_subclasses = st.multiselect(
                'Select Groups to add to training set'
                ,valid_subclasses
                ,help=SUBCLASS_SELECT_HELP)
            today = date.today()
            d4 = today.strftime("%b-%d-%Y")
            new_prefix = st.text_input(
                "Please enter a new prefix to create a project containing the new annotations"
                ,d4 + "_disc"
                ,help=PREFIX_HELP
                ,key = "prefix_select_disc"
                )
            EXPORT_HELP = "Export the cluster assignments to a csv file for further analysis and external use."
            csv_export_checkbox = st.checkbox("Export to csv", help=EXPORT_HELP)

            if st.form_submit_button("Save new classes",help=SAVE_NEW_HELP) and selected_subclasses:
                # translate to numbers:
                selected_subclasses_num = [valid_subclasses_to_number[x] for x in selected_subclasses]

                self.save_subclasses(selected_subclasses_num, new_prefix, export_to_csv=csv_export_checkbox)


    def main(self):
        param_cont = st.container()
        cont = st.container()
        if self.working_dir is not None:


            #load features and targets from all data
            try:
                [self.feature_set,
                 self.target_set,
                 self.feature_extractor.scalar,
                 self.frames2integ] = load_full_feats_targets(self.working_dir,self.prefix)

                with param_cont:
                    selected_class_str = self.select_class()
                try:
                    self.load_results()
                    with cont:
                        #create checkbox to redo discovery
                        if st.checkbox("Redo discovery", key="redo_discovery_key"):
                            # clear cache so it reloads from file rather than cache
                            load_class_clusters.clear()
                            load_class_embeddings.clear()
                            self.run_discovery()

                        else:
                            self.show_results()

                except FileNotFoundError:
                    with cont:
                        self.run_discovery()


            except FileNotFoundError as e:

                if st.button("Prepare data for discovery", key= "all_feat_extract_key", help = PREPARE_DATA_HELP):
                    with st.spinner("Extracting features from all data (This will only run once!)..."):
                        self.feature_extractor.extract_features()
                        self.feature_extractor.downsample_labels()

                        self.feature_set = self.feature_extractor.features
                        self.target_set = self.feature_extractor.targets_mode

                        save_data(self.working_dir,self.prefix,'full_feats_targets.sav',
                                  [self.feature_set,
                                   self.target_set,
                                   self.feature_extractor.scalar,
                                   self.frames2integ])
                    st.experimental_rerun()



        else:
            st.info("Upload config file first.")
