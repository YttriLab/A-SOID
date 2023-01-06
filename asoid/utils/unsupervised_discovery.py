import numpy as np

from psutil import virtual_memory
import streamlit as st
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import hdbscan

from utils.extract_features import Extract
from utils.project_utils import create_new_project, update_config
from utils.load_workspace import load_features,save_data,load_data, load_class_embeddings,load_class_clusters, load_full_feats_targets
from config.help_messages import CLASS_SELECT_HELP,CLUSTER_RANGE_HELP,START_DISCOVERY_HELP, \
    SUBCLASS_SELECT_HELP,SAVE_NEW_HELP, PREPARE_DATA_HELP
from config.global_config import HDBSCAN_PARAMS,UMAP_PARAMS



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

    def export_to_new_project(self, new_targets):
        # create new project folder with prefix as name:
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

    def save_subclasses(self, selected_subclasses):

        idx_class = np.argwhere(self.target_set == self.selected_class_num)
        org_num_classes = len(self.classes)
        class_name = self.number_to_class[self.selected_class_num]
        self.new_annotations = self.target_set.copy()

        for num, new_class in enumerate(selected_subclasses):

            idx_sub_class = np.argwhere(self.hdbscan_assignments == new_class)
            idx_candidates = idx_class[idx_sub_class]
            # create new class
            new_class_num = org_num_classes + num
            self.new_annotations[idx_candidates] = new_class_num
            self.new_number_to_class[new_class_num] = 'Sub-{} group {}'.format(class_name,new_class)

        #Upsample targets to original framerate, then create new project
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
        self.export_to_new_project(split_targets)

    def run_discovery(self):


        self.cluster_range = st.slider('Select a cluster range',
                                       0.1,10.0,(1.5,2.5),
                                       help=CLUSTER_RANGE_HELP)

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

            self.show_results()

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

        with st.form(key="new_class_submit_form"):
            selected_subclasses = st.multiselect(
                'Select Groups to add to training set'
                ,valid_subclasses
                ,help=SUBCLASS_SELECT_HELP)


            if st.form_submit_button("Save new classes",help=SAVE_NEW_HELP) and selected_subclasses:
                # translate to numbers:
                selected_subclasses_num = [valid_subclasses_to_number[x] for x in selected_subclasses]
                self.save_subclasses(selected_subclasses_num)


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
                        if st.checkbox("Redo discovery"):
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
