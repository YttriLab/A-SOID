import os
import numpy as np
import pandas as pd
from scipy import stats



from asoid.utils.extract_features_2D import feature_extraction
from asoid.utils.extract_features_3D import feature_extraction_3d

from asoid.utils.load_workspace import load_data, save_data
from asoid.config.help_messages import *



class Extract:
    #TODO: restructure streamlit use

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