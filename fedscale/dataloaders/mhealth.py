from __future__ import print_function

import csv
import logging
import warnings
import os
import torch

class MHEALTH():
    def __init__(self, root, dataset="train"):

        self.data_file = dataset
        self.root = root
        self.path = os.path.join(self.processed_folder, self.data_file)

        # load data and targets
        self.data, self.targets = self.load_file(self.path)

    def __getitem__(self, index):
        """
        Args: 
            index (int): Index
        Returns:
            tuples: (features, target) where target is index of the target class.
        """

        feature, target = self.data[index], int(self.targets[index])

        return feature, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root
    
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.data_file)))

    def load_meta_data(self, path):
        datas, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    # print(row[1:-1])
                    features = [float(x) for x in row[1:-1]] 
                    datas.append(torch.FloatTensor(features))
                    labels.append(int(row[-1]))
                line_count += 1

        return datas, labels

    def load_file(self, path):

        datas, labels = self.load_meta_data(os.path.join(
            self.processed_folder, 'client_data_mapping', self.data_file + '.csv'))

        return datas, labels
