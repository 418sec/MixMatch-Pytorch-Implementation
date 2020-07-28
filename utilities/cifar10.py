import os
import csv
import torch
import pandas as pd
import numpy as np

import utilities.data as data

class cifar10():
    def __init__(self, id:str, root:str, val_ratio:float, unlabel_ratio:float):
        self.id = id
        self.root = root
        self.val_ratio = val_ratio
        self.unlabel_ratio = unlabel_ratio
    
    def csv(self, dataset, object, label):
        path = os.path.join(self.root, dataset, object)
        filenames = os.listdir(path)
        df = pd.DataFrame(filenames, columns =['file_name']) 
        df['path']=os.path.join(self.root, dataset, object) + os.sep +df['file_name']
        df['unlabeled'] = False
        df['object'] = object
        df['label'] = label

        if dataset == 'train':
            if self.val_ratio: 
                train_idx, val_idx = data.splitSampler(df, 'object', self.val_ratio)
                train_df = df.loc[train_idx].reset_index(drop=True)
                val_df = df.loc[val_idx].reset_index(drop=True)
            else:
                train_df = df

            if self.unlabel_ratio:
                unlabeled_idx = data.splitSampler(train_df, 'object', self.unlabel_ratio)[1]
                train_df.loc[unlabeled_idx, 'unlabeled'] = True
                train_df.loc[unlabeled_idx, 'label'] = 99

            return train_df, val_df

        elif dataset == 'test':
            return df
    
    def master(self):
        trains = []
        vals = []
        train_path = os.path.join(self.root, 'train')
        train_objects = os.listdir(train_path)
        for object in train_objects:
            label = train_objects.index(object)
            train, val = self.csv('train', object, label)
            trains.append(train)
            vals.append(val)

        tests = []
        test_path = os.path.join(self.root, 'test')
        test_objects = os.listdir(test_path)
        for object in test_objects:
            label = test_objects.index(object)
            test = self.csv('test', object, label)
            tests.append(test)

        train_df = pd.concat(trains)
        val_df = pd.concat(vals)
        test_df = pd.concat(tests)

        train_df.to_csv('results/{}/master/train.csv'.format(self.id), index=False, encoding='utf-8-sig')
        val_df.to_csv('results/{}/master/val.csv'.format(self.id), index=False, encoding='utf-8-sig')
        test_df.to_csv('results/{}/master/test.csv'.format(self.id), index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    cifar10('dummy_run', 'cifar10',  0.1, 0.5).master()