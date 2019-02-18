##################################################################
# This script splits the train_images into a train and validation
# set and sets up the directories needed to use the keras
# flow_from_directory data generator.
# To run this, we need the following inside the original_data dir
#
#           original_data/
#               |
#                --> test_images/
#               |
#                --> train_images/ 
#               |
#                --> train_labels.csv
#
# test_images/ and train_images/ contain the .tif image files
# The new data/ directory contains symlinks to the .tif files

import os
import pandas as pd
import numpy as np

VALID_FRAC = .1  # What proportion of train to use as validation?
RANDOM_SEED = 8  # For replicability

def train_valid_dict(df, frac_valid = .1):
    # This splits the dataset in df into train/validation subsets
    n = len(df)
    np.random.seed(seed = RANDOM_SEED)
    random = np.random.rand(n)
    train_id, valid_id = [], []
    for i in range(n):
        if random[i] > frac_valid:
            train_id.append(df['id'][i])
        else:
            valid_id.append(df['id'][i])
    partition = {'train': train_id, 'validation': valid_id}
    return partition

def main():
    # Make subdirectories
    original_data_dir = '/home/dtj/ml/kaggle/histopathologic_cancer/original_data'
    base_dir = '/home/dtj/ml/kaggle/histopathologic_cancer/data'

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    test_images_dir = os.path.join(test_dir, 'test_images')
    
    train_cancer_dir = os.path.join(train_dir, 'cancer')
    train_healthy_dir = os.path.join(train_dir, 'healthy')
    validation_cancer_dir = os.path.join(validation_dir, 'cancer')
    validation_healthy_dir = os.path.join(validation_dir, 'healthy')
    
    try: 
        os.mkdir(base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        
        os.mkdir(train_cancer_dir)
        os.mkdir(train_healthy_dir)
        os.mkdir(validation_cancer_dir)
        os.mkdir(validation_healthy_dir)
        os.mkdir(test_images_dir)
    except OSError as e:
        print("OSError:", e)
        return 0
    
     
    # Split into train and validation sets
    # dictionaries labels and partition
    # will be used to make symlinks
    df = pd.read_csv(os.path.join(original_data_dir, 'train_labels.csv'))
    labels = dict(zip([k for k in df['id']], [v for v in df['label']]))
    partition = train_valid_dict(df, VALID_FRAC)
    
    # Make symlinks    
    for ID in partition['train']:
        fname = ID + '.tif'
        src = os.path.join(original_data_dir, 'train_images', fname)
        if labels[ID] == 0:
            dst = os.path.join(train_healthy_dir, fname)
        else:
            dst = os.path.join(train_cancer_dir, fname)
        os.symlink(src, dst)
    for ID in partition['validation']:
        fname = ID + '.tif'
        src = os.path.join(original_data_dir, 'train_images', fname)
        if labels[ID] == 0:
            dst = os.path.join(validation_healthy_dir, fname)
        else:
            dst = os.path.join(validation_cancer_dir, fname)
        os.symlink(src, dst)
    fnames = [k for k in os.listdir(os.path.join(original_data_dir, 'test_images'))]
    for fname in fnames:
        src = os.path.join(original_data_dir, 'test_images', fname)
        dst = os.path.join(test_images_dir, fname)
        os.symlink(src, dst)
    
    num_train_cancer = len(os.listdir(train_cancer_dir))
    num_train_healthy = len(os.listdir(train_healthy_dir))
    num_valid_cancer = len(os.listdir(validation_cancer_dir))
    num_valid_healthy = len(os.listdir(validation_healthy_dir))
    print('total training cancer images:', num_train_cancer)
    print('total training healthy images:', num_train_healthy)
    print('total validation cancer images:', num_valid_cancer)
    print('total validation healthy images:', num_valid_healthy)
    print('total test images:', len(os.listdir(test_images_dir)))

    
if __name__ == "__main__":
    main()