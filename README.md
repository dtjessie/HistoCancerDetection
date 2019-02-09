# Histopathologic Cancer Detection
Setting up our Kaggle team

This repo contains some python files for the competition https://www.kaggle.com/c/histopathologic-cancer-detection

Once the dataset is put into the original_dataset directory, you can run setup_dirs.py to organize the data that
is needed for the train.simpleCNN.py script

train_simpleCNN.py builds the simpleCNN.h5 model. A trained version is saved in ./models/

generate_submission.py uses the .h5 to create a submission for Kaggle

TO DO: add droput, data augmentation to simpleCNN model

       add callbacks, checkpointing to the training routine

       run a pre-trained model for comparison--NASNet has good results on leaderboard
