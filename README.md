# Histopathologic Cancer Detection
Setting up our Kaggle team

This repo contains some python files for the competition https://www.kaggle.com/c/histopathologic-cancer-detection

Helper files are data_generator, generate_submission, make_plots, setup_dirs.

Main files and model definitions are in the train_\*, pretrain_\* files.

Once the dataset is put into the original_dataset directory, you can run setup_dirs.py to organize the data that
is needed for the train.simpleCNN.py script

train_simpleCNN.py builds the simpleCNN.h5 model. A trained version is saved in ./models/

generate_submission.py uses the .h5 to create a submission for Kaggle

CSV logs and plots are saved in ./logs and ./models, respectively.

VGG16 and NASNet are fine-tuned using files in respective folders

Current best (.9673 on leaderboard) uses ensemble of VGG16 and NASNet.

To Do: Try a couple more architectures and ensemble to reach .98 on public leaderboard
