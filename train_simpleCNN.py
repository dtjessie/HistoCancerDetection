##################################################################
# This builds a simple CNN with 4 convolution layers followed by
# a dense 512 layer.
# No tweaking or optimization has been done; overfits after about
# 10 epochs or so.

import os
from keras.preprocessing.image import ImageDataGenerator
from keras import Input, layers, Model, callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

from data_generator import train_gen, valid_gen
from make_plots import make_plots

def main():
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = 'simpleCNN_' + now + '.h5'
    batch_size = 256
    num_epochs = 30
    lr = .001
    
    num_train_samples = len(os.listdir('./data/train/cancer')) + len(os.listdir('./data/train/healthy'))
    num_valid_samples = len(os.listdir('./data/validation/cancer')) + len(os.listdir('./data/validation/healthy'))
    
    
    # Build our cool model
    input_tensor = Input(shape = (96,96,3))
    x = layers.Conv2D(32, (3,3))(input_tensor)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3))(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3))(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3))(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(.5)(x)
    x = layers.Dense(512, activation = 'relu')(x)
    output_tensor = layers.Dense(1, activation = 'sigmoid')(x)

    model = Model(input_tensor, output_tensor)
    model.summary()
    
    # Get things ready to train: should adjust learning rate, etc.
    model.compile(optimizer = Adam(lr), loss = 'binary_crossentropy', metrics = ['acc'])
    
    train_generator = train_gen(batch_size)
    validation_generator = valid_gen(batch_size)

    steps_per_epoch = num_train_samples / batch_size
    validation_steps = num_valid_samples / batch_size
    
    # Basic callbacks
    checkpoint = callbacks.ModelCheckpoint(filepath = './models/' + model_name,
                                           monitor = 'val_loss',
                                           save_best_only = True)
    early_stop = callbacks.EarlyStopping(monitor = 'val_acc',
                                         patience = 3)
    csv_logger = callbacks.CSVLogger('./logs/' + model_name.split('.')[0] + '.csv')
    
    callback_list = [checkpoint, early_stop, csv_logger]
    
    # Training begins
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs = num_epochs,
                                  verbose = 1,
                                  callbacks = callback_list,
                                  validation_data = validation_generator,
                                  validation_steps = validation_steps)

    model.save('./models/' + model_name)
    
    make_plots(history, model_name)

if __name__ == "__main__":
    main()