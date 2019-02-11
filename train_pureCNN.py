##################################################################
# This builds a simple CNN with for classifying. No dense layers.
# No tweaking or optimization has been done; validation loss
# is unstable...

import os
from keras.preprocessing.image import ImageDataGenerator
from keras import Input, layers, Model, callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = 'pureCNN_' + now + '.h5'
    batch_size = 256
    num_epochs = 30
    
    train_dir = './data/train/'
    validation_dir = './data/validation/'
    
    num_train_samples = len(os.listdir('./data/train/cancer')) + len(os.listdir('./data/train/healthy'))
    num_valid_samples = len(os.listdir('./data/validation/cancer')) + len(os.listdir('./data/validation/healthy'))
    
    
    # Build our cool model
    input_tensor = Input(shape = (96,96,3))
    x = layers.Conv2D(32, (2,2), activation = 'relu', padding = 'same')(input_tensor)
    x = layers.Conv2D(32, (2,2), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (2,2), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(64, (2,2), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(.5)(x)
    output_tensor = layers.Dense(1, activation = 'sigmoid')(x)

    model = Model(input_tensor, output_tensor)
    model.summary()
    
    # Get things ready to train: should adjust learning rate, etc.
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
    train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                       rotation_range = 90,
                                       horizontal_flip = True)
    valid_datagen = ImageDataGenerator(rescale = 1.0/255)
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size = (96, 96),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')

    validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                             target_size = (96, 96),
                                                             batch_size = batch_size,
                                                             class_mode = 'binary')

    steps_per_epoch = num_train_samples / batch_size
    validation_steps = num_valid_samples / batch_size
    
    # Basic callbacks
    checkpoint = callbacks.ModelCheckpoint(filepath = './models/' + model_name,
                                           monitor = 'val_loss',
                                           save_best_only = True)
    early_stop = callbacks.EarlyStopping(monitor = 'val_acc',
                                         patience = 3)
    callback_list = [checkpoint, early_stop]
    
    # Training begins
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs = num_epochs,
                                  verbose = 1,
                                  callbacks = callback_list,
                                  validation_data = validation_generator,
                                  validation_steps = validation_steps)

    model.save('./models/' + model_name)
    
    # Some visualizations
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) +1)

    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()