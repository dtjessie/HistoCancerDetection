##################################################################
# This builds a simple CNN with 4 convolution layers followed by
# a dense 512 layer.
# No tweaking or optimization has been done; overfits after about
# 10 epochs or so.

import os
from keras.preprocessing.image import ImageDataGenerator
from keras import Input, layers, Model, callbacks
from keras.applications.nasnet import NASNetMobile
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = 'pretrain_NASNet_' + now + '.h5'
    batch_size = 128
    num_epochs = 30
    
    train_dir = './data/train/'
    validation_dir = './data/validation/'
    
    num_train_samples = len(os.listdir('./data/train/cancer')) + len(os.listdir('./data/train/healthy'))
    num_valid_samples = len(os.listdir('./data/validation/cancer')) + len(os.listdir('./data/validation/healthy'))
       
    # Build our cool model
    input_tensor = Input(shape = (96, 96, 3))
    NASNet = NASNetMobile(include_top = False, input_shape = (96, 96, 3))
    x = NASNet(input_tensor)
    x1 = layers.GlobalMaxPooling2D()(x)
    x2 = layers.GlobalAveragePooling2D()(x)
    x3 = layers.Flatten()(x)
    z = layers.Concatenate(axis=-1)([x1, x2, x3])
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(512, activation = 'relu')(z)
    output_tensor = layers.Dense(1, activation="sigmoid")(z)
    
    model = Model(input_tensor, output_tensor)
    model.summary()
    
    # Get things ready to train: tweak learning rate, etc.
    model.compile(optimizer = Adam(.00001), loss = 'binary_crossentropy', metrics = ['acc'])
    train_datagen = ImageDataGenerator(samplewise_center = True,
                                       samplewise_std_normalization = True,
                                       rotation_range = 20,
                                       horizontal_flip = True,
                                       vertical_flip = True,
                                       shear_range = 10)
    
    valid_datagen = ImageDataGenerator(samplewise_center = True,
                                       samplewise_std_normalization = True)
    
        
    # In the future, we should do some data augmentation
    # especially noticing that the cancer is in the center of the image
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
                                         patience = 5)
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