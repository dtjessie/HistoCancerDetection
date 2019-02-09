##################################################################
# This builds a simple CNN with 4 convolution layers followed by
# a dense 512 layer.
# No tweaking or optimization has been done; overfits after about
# 10 epochs or so.

from keras.preprocessing.image import ImageDataGenerator
from keras import Input, layers, Model
import matplotlib.pyplot as plt
import os

def main():
    model_name = 'simpleCNN.h5' # should add a time stamp to avoid overwrites
    batch_size = 256
    num_epochs = 15
    
    train_dir = './data/train/'
    validation_dir = './data/validation/'
    
    num_train_samples = len(os.listdir('./data/train/cancer')) + len(os.listdir('./data/train/healthy'))
    num_valid_samples = len(os.listdir('./data/validation/cancer')) + len(os.listdir('./data/validation/healthy'))
    
    
    # Build our cool model
    input_tensor = Input(shape = (96,96,3))
    x = layers.Conv2D(32, (3,3), activation = 'relu')(input_tensor)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation = 'relu')(x)
    output_tensor = layers.Dense(1, activation = 'sigmoid')(x)

    model = Model(input_tensor, output_tensor)
    model.summary()
    
    # Get things ready to train
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
    train_datagen = ImageDataGenerator(rescale = 1.0/255)
    valid_datagen = ImageDataGenerator(rescale = 1.0/255)
    

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
    
    # Training begins
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs = num_epochs,
                                  verbose = 1,
                                  validation_data = validation_generator,
                                  validation_steps = validation_steps)

    model.save(model_name)
    
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