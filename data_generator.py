# This file standardizes the data augmentation used by the
# various train_XXX.py files.

from keras.preprocessing.image import ImageDataGenerator

global train_dir
global validation_dir
global test_dir

train_dir = './data/train/'
validation_dir = './data/validation/'
test_dir = './data/test/'


def train_gen(batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=90,
                                       width_shift_range=.1,
                                       height_shift_range=.1,
                                       shear_range=.05,
                                       zoom_range=.2,
                                       channel_shift_range=.1,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(96, 96),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    return train_generator


def valid_gen(batch_size):
    valid_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                             target_size=(96, 96),
                                                             batch_size=batch_size,
                                                             class_mode='binary')
    return validation_generator


def test_gen(batch_size):
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(96, 96),
                                                      batch_size=batch_size,
                                                      class_mode='binary',
                                                      shuffle=False)
    return test_generator
