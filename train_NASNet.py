##################################################################
# NASNest has good results on the Kaggle leaderboard, so give
# This uses the model with pretrained ImageNet weights
# This is training round 1/2

import os
from datetime import datetime
from keras import Input, layers, Model, callbacks
from keras.applications.nasnet import NASNetMobile
from keras.optimizers import Adam

from data_generator import train_gen, valid_gen
from make_plots import make_plots


def main():
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = 'pretrain_NASNet_' + now + '.h5'
    batch_size = 32
    num_epochs = 30
    lr = .0001

    num_train_samples = len(os.listdir('./data/train/cancer')) + len(os.listdir('./data/train/healthy'))
    num_valid_samples = len(os.listdir('./data/validation/cancer')) + len(os.listdir('./data/validation/healthy'))

    # Build our model
    input_tensor = Input(shape=(96, 96, 3))
    NASNet = NASNetMobile(include_top=False, input_shape=(96, 96, 3))
    x = NASNet(input_tensor)
    x1 = layers.GlobalMaxPooling2D()(x)
    x2 = layers.GlobalAveragePooling2D()(x)
    x3 = layers.Flatten()(x)
    z = layers.Concatenate(axis=-1)([x1, x2, x3])
    z = layers.Dropout(.5)(z)
    output_tensor = layers.Dense(1, activation='sigmoid')(z)

    model = Model(input_tensor, output_tensor)
    model.summary()

    # Get things ready to train: tweak learning rate, etc.
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['acc'])

    train_generator = train_gen(batch_size)
    validation_generator = valid_gen(batch_size)

    steps_per_epoch = num_train_samples / batch_size
    validation_steps = num_valid_samples / batch_size

    # Basic callbacks
    checkpoint = callbacks.ModelCheckpoint(filepath='./models/' + model_name,
                                           monitor='val_loss',
                                           save_best_only=True)
    early_stop = callbacks.EarlyStopping(monitor='val_acc',
                                         patience=4)
    csv_logger = callbacks.CSVLogger('./logs/' + model_name.split('.')[0] + '.csv')

    callback_list = [checkpoint, early_stop, csv_logger]

    # Training begins
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  verbose=1,
                                  callbacks=callback_list,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps)

    model.save('./models/' + model_name)

    make_plots(history, model_name)


if __name__ == "__main__":
    main()
