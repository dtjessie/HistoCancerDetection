##################################################################
# This script loads one of the .h5 models and then builds a
# submission file

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def main():
    model_name = 'KaiYu_ZhuTu_v1.h5'
    prediction_thresh = .6
    batch_size = 128
    
    test_dir = './data/test/'
    num_test_samples = len(os.listdir('./data/test/test_images/'))
    steps = num_test_samples / batch_size
        
    model = load_model(model_name)
    model.summary()
        
    test_datagen = ImageDataGenerator(rescale = 1.0/255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size = (96, 96),
                                                      batch_size = batch_size,
                                                      class_mode = 'binary',
                                                      shuffle = False)
    output = model.predict_generator(test_generator,
                                     steps = steps + 1,
                                     verbose = 1)
    
    pred = np.floor_divide(output, prediction_thresh)
    
    test_ids = np.sort([k.split('.')[0] for k in os.listdir('./data/test/test_images/')])
    lines = [test_ids[i] + ',' + str(int(pred[i][0])) for i in range(len(test_ids))]
    
    f = open('submit_' + model_name.split('.')[0] + '.csv', 'w')
    f.write('id,label\n')
    for l in lines:
        f.write(l+"\n")
    f.close()
    
if __name__ == "__main__":
    main()