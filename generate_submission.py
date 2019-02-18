##################################################################
# This script loads one of the .h5 models and then builds a
# submission file

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from data_generator import test_gen

def main():
    model_name = 'pretrain_NASNet_2019-02-12_22:03:53.h5'
    batch_size = 128
    
    num_test_samples = len(os.listdir('./data/test/test_images/'))
    steps = num_test_samples / batch_size
        
    model = load_model('./models/' + model_name)
    model.summary()
        
    test_generator = test_gen(batch_size)
    #################################################################
    # Test-time augmentation flipping image gives about a 1% boost! #    
    #################################################################
    pred = []
    for i in range(steps + 1):
        x = test_generator.next()
        X = np.array(x[0])
        pred_0 = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*
                   model.predict(X[:, ::-1, ::-1,:]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
        pred += pred_0
   
    fnames = test_generator.filenames
    test_ids = [k.split('/')[1][:-4] for k in fnames]
    lines = [test_ids[i] + ',' + str(float(pred[i])) for i in range(len(test_ids))]
    
    f = open('./submissions/' + model_name.split('.')[0] + '.csv', 'w')
    f.write('id,label\n')
    for l in lines:
        f.write(l+"\n")
    f.close()
    print("Done! Submission saved in ./submissions/" + model_name.split('.')[0] + '.csv')
    
if __name__ == "__main__":
    main()