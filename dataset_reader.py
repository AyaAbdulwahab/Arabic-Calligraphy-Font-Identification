import os
import cv2
import numpy as np
import re

class DatasetReader:

    @staticmethod
    def get_images_from_directory(directory):

        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        fnames = sorted(os.listdir(directory),key= alphanum_key)
        to_return = []
        for fn in fnames:
            path = os.path.join(directory, fn)
            gray_scale_image = (cv2.imread(path)).astype(np.uint8)
            to_return.append( gray_scale_image)
        return to_return


    @staticmethod
    def read_train_data():
            
        X_train=np.array([])
        Y_train=np.array([])
        X_test=np.array([])
        Y_test=np.array([])
        train_data = DatasetReader.get_images_from_directory(r'../Project Submission/test')
        X_test =  np.append(X_test, np.asarray(train_data[:]))



        for i in range(1,10):

            train_data = DatasetReader.get_images_from_directory(r'../ACdata_base/'+str(i))
            X_train =  np.append(X_train, np.asarray(train_data[:] ))
            Y_train = np.append(Y_train,np.ones(len(train_data))*i)

        
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def read_test_data(test_dir):

        X_test=np.array([], dtype = np.int8)
        train_data = DatasetReader.get_images_from_directory(test_dir)
        X_test =  np.append(X_test, np.asarray(train_data[:] ))
        
        return X_test
        
