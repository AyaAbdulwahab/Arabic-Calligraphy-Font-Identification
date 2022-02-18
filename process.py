import numpy as np
from preprocessing import Preprocessing
from features_extraction import Features_Extraction
class Process:

    @staticmethod
    def data_processing(X_train):
        features_arr = np.array([])
        for i in range(X_train.shape[0]):
            img_g = Preprocessing.gray_scale_img(X_train[i])
            point_feature = Features_Extraction.lpq(img_g)
            if i ==0:
                features_arr = np.array(point_feature)

            else:
                features_arr= np.vstack((features_arr, point_feature))

        return features_arr

    @staticmethod
    def point_processing(X_point):
        img_g = Preprocessing.gray_scale_img(X_point)
        point_feature = Features_Extraction.lpq(img_g)
        features_arr = np.array([point_feature])
        return features_arr
