from preprocessing import Preprocessing
from features_extraction import Features_Extraction
from dataset_reader import DatasetReader
from process import Process
from classifiers import Classifiers
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pickle
import time
import sys

train_model = False
cross_validation_applied = True

if train_model:
    X_train, X_test, Y_train, Y_test = DatasetReader.read_train_data()
    if cross_validation_applied:
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        acc = 0
        index = 0
        for train_index, test_index in kf.split(X_train):
            x_train1, x_test1 = X_train[train_index], X_train[test_index]
            y_train1, y_test1 = Y_train[train_index], Y_train[test_index]
            features_arr = Process.data_processing(x_train1)
            features_test_arr = Process.data_processing(x_test1)
            majority_vote_classification = Classifiers.MajorityVote(features_arr, y_train1, features_test_arr, False)
            _ , svm_classification, _ = Classifiers.SVM_Classifier(features_arr, y_train1, features_test_arr)
            index += 1
            acc += np.sum([majority_vote_classification == y_test1])/len(y_test1)
            print("Majority Vote",np.sum([majority_vote_classification == y_test1])/len(y_test1))
            print("SVM Vote",np.sum([svm_classification == y_test1])/len(y_test1))
        acc /= index
        print("Overall accuracy: ",acc)
    else:
        features_arr = Process.data_processing(X_train)
        majority_vote_classification = Classifiers.MajorityVote(features_arr, Y_train, features_arr)

else:
    sys_var = sys.argv
    _, test_dir, out_dir = sys_var
    X_test = DatasetReader.read_test_data(test_dir)
    results = np.zeros(len(X_test))
    timer = np.zeros(len(X_test))
    filename = 'maj_vote_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    for i, point in enumerate(X_test):
        
        start_time = time.time()
        Point_features = Process.point_processing(point)
        result = loaded_model.predict(Point_features)
    
        point_timer = np.round(time.time() - start_time,2)

        timer[i] = point_timer
        results[i] = result

    with open(out_dir + "results.txt", 'w') as res:
        for ind,i in enumerate(results):
            res.write(str(int(i)))
            if ind is not len(results)-1:
                res.write("\n")

    with open(out_dir + 'times.txt', 'w') as t:
        for ind, i in enumerate(timer):
            if i == 0:
                t.write("0.001")
            else:    
                t.write(str(i))
            if ind is not len(timer)-1:
                t.write("\n")

# print("KNN",np.sum([knn_classification == Y_test])/len(Y_test))
# print("Out KNN", knn_classification)
# print("MAjority Vote",np.sum([majority_vote_classification == Y_test])/len(Y_test))
# print("Majority Out", majority_vote_classification)
# print("SVM",np.sum([svm_classification == Y_test])/len(Y_test))
# print("SVM Out", svm_classification)
# print("RandForest",np.sum([rand_forest_classification == Y_test])/len(Y_test))
# print("Rand Forrest Out", rand_forest_classification)
# print("MLP",np.sum([mlp_classification == Y_test])/len(Y_test))
# print("MLP Out", mlp_classification)

# f1_score_res = f1_score(Y_test, majority_vote_classification, average='macro')
# print(f1_score_res)

