import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize
import cv2
import skimage

import Preprocess

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(1,30):
            if each_letter >= 'A' and each_letter <= 'Z':
                image_num = str(11 + ord(each_letter) - ord('A'))
            elif each_letter == '9':
                image_num = '10'
            else:
                image_num = '0'+str(int(each_letter) - int('0') + 1)
            if each <= 9:
                image_num2 = '0'+str(each)
            else:
                image_num2 = str(each)
            image_path = os.path.join(training_directory,'Sample'+each_letter,'img0'+image_num +'-000'+ image_num2 + '.png')
            # read each image of each character
            img_load = imread(image_path, as_grey=True)
            img_details = skimage.color.rgb2gray(img_load)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            binary_resized = resize(binary_image,(20,20))
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_resized.reshape(-1)
            
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training

    accuracy_result = cross_val_score(model, train_data, train_label,cv=num_of_fold)

    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


current_dir = os.path.dirname(os.path.realpath(__file__))

training_dataset_dir = os.path.join(current_dir, 'EnglishImg/English/Bmp')

image_data, target_data = read_training_data(training_dataset_dir)

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svc_model = SVC(kernel='rbf', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don't need to train the model again
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory+'/svc.pkl')