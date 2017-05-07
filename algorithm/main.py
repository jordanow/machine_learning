# Import modules
import numpy as np

# Import local modules
import classifier
import fileRW
import logger
import reducer

# Input File paths
input_test_data = '../input/test_data.csv'
input_training_data = '../input/training_data.csv'
input_training_labels = '../input/training_labels.csv'

# Output File paths
output_dimensional_reduction_test_data = '../output/DR_test_data.csv'
output_dimensional_reduction_training_data = '../output/DR_training_data.csv'
output_final_result = '../output/predicted_labels.csv'

# global variables
training_set_size = 90


# Returns file contents as array
def get_array(file_path):
    return np.array(fileRW.csv_reader(file_path))


# We need to remove the appid column as the dimensional reduction
# is to be performed on the tf-idf values and not the appid
def remove_first_column(arr):
    return np.delete(arr, np.s_[0:1], axis=1)


# Returns the selected column
def get_arr_column(arr, column_number):
    return arr[:, [column_number]]


# Merges two given arrays along a given axis
def merge_arrs(arr1, arr2, axis=1):
    return np.concatenate((arr1, arr2), axis=axis)


"""
Randomise the array
Then split it into 2 parts

arrX and arrY represent input and output array respectively
"""


def get_cross_validation_input_output_pairs(arrX, arrY, percent_range):
    np.random.shuffle(arrX)
    length_of_arrX = len(arrX)
    length_of_arr_train = int((percent_range * length_of_arrX) / 100)

    arrX1 = []
    arrX2 = []
    arrY1 = []
    arrY2 = []

    i = 0
    while i < length_of_arr_train:
        arrX1.append(arrX[i])
        arrY1.append(arrY[i])
        i = i + 1

    i = length_of_arr_train
    while i < length_of_arrX:
        arrX2.append(arrX[i])
        arrY2.append(arrY[i])
        i = i + 1

    return np.array(arrX1), np.array(arrX2), np.array(arrY1), np.array(arrY2)


print('Starting classification task for Assignment 1')

"""
Step 1 :
Read training_data.csv
Perform dimensional reduction
Save the reduced training_data.csv
"""
# prepare the dimensionally reduced training_data.csv
# print("Reading training_data.csv")
# training_data_arr = get_array(input_training_data)
# logger.method_timer('Getting array from training_data.csv')

# training_data_arr_app_names = get_arr_column(training_data_arr, 0)
# logger.method_timer('Getting App Id from training_data.csv')

# training_data_arr_values = remove_first_column(training_data_arr)
# logger.method_timer('Getting tf-idf values from training_data.csv')

# dr_training_data_arr = reducer.dimensional_reduction(training_data_arr_values)
# logger.method_timer('Reducing training_data.csv')

# test_data_arr_reduced = merge_arrs(
#     training_data_arr_app_names, dr_training_data_arr)
# logger.method_timer(
#     'Merging training app names and their dimensionally reduced values')

# fileRW.csv_writer(test_data_arr_reduced,
#                   output_dimensional_reduction_training_data)
# logger.method_timer('Writing reduced training_data.csv to disk')
# The process of dimensionally reducing training_data.csv alone takes
# around 550 seconds or 10 minutes


"""
Step 2 :
Read test_data.csv
Perform dimensional reduction
Save the reduced test_data.csv
"""

# prepare the dimensionally reduced test_data.csv
# print("Reading test_data.csv")
# test_data_arr = get_array(input_test_data)
# logger.method_timer('Getting array from test_data.csv')

# test_data_arr_app_names = get_arr_column(test_data_arr, 0)
# logger.method_timer('Getting App Id from test_data.csv')

# test_data_arr_values = remove_first_column(test_data_arr)
# logger.method_timer('Getting tf-idf values from test_data.csv')

# dr_test_data_arr = reducer.dimensional_reduction(test_data_arr_values)
# logger.method_timer('Reducing test_data.csv')

# test_data_arr_reduced = merge_arrs(test_data_arr_app_names, dr_test_data_arr)
# logger.method_timer(
#     'Merging test app names and their dimensionally reduced values')

# fileRW.csv_writer(test_data_arr_reduced,
#                   output_dimensional_reduction_test_data)
# logger.method_timer('Writing reduced test_data.csv to disk')
# The process of dimensionally reducing test_data.csv alone takes around
# 35 seconds


"""
Step 3
Read DR_training_data.csv
Split the training data into training set and test set in a ratio
of 90:10, where 90% belongs to training set
"""
print('Starting cross validation')
dr_training_data_arr = get_array(
    output_dimensional_reduction_training_data)
logger.method_timer('Getting array from DR_training_data.csv')

dr_training_data_arr_no_app_id = remove_first_column(dr_training_data_arr)
logger.method_timer('Getting the tf-idf values for training')

training_labels_arr = get_array(input_training_labels)
training_labels_arr_no_app_id = remove_first_column(training_labels_arr)

X_Train, X_Test, Y_Train, Y_Test = get_cross_validation_input_output_pairs(
    dr_training_data_arr_no_app_id, training_labels_arr_no_app_id, training_set_size)
logger.method_timer(
    'Splitting training data into test and training set')

# print(X_Test.shape, X_Train.shape);
# print(Y_Test.shape, Y_Train.shape);

# Step 3 :  Run classifier to get a function from DR'd training_data.csv and training_labels.csv
# Step 3.1 : Run the classifier on DR'd test_data.csv to get
# predicted_labels.csv

# Step 3 : Read DR_training_data.csv and training_labels.csv and send the
# data to classifier
# dr_training_data_arr = get_array(
#     output_dimensional_reduction_training_data)
# logger.method_timer('Getting array from DR_training_data.csv')
# training_labels_arr = get_array(input_training_labels)
# logger.method_timer('Getting array from training_labels.csv')
# classifier_dict = classifier.create_classifier(
#     dr_training_data_arr, training_labels_arr)


# Step 3.1 Use the classifier result and pass the DR_test_data.csv
# and match the data to get the app ids for the data in DR_test_data.csv
# dr_test_data_arr = get_array(output_dimensional_reduction_test_data)
# logger.method_timer('Getting array from DR_test_data.csv')
# classifier_final_result = classifier.classify(
#     classifier_dict, dr_test_data_arr)
# print(classifier_final_result['Social'])
logger.total_runtime()
