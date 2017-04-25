# Import modules
import math
import numpy as np
import time

import reducer
import fileRW

# Input File paths
input_test_data = '../input/test_data.csv'
input_training_data = '../input/training_data.csv'
input_training_labels = '../input/training_labels.csv'

# Output File paths
output_dimensional_reduction_test_data = '../output/DR_test_data.csv'
output_dimensional_reduction_training_data = '../output/DR_training_data.csv'
output_final_result = '../output/predicted_labels.csv'

# Program start time
beginning_of_time = time.time()
program_start_time = time.time()


# program run time counter
def method_timer(label, round_to_decimals=10000):
    global program_start_time
    program_end_time = time.time() - program_start_time
    program_end_time = math.ceil(
        program_end_time * round_to_decimals) / round_to_decimals
    program_start_time = time.time()
    print(label, 'took', program_end_time, 'seconds')

def total_runtime(round_to_decimals=10000):
    program_end_time = time.time() - beginning_of_time
    program_end_time = math.ceil(
        program_end_time * round_to_decimals) / round_to_decimals
    print('Total computation took', program_end_time, 'seconds')

# returns file contents as array
def get_array(file_path):
    return np.array(fileRW.csv_reader(file_path))

# we need to remove the appid column as the dimensional reduction
# is to be performed on the tf-idf values and not the appid


def remove_appid_colums(arr):
    return np.delete(arr, np.s_[0:1], axis=1)


# Step 1 : Read training_data.csv
# Step 1.1 : Perform dimensional reduction
# Step 1.2 : Save the reduced training_data.csv

# prepare the dimensionally reduced training_data.csv
training_data_arr = get_array(input_training_data)
method_timer('Getting array from training_data.csv')
training_data_arr = remove_appid_colums(training_data_arr)
method_timer('Removing App Id from training_data.csv')
dr_training_data_arr = reducer.dimensional_reduction(training_data_arr)
method_timer('Reducing training_data.csv')
fileRW.csv_writer(dr_training_data_arr,
                  output_dimensional_reduction_training_data)
method_timer('Writing reduced training_data.csv to disk')

# Step 2 : Read test_data.csv
# Step 2.1 : Perform dimensional reduction
# Step 2.2 : Save the reduced test_data.csv

# prepare the dimensionally reduced test_data.csv
test_data_arr = get_array(input_test_data)
method_timer('Getting array from test_data.csv')
test_data_arr = remove_appid_colums(test_data_arr)
method_timer('Removing App Id from test_data.csv')
dr_test_data_arr = reducer.dimensional_reduction(test_data_arr)
method_timer('Reducing test_data.csv')
fileRW.csv_writer(dr_test_data_arr, output_dimensional_reduction_test_data)
method_timer('Writing reduced test_data.csv to disk')

# Step 3 :  Run classifier to get a function from DR'd training_data.csv and training_labels.csv
# Step 3.1 : Run the classifier on DR'd test_data.csv to get
# predicted_labels.csv
total_runtime()