# Import modules
import math
import numpy as np

# Import local modules
import logger
import reducer

# Every row in training_data.csv has a corresponding row in training_labels.csv
# So it can be assumed that the RHS of both the CSVs is similar.
# After dimensionally reducing training_data.csv, we can say that the dimensionally
# reduced DR_training_data.csv contains numbers which have an equivalent
# label in training_labels.csv

# So, the sequence of numbers in each row DR_training_data.csv gives
# a label for the same row in training_labels.csv.

# We need a Classifier which can identify this  f(sequence of
# numbers)->label for each row. Since there are more sequence of numbers than the labels,
# it is obvious that many sequences lead to the same label


# Create a dictionary
# Since we're creating a dictionary for arrays
# we can skip the check if the array has already been included in the dictionary
# This is because we're going to reduce the arrays later
def createDict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


# Classifier will take 2 arguments
# 1. The numbers (from DR_training_data.csv) and
# 2. The labels (from training_labels.csv)
# The output of the Classifier is a map of function from sequence of
# numbers to labels
def create_classifier(seq_numbers, labels):
    label_to_seq_dict = {}
    index = 0
    # create a dictionary of labels and the corresponding sequence of numbers
    for app_key, label in labels:
        createDict(label_to_seq_dict, label, seq_numbers[index])
        index = index + 1
    logger.method_timer(
        'Creating a dictionary of label to array of its corresponding sequences')

    # Reduce the multi-dimensional values of dictionary's key to single
    # dimension
    # Transpose the label_to_seq_dict and then reduce it to one column
    # Transpose it again to get label -> one seq of 200 numbers mapping
    label_to_single_val_dict = {}
    for key in label_to_seq_dict.keys():
        arr = np.array(label_to_seq_dict[key])
        arr = arr.T
        dr_seq_numbers = reducer.dimensional_reduction(arr, 1)
        dr_seq_numbers = dr_seq_numbers.T
        createDict(label_to_single_val_dict, key, dr_seq_numbers[0])
    logger.method_timer(
        'Creating a dictionary of label to one dimensional number sequence')
    return label_to_single_val_dict;


def classify(classifier_arr_dict, unknown_arr):
  for key in classifier_arr_dict.keys():
    arr = np.array(classifier_arr_dict[key])
    arr = arr.T
    print(arr.shape)
  print(unknown_arr.shape)
  return []