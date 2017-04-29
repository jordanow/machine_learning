# Import modules
import math
import numpy as np

# Import local modules
import logger

# Every row in training_data.csv has a corresponding row in training_labels.csv
# So it can be assumed that the RHS of both the CSVs is similar.
# After dimensionally reducing training_data.csv, we can say that the dimensionally
# reduced DR_training_data.csv contains numbers which have an equivalent
# label in training_labels.csv

# So, the sequence of numbers in each row DR_training_data.csv gives
# a label for the same row in training_labels.csv.

# We need a Classifier which can identify this  f(sequence of
# numbers)->label for each row.

# Classifier will take 2 arguments
# 1. The numbers (from DR_training_data.csv) and
# 2. The labels (from training_labels.csv)
# The output of the Classifier is a map of function from sequence of
# numbers to labels


def createMap(seq_numbers, labels):


