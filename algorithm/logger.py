# import modules
import math
import time

# Program start time
beginning_of_time = time.time()
program_start_time = time.time()


# Program run time counter
def method_timer(label, round_to_decimals=10000):
    global program_start_time
    program_end_time = time.time() - program_start_time
    program_end_time = math.ceil(
        program_end_time * round_to_decimals) / round_to_decimals
    program_start_time = time.time()
    print(label, 'took', program_end_time, 'seconds')


# Calculates the total runtime of the program
def total_runtime(round_to_decimals=10000):
    program_end_time = time.time() - beginning_of_time
    program_end_time = math.ceil(
        program_end_time * round_to_decimals) / round_to_decimals
    print('Total computation took', program_end_time, 'seconds')
