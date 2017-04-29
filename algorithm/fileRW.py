# import modules
import csv
import pandas as pd

# csv delimiter to be used
csv_delimiter = ','


# csv reader fn
def csv_reader(file_path):
    return pd.read_csv(
        file_path, delimiter=csv_delimiter, header=None).values


# write csv to disk
def csv_writer(contents, file_name):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=csv_delimiter,
                            quoting=csv.QUOTE_ALL)
        for content in contents:
            writer.writerow(content)
