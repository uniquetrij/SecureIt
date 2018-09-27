import csv

def read_csv(file_name, ignore_header = False):
    output_list = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        if ignore_header:
            next(csvreader)
        for row in csvreader:
            output_list.append(row)
    return output_list