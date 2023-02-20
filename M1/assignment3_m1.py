import time
import os

directory_name = "DEV"

def get_file_names():
    pass

def process_data(file_names):
    pass

if __name__ == "__main__":
    start_time = time.time()
    file_names = get_file_names()
    print("number of files: " + str(len(file_names)))
    process_data(file_names)
    end_time = time.time()
    print("Total execution time: " + str(end_time - start_time))