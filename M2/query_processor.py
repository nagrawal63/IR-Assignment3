from InvertedIndex import InvertedIndex, Postings
import time

def process_query(query):
    pass

if __name__ == "__main__":
    while True:
        query = input()
        start_time = time.time()
        process_query(query)
        end_time = time.time()
        print("Processing query({}) took {} seconds".format(query, str(end_time - start_time)))