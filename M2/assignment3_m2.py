from assignment3_m1 import tokenize_content
from InvertedIndexLoader import loadInvertedIndexFromFile
import time
import json

def process_query(query):
    query_tokens = tokenize_content(query)
    print(f"{query_tokens}")
    return query_tokens

def merge_inverted_index(datal):
    final_page = {}
    for td in datal:
        for d in td:
            if d.docId in final_page:
                final_page[d.docId] += d.count
            else:
                final_page[d.docId] = d.count
    
    return sorted(final_page.items(),key=lambda x :x[1],reverse=True)

def retrieve_pages(tokens):
    tokens = sorted(tokens) # sorted for reducing loading time for same character
    prevc= None
    datal = []
    for t in tokens:
        currentc = t[0]
        if currentc != prevc:
            # data = loadInvertedIndexFromFile(f'./M2/splitted_index/{currentc}.json')
            with open(f'./M2/splitted_index/{currentc}.json') as f:
                l = f.readlines()
                skippointer  = json.loads(l[-1])
            from InvertedIndex import Postings
            data = [Postings.from_json(d) for d in json.loads(l[skippointer[t]])[t]]       
            datal.append(data)
            prevc = currentc
    pages = merge_inverted_index(datal)
    return pages[:10]


if __name__ == "__main__":
    while True:
        with open("./M1/url_docID_map.json") as f:
            doc2id = json.load(f)
        id2doc = {v:k for k,v in doc2id.items()}

        print("Enter Query:")
        query = input()
        start_time = time.time()
        tokens = process_query(query)
        pages = retrieve_pages(tokens)
        for p in pages:
            print(id2doc[p[0]-1]) # need to change 
        end_time = time.time()
        print("Processing query({}) took {} seconds".format(query,end_time-start_time))
