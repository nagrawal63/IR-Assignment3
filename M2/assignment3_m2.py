from assignment3_m1 import tokenize_content
from InvertedIndexLoader import loadInvertedIndexFromFile , getIndexDataAllTokens
import time
import json
import numpy as np
from enum import IntEnum
from numpy import dot
from numpy.linalg import norm
# from sklearn.metrics.pairwise import cosine_similarity

def process_query(query):
    query_tokens = tokenize_content(query)
    print(f"{query_tokens}")
    return query_tokens

class ImportanceEnum(IntEnum):
    TITLE = 1
    H1 = 2
    H2 = 3
    H3 = 4
    B = 5
    NORMAL = 6
    IMPORTANT = 7

def merge_inverted_index(datal,doc2features):
    importance_map = {ImportanceEnum.NORMAL: 1, ImportanceEnum.B: 2, ImportanceEnum.H3: 3,
                           ImportanceEnum.H2: 4, ImportanceEnum.H1: 5, ImportanceEnum.TITLE: 6}
    final_page = {}
    queryv =[1] *len(datal)
    
    for i,td in enumerate(datal):
        #queryv[i] = 1/ len(td)
        for d in td:
            if d.docId in final_page:
                final_page[d.docId][i] = d.tfidf
            else:
                final_page[d.docId] = [0] * (len(datal) +1 )  #[len(td)TODO]+1 page features # initialize with zero vector
                final_page[d.docId][i] = d.tfidf
    queryv.extend([0.01])
    for d in final_page:
        final_page[d][-1] = doc2features[str(d)]['pagein']/len(doc2features)

    return sorted({k:dot(queryv,v) for k,v in final_page.items() if 0 not in v}.items(),key=lambda x :x[1],reverse=True) # [TODO] & option adding ? 

def retrieve_pages(tokens,doc2features):
    datal = getIndexDataAllTokens(tokens)
    pages = merge_inverted_index(datal,doc2features)
    return pages[:10]


if __name__ == "__main__":
    while True:
        with open("./url_docID_map.json") as f:
            doc2id = json.load(f)
        with open("./docID_url_map.json") as f:
            id2doc = json.load(f)
        with open('../M1/page_quality_features.json') as f:
            doc2features = json.load(f)
        print("Enter Query:")
        query = input()
        start_time = time.time()
        tokens = process_query(query)
        pages = retrieve_pages(tokens,doc2features)
        for p in pages:
            print(p,p[1],doc2features[str(p[0])],id2doc[str(p[0])]) 
        end_time = time.time()
        print("Processing query({}) took {} seconds".format(query,end_time-start_time))
