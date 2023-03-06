from InvertedIndexLoader import getIndexEntry
import time
from assignment3_m1 import tokenize_content
import numpy as np
import json, math
from collections import defaultdict
TOTAL_DOCS = 55393

def loadDocID_to_URL_map():
    docDict = {}
    with open("docID_url_map.json",'r') as f:
        docDict = json.load(f)
    return docDict

def process_query(query):
    tokenized_query = tokenize_content(query)
    query_freq = {}
    for token in tokenized_query:
        if token in query_freq.keys():
            query_freq[token] += 1
        else:
            query_freq[token] = 1

    query_freq = dict(sorted(query_freq.items()))
    print("sorted query freq", query_freq)

    common_docs = [];query_docs_len = [];query_docs = [];docs = [];query_tfidf_dict_list = []

    start_time = time.time()
    for token in query_freq.keys():
        token_index = getIndexEntry(token)
        if token_index is not None:
            docs = list(map(lambda x: x.docId, token_index[token]))
            query_tfidf_map = {posting.docId: posting.tfidf for posting in token_index[token]}
            common_docs += docs
            query_docs.append(set(docs))
            query_docs_len.append(len(docs))
            query_tfidf_dict_list.append(query_tfidf_map)
        else:
            del query_freq[token]
    end_time = time.time()
    print(" query_indexes Processing query({}) took {} seconds".format(query, str(end_time - start_time)))

    if len(common_docs) == 0:
        print("sorry, nothing matches, please check your query")
        return None

    common_docs = list(set(common_docs))

    start_time = time.time()
    query_tfidf_array = calculate_query_tf_idf_score(query_freq,query_docs_len)
    end_time = time.time()
    print(" query_tfidf_array Processing query({}) took {} seconds".format(query, str(end_time - start_time)))

    start_time = time.time()
    common_docs_tfidf_map = calculate_tf_idf_array_common_docs(query_freq,common_docs,query_docs,query_tfidf_dict_list)
    end_time = time.time()
    print("common_docs_tfidf_map Processing query({}) took {} seconds".format(query, str(end_time - start_time)))

    start_time = time.time()
    rankedDocs = calculate_cosineSimilarityVec(common_docs_tfidf_map,common_docs,query_tfidf_array)
    end_time = time.time()
    print("rankedDocs Processing query({}) took {} seconds".format(query, str(end_time - start_time)))
    #Sort in reverse order of similarity
    rankedDocs.sort(key= lambda entry: -entry[1])

    docID_to_URL_map = loadDocID_to_URL_map()
    for i in range(len(rankedDocs[:5])):
        print(docID_to_URL_map[str(rankedDocs[i][0])] + ", similarity: " + str(rankedDocs[i][1]))

def calculate_query_tf_idf_score(query_freq,query_docs_len):
    query_score = []
    for token,docs_len in zip(query_freq.keys(),query_docs_len):
        # tf idf score calculation for the query
        tf_idf = (1 + math.log(query_freq[token])) * math.log(TOTAL_DOCS/docs_len)
        query_score.append(tf_idf)
    return query_score

def calculate_tf_idf_array_common_docs(query_freq,common_docs,query_docs,query_tfidf_dict_list):
    URL_tfidf_array_map = defaultdict(list)
    for word, docs, url_tfidf_map in zip(query_freq.keys(), query_docs, query_tfidf_dict_list):
        start_time = time.time()
        for doc in common_docs:
            if doc in docs:
                URL_tfidf_array_map[doc].append(url_tfidf_map[doc])
            else:
                URL_tfidf_array_map[doc].append(0)
        end_time = time.time()
        print("tf_idf_array Processing word({}) took {} seconds".format(word, str(end_time - start_time)))
    return URL_tfidf_array_map

def calculate_cosineSimilarityVec(common_docs_tfidf_map,common_docs,query_tfidf_array):
    rankedDocs = []
    for doc in common_docs:
        cosine_val = np.dot(common_docs_tfidf_map[doc], query_tfidf_array)/(np.linalg.norm(common_docs_tfidf_map[doc]) * np.linalg.norm(query_tfidf_array))
        rankedDocs.append((doc,cosine_val))
    return rankedDocs

if __name__ == "__main__":
    while True:
        query = input()
        start_time = time.time()
        process_query(query)
        end_time = time.time()
        print("Processing query({}) took {} seconds".format(query, str(end_time - start_time)))