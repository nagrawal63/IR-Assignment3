from InvertedIndexLoader import getIndexEntry
import time
from assignment3_m1 import tokenize_content
import numpy as np
import json, math
from collections import defaultdict
TOTAL_DOCS = 55393

def get_tf_idf_scores(query_index, docID):
    for index in query_index:
        for web_page in query_index[index]:
            if web_page.docId == docID:
                return web_page.tfidf
    return 0

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

    query_indexes = []; common_docs = [];query_docs_len = [];query_docs = [];docs = []

    for token in sorted(query_freq.keys()):
        token_index = getIndexEntry(token)
        if token_index is not None:
            query_indexes.append(token_index)
            docs = list(map(lambda x: x.docId, token_index[token]))
            common_docs += docs
            query_docs.append(list(docs))
            query_docs_len.append(len(docs))
        else:
            del query_freq[token]

    if len(common_docs) == 0:
        print("sorry, nothing matches, please check your query")
        return None
    print("after deletion", query_freq)
    common_docs = list(set(common_docs))
    query_tfidf_array = calculate_query_tf_idf_score(query_freq,query_docs_len)

    common_docs_tfidf_map = calculate_tf_idf_array_common_docs(query_freq,common_docs,query_indexes,query_docs)

    rankedDocs = calculate_cosineSimilarityVec(common_docs_tfidf_map,common_docs,query_tfidf_array)

    #Sort in reverse order of similarity
    rankedDocs.sort(key= lambda entry: -entry[1])

    docID_to_URL_map = loadDocID_to_URL_map()
    for i in range(len(rankedDocs[:5])):
        print(docID_to_URL_map[str(rankedDocs[i][0])] + ", similarity: " + str(rankedDocs[i][1]))

def calculate_query_tf_idf_score(query_freq,query_docs_len):
    query_score = []
    for token,docs_len in zip(sorted(query_freq.keys()),query_docs_len):
        # tf idf score calculation for the query
        tf_idf = (1 + math.log(query_freq[token])) * math.log(TOTAL_DOCS/docs_len)
        query_score.append(tf_idf)
    return query_score

def calculate_tf_idf_array_common_docs(query_freq,common_docs,query_indexes,query_docs):
    URL_tfidf_array_map = defaultdict(list)
    for doc in common_docs:
        for word,query_index,docs in zip(sorted(query_freq.keys()),query_indexes,query_docs):
            if doc in docs:
                URL_tfidf_array_map[doc].append(get_tf_idf_scores(query_index,doc))
            else:
                URL_tfidf_array_map[doc].append(0)
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