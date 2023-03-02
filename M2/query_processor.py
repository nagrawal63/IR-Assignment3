from InvertedIndexLoader import getIndexEntry
import time
from assignment3_m1 import tokenize_content
import numpy as np
import json, math

TOTAL_DOCS = 55393

def get_tf_idf_scores(commonDocs, query_index):
    tfidfScores = []
    for index in query_index:
        if index.docId in commonDocs:
            tfidfScores.append(index.tfidf)
    return tfidfScores

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

    query_indexes = []; common_docs = None
    query_score = []

    for token in query_freq.keys():
        # print(token)
        token_index = getIndexEntry(token)
        docs = set(map(lambda x: x.docId, token_index[token]))
        # print(token_index["a"][0].docId)
        if token_index is not None:
            query_indexes.append(token_index)
            if common_docs is None:
                common_docs = set(docs)
            else:
                common_docs = common_docs.intersection(docs)

        # tf idf score calculation for the query
        query_score.append(math.log(1 + query_freq[token]) * math.log(TOTAL_DOCS/len(token_index[token])))

    if len(common_docs) == 0:
        print("Query not found in data")
        return

    print(common_docs)
    # print(list(common_docs)[:6])
    query_score = np.array(query_score)

    tfIdfMatrix = []
    for query_index in query_indexes:
        token = list(query_index.keys())[0]
        tfidfscoresQ = get_tf_idf_scores(common_docs, query_index[token])
        tfIdfMatrix.append(tfidfscoresQ)
    
    tfIdfMatrix = np.array(tfIdfMatrix).T
    common_docs = list(common_docs)

    cosineSimilarityVec = np.dot(tfIdfMatrix, query_score)/(np.linalg.norm(tfIdfMatrix, axis=1) * np.linalg.norm(query_score))
    rankedDocs = [(common_docs[i], cosineSimilarityVec[i]) for i in range(len(common_docs))]

    # docScores = np.matmul(tfIdfMatrix, query_score.T)
    # rankedDocs = [(common_docs[i], docScores[i]) for i in range(len(common_docs))]

    #Sort in reverse order of similarity
    rankedDocs.sort(key= lambda entry: -entry[1])

    docID_to_URL_map = loadDocID_to_URL_map()
    for i in range(len(rankedDocs[:5])):
        print(docID_to_URL_map[str(rankedDocs[i][0])] + ", simimlarity: " + str(rankedDocs[i][1]))
    

if __name__ == "__main__":
    while True:
        query = input()
        start_time = time.time()
        process_query(query)
        end_time = time.time()
        print("Processing query({}) took {} seconds".format(query, str(end_time - start_time)))