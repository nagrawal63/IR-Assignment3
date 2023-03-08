from InvertedIndexLoader import getIndexEntry
import time
from assignment3_m1 import tokenize_content
import numpy as np
import json, math

TOTAL_DOCS = 55393
rankingTechnique = ["docScore", "pageRank"]

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

def load_page_rank():
    pageRankDict = {}
    with open("page_quality_features.json", 'r') as f:
        pageRankDict = json.load(f)
    return pageRankDict

def rankingResults(rankingTechniques, pageRankDict, queryIndexes, commonDocs, queryScore):
    rankedDocs = [(doc, ) for doc in commonDocs]
    if len(rankingTechniques) == 0:
        return rankedDocs
    i=0
    while i < len(rankingTechniques):
        rankingTechnique = rankingTechniques[i]
        if rankingTechnique == "tfIdfSum":
            tfIdfMatrix = [get_tf_idf_scores(commonDocs, v) for k, v in queryIndexes.items()]
            tfIdfScore = np.sum(tfIdfMatrix, axis=0)
            rankedDocs = [(rankedDocs[i] + (tfIdfScore[i], )) for i in range(len(rankedDocs))]
        elif rankingTechnique == "tfIdfCosineSimilarity":
            tfIdfMatrix = [get_tf_idf_scores(commonDocs, v) for k, v in queryIndexes.items()]
            tfIdfMatrix = np.array(tfIdfMatrix).T
            cosineSimilarityVec = np.dot(tfIdfMatrix, queryScore)/(np.linalg.norm(tfIdfMatrix, axis=1) * np.linalg.norm(queryScore))
            rankedDocs = [(rankedDocs[i] + (cosineSimilarityVec[i], )) for i in range(len(rankedDocs))]
        elif rankingTechnique == "pageRank":
            pageRankDict = load_page_rank()
            commonDocsPageRank = {docId: pageRankDict[str(docId)]["pagerank"] for docId in commonDocs}
            rankedDocs = [(rankedDocs[i] + (commonDocsPageRank[rankedDocs[i][0]], )) for i in range(len(rankedDocs))]
        elif rankingTechnique == "docScore":
            tfIdfMatrix = [get_tf_idf_scores(commonDocs, v) for k, v in queryIndexes.items()]
            tfIdfMatrix = np.array(tfIdfMatrix).T
            docScores = np.matmul(tfIdfMatrix, queryScore.T)
            rankedDocs = [(rankedDocs[i] + (docScores[i], )) for i in range(len(rankedDocs))]
        i += 1
    rankedDocs.sort(key=lambda x: (x[1:]), reverse=True)
    return rankedDocs

def process_query(query):
    tokenized_query = tokenize_content(query)
    query_freq = {}
    for token in tokenized_query:
        if token in query_freq.keys():
            query_freq[token] += 1
        else:
            query_freq[token] = 1

    query_indexes = {}
    common_docs = None
    query_score = []

    for token in query_freq.keys():
        token_index = getIndexEntry(token)
        docs = set(map(lambda x: x.docId, token_index[token]))
        if token_index is not None:
            query_indexes[token] = token_index[token]
            if common_docs is None:
                common_docs = set(docs)
            else:
                common_docs = common_docs.intersection(docs)

        # tf idf score calculation for the query
        query_score.append(1 + math.log(query_freq[token]) * math.log(TOTAL_DOCS/len(token_index[token])))

    if len(common_docs) == 0:
        print("Query not found in data")
        return

    query_score = np.array(query_score)
    pageRanksDict = load_page_rank()
    rankedDocs = rankingResults(rankingTechnique, pageRanksDict, query_indexes, common_docs, query_score)

    query_output = []
    docID_to_URL_map = loadDocID_to_URL_map()
    for i in range(len(rankedDocs[:5])):
        query_output.append(docID_to_URL_map[str(rankedDocs[i][0])] + ", simimlarity: " + str(rankedDocs[i][1]))
    return query_output
    

if __name__ == "__main__":
    while True:
        query = input()
        start_time = time.time()
        query_output = process_query(query)
        end_time = time.time()
        for output in query_output:
            print(output)
        print("Processing query({}) took {} seconds".format(query, str(end_time - start_time)))