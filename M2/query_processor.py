from InvertedIndexLoader import getIndexEntry
from InvertedIndexLoader import getIndexDataAllTokens
import time
from assignment3_m1 import tokenize_content,tokenize_content_without_stopwords
from assignment3_m1 import generate_trigram_tokens
from assignment3_m1 import generate_bigram_tokens
import numpy as np
import json, math
from collections import defaultdict

TOTAL_DOCS = 55393
rankingTechnique = ["tfIdfCosineSimilarity", "pageRank"]


def get_tf_idf_scores(query_index, docID):
    for index in query_index:
        for web_page in query_index[index]:
            if web_page.docId == docID:
                return web_page.tfidf_improved
    return 0


def loadDocID_to_URL_map():
    docDict = {}
    with open("docID_url_map.json", 'r') as f:
        docDict = json.load(f)
    return docDict


def load_page_rank():
    pageRankDict = {}
    with open("page_quality_features.json", 'r') as f:
        pageRankDict = json.load(f)
    return pageRankDict


# def rankingResults(rankingTechniques, pageRankDict, queryIndexes, commonDocs, queryScore):
#     rankedDocs = [(doc,) for doc in commonDocs]
#     if len(rankingTechniques) == 0:
#         return rankedDocs
#     i = 0
#     while i < len(rankingTechniques):
#         rankingTechnique = rankingTechniques[i]
#         if rankingTechnique == "tfIdfSum":
#             tfIdfMatrix = get_tf_idf_scores(commonDocs, queryIndexes)
#             tfIdfScore = {k: sum(v) for k, v in tfIdfMatrix.items()}
#             rankedDocs = [(rankedDocs[i] + (tfIdfScore[rankedDocs[i][0]],)) for i in range(len(rankedDocs))]
#         elif rankingTechnique == "tfIdfCosineSimilarity":
#             tfIdfMatrix = get_tf_idf_scores(commonDocs, queryIndexes)
#             cosineSimilarityVec = {k: np.dot(v, queryScore) / (np.linalg.norm(v) * np.linalg.norm(queryScore)) for k, v
#                                    in tfIdfMatrix.items()}
#             rankedDocs = [(rankedDocs[i] + (cosineSimilarityVec[rankedDocs[i][0]],)) for i in range(len(rankedDocs))]
#         elif rankingTechnique == "pageRank":
#             pageRankDict = load_page_rank()
#             commonDocsPageRank = {docId: pageRankDict[str(docId)]["pagerank"] for docId in commonDocs}
#             rankedDocs = [(rankedDocs[i] + (commonDocsPageRank[rankedDocs[i][0]],)) for i in range(len(rankedDocs))]
#         elif rankingTechnique == "docScore":
#             tfIdfMatrix = get_tf_idf_scores(commonDocs, queryIndexes)
#             docScores = {k: np.dot(v, queryScore) for k, v in tfIdfMatrix.items()}
#             rankedDocs = [(rankedDocs[i] + (docScores[rankedDocs[i][0]],)) for i in range(len(rankedDocs))]
#         i += 1
#     rankedDocs.sort(key=lambda x: (x[1:]), reverse=True)
#     return rankedDocs


def process_query(query, pageRanksDict):
    tokenized_query = tokenize_content(query)
    query_freq = {}
    # for token in tokenized_query:
    #     if token in query_freq.keys():
    #         query_freq[token] += 1
    #     else:
    #         query_freq[token] = 1
    # bigrams = generate_bigram_tokens(tokenize_content_without_stopwords(query))
    # for token in bigrams:
    #     if token in query_freq.keys():
    #         query_freq[token] += 10
    #     else:
    #         query_freq[token] = 10

    trigrams = generate_trigram_tokens(tokenize_content_without_stopwords(query))
    for token in trigrams:
        if token in query_freq.keys():
            query_freq[token] += 20
        else:
            query_freq[token] = 20

    # print(query_freq)

    query_indexes = []; common_docs = [];query_docs_len = [];query_docs = [];docs = []
    indexes = getIndexDataAllTokens(sorted(query_freq.keys()))
    token_index = defaultdict()
    # print(token_indexes)
    for token,index in zip(sorted(query_freq.keys()),indexes):
        token_index[token] = index
        # print(token_index)
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
    rankedDocs = calculate_cosineSimilarityVec(common_docs_tfidf_map, common_docs, query_tfidf_array)

    # rankedDocs = rankingResults(rankingTechnique, pageRanksDict, query_indexes, common_docs, np.array(query_score))
    pageRankDict = load_page_rank()
    commonDocsPageRank = {docId: pageRankDict[str(docId)]["pagerank"] for docId in [rankedDoc[0] for rankedDoc in rankedDocs]}

    rankedDocs = [(rankedDocs[i] + (commonDocsPageRank[rankedDocs[i][0]],)) for i in range(len(rankedDocs))]
    rankedDocs.sort(key=lambda x: (x[1:]), reverse=True)
    query_output = []
    docID_to_URL_map = loadDocID_to_URL_map()
    for i in range(len(rankedDocs[:5])):
        query_output.append(docID_to_URL_map[str(rankedDocs[i][0])])
    return query_output

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
        pageRanksDict = load_page_rank()
        start_time = time.time()
        query_output = process_query(query, pageRanksDict)
        end_time = time.time()
        for output in query_output:
            print(output)
        print("Processing query({}) took {} seconds".format(query, str(end_time - start_time)))