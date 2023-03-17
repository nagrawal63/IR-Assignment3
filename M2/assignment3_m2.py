from assignment3_m1 import tokenize_content
from InvertedIndexLoader import loadInvertedIndexFromFile, getIndexDataAllTokens,getIndexDataAllTokensBigram,getIndexDataAllTokensTrigram
import time
import json, math
import numpy as np
from enum import IntEnum
from numpy import dot
from numpy.linalg import norm
# from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask_cors import CORS
from assignment3_m1 import tokenize_content,tokenize_content_without_stopwords
from assignment3_m1 import generate_trigram_tokens
from assignment3_m1 import generate_bigram_tokens

blacklist_urls = set(["http://mondego.ics.uci.edu/datasets/maven-contents.txt"])

app = Flask(__name__)
CORS(app)
with open("./url_docID_map.json") as f:
    doc2id = json.load(f)
with open("./docID_url_map.json") as f:
    id2doc = json.load(f)
with open('./page_quality_features.json') as f:
    doc2features = json.load(f)
with open('./docId_title_map.json') as f:
    id2title = json.load(f)
with open("./anchor_text_dict.json") as f:
    anchorWordFeatures = json.load(f)


def process_query(query, type):
    if type == 3:
        query_tokens = generate_trigram_tokens(tokenize_content_without_stopwords(query))
    elif type == 2:
        query_tokens = generate_bigram_tokens(tokenize_content_without_stopwords(query))
    else:
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


def merge_inverted_index(datal, doc2features, tokens, anchorWordFeatures):
    importance_map = {ImportanceEnum.NORMAL: 1, ImportanceEnum.B: 2, ImportanceEnum.H3: 3,
                      ImportanceEnum.H2: 4, ImportanceEnum.H1: 5, ImportanceEnum.TITLE: 6}
    final_page = {}
    queryv = [1] * len(datal)

    for i, td in enumerate(datal):
        # queryv[i] = 1/ len(td)
        for d in td:
            if d.docId in final_page:
                final_page[d.docId][i] = d.tfidf_improved
            else:
                final_page[d.docId] = [0] * (
                            len(datal) + 1)  # [len(td)TODO]+1 page features # initialize with zero vector
                final_page[d.docId][i] = d.tfidf_improved
    queryv.extend([0.01])

    # Get documents which are target pages of some anchor texts
    joined_query = ' '.join(tokens)
    anchorTargetPages = None
    if joined_query in anchorWordFeatures.keys():
        anchorTargetPages = anchorWordFeatures[joined_query]

    for d in final_page:
        final_page[d][-1] = doc2features[str(d)]['pagein'] / len(doc2features)
        if anchorTargetPages and d in anchorTargetPages.keys():
            final_page[d][-1] *= 100   # Increase pagein value if the page is a target page for an anchor word(form of query)

    return sorted({k: dot(queryv, v) for k, v in final_page.items() if 0 not in v}.items(), key=lambda x: x[1],
                  reverse=True)  # [TODO] & option adding ?


def retrieve_pages(tokens, doc2features,type, anchorWordFeatures):
    if type == 3:
        datal = getIndexDataAllTokensTrigram(tokens)
    elif type == 2:
        datal = getIndexDataAllTokensBigram(tokens)
    else:
        datal = getIndexDataAllTokens(tokens)
    pages = merge_inverted_index(datal, doc2features, tokens, anchorWordFeatures)
    pages = [page for page in pages if str(page[0]) not in blacklist_urls]
    return pages[:10]


@app.route('/main', methods=['GET'])
def main():
    from flask import request
    import time
    start =time.time()
    query = request.args.get('query')
    pages = []; pages_trigram=[];pages_bigram = []
    tokens = process_query(query, 3)
    pages_trigram = retrieve_pages(tokens, doc2features, 3, anchorWordFeatures)
    if len(pages_trigram)<5:
        tokens = process_query(query, 2)
        pages_bigram = (retrieve_pages(tokens, doc2features, 2, anchorWordFeatures))
    if len(pages_trigram) + len(pages_bigram) < 5:
        tokens = process_query(query, 1)
        pages = (retrieve_pages(tokens, doc2features, 1, anchorWordFeatures))
    pages.extend(pages_bigram)
    pages.extend(pages_trigram)
    pages = sorted(pages, reverse=True, key = lambda x: x[1])
    end = time.time()
    result = {i: (id2title[str(p[0])],id2doc[str(p[0])]) for i, p in enumerate(pages)}
    result["time"] = end-start
    resp = json.dumps(result)
    return resp


if __name__ == "__main__":
    while True:
        with open("./url_docID_map.json") as f:
            doc2id = json.load(f)
        with open("./docID_url_map.json") as f:
            id2doc = json.load(f)
        with open('./page_quality_features.json') as f:
            doc2features = json.load(f)
        with open("./anchor_text_dict.json") as f:
            anchorWordFeatures = json.load(f)

        print("Enter Query:")
        query = input()
        start_time = time.time()
        pages = []; pages_trigram=[];pages_bigram = []
        tokens = process_query(query, 3)
        pages_trigram = retrieve_pages(tokens, doc2features, 3, anchorWordFeatures)
        # print(pages_trigram)
        if len(pages_trigram)<5:
            tokens = process_query(query, 2)
            pages_bigram = (retrieve_pages(tokens, doc2features, 2, anchorWordFeatures))
            # print(pages_bigram)
        if len(pages_trigram) + len(pages_bigram) < 5:
            tokens = process_query(query, 1)
            pages = (retrieve_pages(tokens, doc2features, 1, anchorWordFeatures))
            # print(pages)
        for p in pages_trigram:
            # print(p,id2doc[str(p[0])])
            print(id2doc[str(p[0])])
        for p in pages_bigram:
            # print(p,id2doc[str(p[0])])
            print(id2doc[str(p[0])])
        for p in pages:
            # print(p,id2doc[str(p[0])])
            print(id2doc[str(p[0])])
        end_time = time.time()
        print("Processing query({}) took {} seconds".format(query, end_time - start_time))
