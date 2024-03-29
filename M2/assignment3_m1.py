import time
import os
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# nltk.download('stopwords')
import json
from urllib.parse import urldefrag
import re
from InvertedIndex import InvertedIndex
from PageQualFeature import PageQualFeature
import sys
from InvertedIndex import ImportanceEnum
from collections import defaultdict
from nltk.corpus import stopwords

inverted_index = InvertedIndex()
page_qual_feature = PageQualFeature()
directory_name = "DEV"
urls_visited = set()
URL_to_docID_map = {}
docID_to_URL_map = {}
docID_wordcount_map = defaultdict()
docId_title_map = defaultdict()
anchor_text_dict = {}

BATCH_SIZE = 4000

def get_file_names():
    file_names = []
    for root, dirs, files in os.walk(directory_name):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

def process_data(file_names):
    docID = 1
    batch_size_processed = 0
    for file in file_names:
        sys.stdout.write("\rProcessing file " + str(docID))
        sys.stdout.flush()
        if file[len(file)-4:len(file)] == "json":
            with open(file) as f:
                file_dict = json.load(f)
                extracted_link = file_dict['url']
                extracted_link, _ = urldefrag(extracted_link)
                if extracted_link in urls_visited:
                    continue
                urls_visited.add(extracted_link)
                URL_to_docID_map[extracted_link] = docID
                docID_to_URL_map[docID] = extracted_link
                soup = BeautifulSoup(file_dict['content'], features="lxml")
                important_words_set, important_words_tags,title = find_important_words(soup)
                tokens = tokenize_content(soup.get_text())
                docId_title_map[docID] = title
                docID_wordcount_map[docID] = len(tokens)
                # bigrams = generate_bigram_tokens(tokenize_content_without_stopwords(soup.get_text()))
                # trigrams = generate_trigram_tokens(tokenize_content_without_stopwords(soup.get_text()))
                # tokens = bigrams
                inverted_index.addDocToInvertedIndex(docId= docID , tokens=tokens , important_words_set = important_words_set,
                                                     important_words_tags = important_words_tags)
                hls = page_qual_feature._extract_hyperlinks(soup)
                page_qual_feature._build_pagerankdb(extracted_link,hls)
                add_to_anchor_text_dict(get_anchor_words(soup))
                docID += 1
                batch_size_processed+=1
        if batch_size_processed >= BATCH_SIZE:
            inverted_index.offloadIndex()
            batch_size_processed = 0
    if batch_size_processed != 0:
        inverted_index.offloadIndex()
        batch_size_processed = 0
    print("Processed {} documents".format(docID))
    store_docID_wordcount_dict()
    store_docID_title_dict()
    anchor_text_dict = transform_anchor_text_dict()
    store_anchor_text(anchor_text_dict)
    inverted_index.mergeInvertedIndexFiles()
    inverted_index.addTfIdfScores(inverted_index.inverted_index_files[0], len(URL_to_docID_map))
    inverted_index.splitIndexIntoFiles()

def store_docID_wordcount_dict():
    with open("docID_wordcount_map.json", 'w') as f:
        json.dump(docID_wordcount_map, f)

def store_docID_title_dict():
    with open("docId_title_map.json", 'w') as f:
        json.dump(docId_title_map, f)

def find_important_words(soup):
    tags = ["title","h1", "h2", "h3", "b"]
    tags_map = {'title' : ImportanceEnum.TITLE ,'h1' :ImportanceEnum.H1 , 'h2' :ImportanceEnum.H2, 'h3' :ImportanceEnum.H3,
                'b' : ImportanceEnum.B}

    impWords_tags_map= defaultdict()
    tokens = []
    final_tokens = []
    title = ''
    text = soup.find_all(string=True)
    for sub_text in text:
        if sub_text.parent.name in tags:
            if sub_text.parent.name == "title":
                title = sub_text
            tokens = tokenize_content(sub_text)
            for token in tokens:
                if token not in impWords_tags_map:
                    impWords_tags_map[token] =  dict.fromkeys(ImportanceEnum,0)
                else:
                    impWords_tags_map[token][tags_map[sub_text.parent.name]] += 1
            final_tokens += (tokens)
    return set(final_tokens), impWords_tags_map,title

def get_anchor_words(soup):
    anchor_text_dict = {} # stores the following dict -> {token: {link:freq}}
    for tags in soup.find_all("a"):
        link, _ = urldefrag(tags.get('href'))
        tokens = " ".join(tokenize_content_without_stopwords(tags.text.strip()))
        token_hashmap = {tokens: 1}
        
        if page_qual_feature.is_valid(link):
            for token, freq in token_hashmap.items():
                if token not in anchor_text_dict.keys():
                    anchor_text_dict[token] = {link: freq}
                else:
                    if link not in anchor_text_dict[token].keys():
                        anchor_text_dict[token][link] = freq
                    else:
                        anchor_text_dict[token][link] += freq
    return anchor_text_dict

def add_to_anchor_text_dict(new_anchor_texts, anchor_text_dict = anchor_text_dict):
    for token in new_anchor_texts.keys():
        if token not in anchor_text_dict.keys():
            anchor_text_dict[token] = new_anchor_texts[token]
        else:
            for link,freq in new_anchor_texts[token].items():
                if link not in anchor_text_dict[token].keys():
                    anchor_text_dict[token][link] = freq
                else:
                    anchor_text_dict[token][link] += freq

def transform_anchor_text_dict(anchor_text_dict = anchor_text_dict, conversion_map = URL_to_docID_map):
    docId_to_anchor_map = {}
    for k, v in anchor_text_dict.items():
        docId_to_anchor_map[k] = {str(conversion_map[link.replace("http://", "https://")]):freq for link, freq in v.items() if link.replace("http://", "https://") in conversion_map.keys()}

        # Remove token from dictionary if no links for that token were in the conversion map
        if not docId_to_anchor_map[k]:
            del docId_to_anchor_map[k]
    return docId_to_anchor_map

def store_anchor_text(anchor_text_dict):
    with open("anchor_text_dict.json", 'w') as f:
        json.dump(anchor_text_dict, f)

def tokenize_content(content):
    ps = PorterStemmer()
    content = re.sub(r"[^a-zA-Z0-9\n]", " ", content)
    tokens = word_tokenize(content)
    stemmed_tokens = []
    for token in tokens:
        if len(token) < 3:
            continue
        stemmed_tokens.append(ps.stem(token.lower()))
    return stemmed_tokens

def tokenize_content_without_stopwords(content):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    content = re.sub(r"[^a-zA-Z0-9\n]", " ", content)
    tokens = word_tokenize(content)
    stemmed_tokens = []
    for token in tokens:
        if len(token) < 3 or token in stop_words or len(token) > 13 or token.isnumeric():
            continue
        stemmed_tokens.append(ps.stem(token.lower()))
    return stemmed_tokens

def generate_bigram_tokens(tokens):
    bigram_tokens = []
    bigrams_list = list(nltk.bigrams(tokens))
    for bigram_tuple in bigrams_list:
        bigram_tokens.append(bigram_tuple[0] + " " + bigram_tuple[1])
    return bigram_tokens

def generate_trigram_tokens(tokens):
    trigram_tokens = []
    trigrams_list = list(nltk.trigrams(tokens))
    for trigram_tuple in trigrams_list:
        trigram_tokens.append(trigram_tuple[0] + " " + trigram_tuple[1] +  " " + trigram_tuple[2])
    return trigram_tokens

def store_url_docID_map():
    with open("url_docID_map.json", 'w') as f:
        json.dump(URL_to_docID_map, f)

def store_docID_URL_map():
    with open("docID_url_map.json", 'w') as f:
        json.dump(docID_to_URL_map, f)

if __name__ == "__main__":
    start_time = time.time()
    file_names = get_file_names()
    print("number of files: " + str(len(file_names)))
    process_data(file_names)
    store_url_docID_map()
    store_docID_URL_map()
    end_time = time.time()
    print("Total execution time: " + str(end_time - start_time))
    print("[START]Cal pageRank")
    page_qual_feature.cal_pagerank()
    print(f"[END] Cal pageRank")
    page_qual_feature.save_pagefeat(URL_to_docID_map)