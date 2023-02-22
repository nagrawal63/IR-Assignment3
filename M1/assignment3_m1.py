import time
import os
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import json
from urllib.parse import urldefrag
import re
from InvertedIndex import InvertedIndex

inverted_index = InvertedIndex()
directory_name = "DEV"
urls_visited = set()
URL_to_docID_map = {}

def get_file_names():
    file_names = []
    for root, dirs, files in os.walk(directory_name):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

def process_data(file_names):
    docID = 1
    for file in file_names:
        if file[len(file)-4:len(file)] == "json":
            with open(file) as f:
                file_dict = json.load(f)
                extracted_link = file_dict['url']
                extracted_link, _ = urldefrag(extracted_link)
                if extracted_link in urls_visited:
                    continue
                urls_visited.add(extracted_link)
                URL_to_docID_map[extracted_link] = docID
                docID += 1
                soup = BeautifulSoup(file_dict['content'], features="lxml")
                important_words = find_important_words(soup)
                tokens = tokenize_content(soup.get_text())
                inverted_index.addDocToInvertedIndex(docId= docID , tokens=tokens , important_words = important_words)


def find_important_words(soup):
    tags = ["title","h1", "h2", "h3", "b"]
    tokens = []
    final_tokens = []
    text = soup.find_all(string=True)
    for sub_text in text:
        if sub_text.parent.name in tags:
            tokens = tokenize_content(sub_text)
        final_tokens += tokens
    return set(final_tokens)


def tokenize_content(content):
    ps = PorterStemmer()
    filtered_content = re.sub(r"[^a-zA-Z0-9\n]", " ", content)
    tokens = re.findall(r'\w{3,}', filtered_content.lower())
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens

if __name__ == "__main__":
    start_time = time.time()
    file_names = get_file_names()
    print("number of files: " + str(len(file_names)))
    process_data(file_names)
    end_time = time.time()
    print("Total execution time: " + str(end_time - start_time))