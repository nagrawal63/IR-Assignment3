import json
import os
from json import JSONEncoder
from sortedcontainers import SortedDict
from enum import IntEnum
import math

class InvertedIndex:
    def __init__(self) -> None:
        self.inverted_index = SortedDict()
        self.inverted_index_files = []

    # Add a document to inverted index
    def addDocToInvertedIndex(self, docId, tokens, important_words_set, important_words_tags):
        # Calculate tokens to number of occurences hashmap
        tokens_hashmap = {}
        for token in tokens:
            if token in tokens_hashmap:
               tokens_hashmap[token] += 1
            else:
                tokens_hashmap[token] = 1
        
        # Add tokens from tokens map to inverted index
        for token in tokens_hashmap:
            if token not in self.inverted_index.keys():
                self.inverted_index[token] = list()
            
            if token in important_words_set:
                self.inverted_index[token].append(Postings(docId, tokens_hashmap[token], important_words_tags[token]))
            else:
                self.inverted_index[token].append(Postings(docId, tokens_hashmap[token], ImportanceEnum.NORMAL))

    '''
    Write the inverted index into memory
    It writes the index as json lines where each line is {<token>: [<Postings>]}
    '''
    def offloadIndex(self):
        invertedIndexFileName = str(len(self.inverted_index_files)) + ".json"
        self.inverted_index_files.append(invertedIndexFileName)

        # Write the index as json lines
        with open("index/" + invertedIndexFileName, 'w', encoding='utf-8') as f:
            for line_num in range(len(self.inverted_index.keys())):
                json_record = json.dumps({self.inverted_index.items()[line_num][0]: self.inverted_index.items()[line_num][1]}, ensure_ascii=False, cls=CustomEncoder)
                f.write(json_record + '\n')
        self.inverted_index = SortedDict()

    def mergeInvertedIndexFiles(self):
        def mergeTwoFiles(file1, file2, finalFileName):
            filePtr1 = self.loadInvertedIndex(file1)
            filePtr2 = self.loadInvertedIndex(file2)
            data1 = next(filePtr1); data2 = next(filePtr2)
            tokens1 = list(data1.keys())[0]; tokens2 = list(data2.keys())[0]
            file1NotEOD = True ; file2NotEOD= True

            f = open(finalFileName, 'w', encoding='utf-8')
            while file1NotEOD  and file2NotEOD:
                resultDict = {}
                if tokens1 == tokens2:
                    resultDict[tokens1] = list()
                    resultDict[tokens1].extend(data1[tokens1])
                    resultDict[tokens1].extend(data2[tokens2])
                    try:
                        data1 = next(filePtr1);tokens1 = list(data1.keys())[0]
                    except StopIteration:
                        file1NotEOD = False
                    try:
                        data2 = next(filePtr2);tokens2 = list(data2.keys())[0]
                    except StopIteration:
                        file2NotEOD = False

                elif tokens1 < tokens2:
                    resultDict[tokens1] = list()
                    resultDict[tokens1].extend(data1[tokens1])
                    try:
                        data1 = next(filePtr1);tokens1 = list(data1.keys())[0]
                    except StopIteration:
                        file1NotEOD = False
                else:
                    resultDict[tokens2] = list()
                    resultDict[tokens2].extend(data2[tokens2])
                    try:
                        data2 = next(filePtr2);tokens2 = list(data2.keys())[0]
                    except StopIteration:
                        file2NotEOD = False
                json_record = json.dumps(resultDict, ensure_ascii=False, cls=CustomEncoder)
                f.write(json_record + '\n')

            while file1NotEOD:
                json_record = json.dumps({tokens1: data1[tokens1]}, ensure_ascii=False, cls=CustomEncoder)
                f.write(json_record + '\n')
                try:
                    data1 = next(filePtr1);tokens1 = list(data1.keys())[0]
                except StopIteration:
                    file1NotEOD = False

            while file2NotEOD:
                json_record = json.dumps({tokens2: data2[tokens2]}, ensure_ascii=False, cls=CustomEncoder)
                f.write(json_record + '\n')
                try:
                    data2 = next(filePtr2);tokens2 = list(data2.keys())[0]
                except StopIteration:
                    file2NotEOD = False

            f.close()
        
        while(len(self.inverted_index_files) != 1):
            file1 = self.inverted_index_files.pop(0)
            file2 = self.inverted_index_files.pop(0)
            finalFileName = file1.rsplit('.', 1)[0] + "_" + file2.rsplit('.', 1)[0] + ".json"

            mergeTwoFiles("index/" + file1, "index/" + file2, "index/" + finalFileName)
            os.system("rm -rf index/" + file1)
            os.system("rm -rf index/" + file2)
            self.inverted_index_files.append(finalFileName)
    
    '''
    Loads the inverted index from file(where inverted index is stored as jsonlines)
    into a dictionary
    '''
    def loadInvertedIndex(self, filePath):

        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                data = {}
                line_data = json.loads(line.rstrip('\n|\r'))
                token = list(line_data.keys())[0]
                data[token] = [Postings.from_json(value) for value in line_data[token]]
                yield data
    
    def addTfIdfScores(self, filePath: str, total_docs):
        def calculateTfIdfScore(token_freq, token_docs, total_docs):
            return (math.log(1 + token_freq)) * (math.log(total_docs/token_docs))
        
        print("Adding TfIdf scores to inverted index")
        data = {}
        with open("index/" + filePath, 'r', encoding='utf-8') as f:
            with open("index/" + filePath.rsplit('.', 1)[0] + "_tfidf.json", 'w', encoding='utf-8') as out_f:
                for line in f:
                    line_data = json.loads(line.rstrip('\n|\r'))
                    token = list(line_data.keys())[0]
                    dict_with_tfidf = {token: list()}
                    for value in line_data[token]:
                        dict_val = Postings.from_json(value)
                        dict_val.tfidf = calculateTfIdfScore(dict_val.count, len(line_data[token]), total_docs)
                        dict_with_tfidf[token].append(dict_val)
                    json_record = json.dumps(dict_with_tfidf, ensure_ascii=False, cls=CustomEncoder)
                    out_f.write(json_record + '\n')

        return data
        
'''
Enum to capture importance characteristics of a token

Currently only using NORMAL and IMPORTANT since find_important_words function 
only gives important or not important
'''
class ImportanceEnum(IntEnum):
    TITLE = 1
    H1 = 2
    H2 = 3
    H3 = 4
    B = 5
    NORMAL = 6
    IMPORTANT = 7

class Postings(object):
    def __init__(self, docId: int, count: int, importance: ImportanceEnum, tfidf = 0.0):
        self.count = count
        self.docId = docId
        if importance is None:
            self.importance = ImportanceEnum.NORMAL
        self.importance = importance
        self.tfidf = tfidf

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    def from_json(str):
        obj = json.loads(str)
        if "tfidf" in obj.keys():
            return Postings(int(obj['count']), int(obj['docId']), ImportanceEnum(int(obj['importance'])), float(obj['tfidf']))
        else:
            return Postings(int(obj['count']), int(obj['docId']), ImportanceEnum(int(obj['importance'])))

    def to_json(self):
        return self.__str__()

class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Postings):
            return obj.to_json()
        return obj.__dict__
