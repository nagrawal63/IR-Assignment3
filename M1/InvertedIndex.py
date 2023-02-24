import json
import os
from json import JSONEncoder
from sortedcontainers import SortedDict
from enum import IntEnum

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
        invertedIndexFileName = "initial_" + str(len(self.inverted_index_files)) + ".json"
        self.inverted_index_files.append(invertedIndexFileName)

        # Write the index as json lines
        with open("index/" + invertedIndexFileName, 'w', encoding='utf-8') as f:
            for line_num in range(len(self.inverted_index.keys())):
                json_record = json.dumps({self.inverted_index.items()[line_num][0]: self.inverted_index.items()[line_num][1]}, ensure_ascii=False, cls=CustomEncoder)
                f.write(json_record + '\n')
        self.inverted_index = SortedDict()

    def mergeInvertedIndexFiles(self):
        def mergeTwoFiles(file1, file2, finalFileName):
            data1 = self.loadInvertedIndex(file1)
            data2 = self.loadInvertedIndex(file2)
            tokens1 = list(data1.keys()); tokens2 = list(data2.keys())
            resultJSON = []
            filePtr1 = 0; filePtr2 = 0; size1 = len(data1); size2 = len(data2) 

            while filePtr1 < size1 and filePtr2 < size2:
                if tokens1[filePtr1] == tokens2[filePtr2]:
                    resultDict = {tokens1[filePtr1]: list()}
                    resultDict[tokens1[filePtr1]].extend(data1[tokens1[filePtr1]])
                    resultDict[tokens1[filePtr1]].extend(data2[tokens2[filePtr2]])
                    resultJSON.append(resultDict)
                    filePtr1 += 1; filePtr2 += 1
                elif tokens1[filePtr1] < tokens2[filePtr2]:
                    resultDict = {tokens1[filePtr1]: list()}
                    resultDict[tokens1[filePtr1]].extend(data1[tokens1[filePtr1]])
                    resultJSON.append(resultDict)
                    filePtr1 += 1
                else:
                    resultDict = {tokens2[filePtr2]: list()}
                    resultDict[tokens2[filePtr2]].extend(data2[tokens2[filePtr2]])
                    resultJSON.append(resultDict)
                    filePtr2 += 1

            while filePtr1 < size1:
                resultJSON.append({tokens1[filePtr1]: data1[tokens1[filePtr1]]})
                filePtr1 += 1

            while filePtr2 < size2:
                resultJSON.append({tokens2[filePtr2]: data2[tokens2[filePtr2]]})
                filePtr2 += 1
            
            with open(finalFileName, 'w', encoding='utf-8') as f:
                for line in resultJSON:
                    json_record = json.dumps(line, ensure_ascii=False, cls=CustomEncoder)
                    f.write(json_record + '\n')

        
        while(len(self.inverted_index_files) != 1):
            file1 = self.inverted_index_files.pop(0)
            file2 = self.inverted_index_files.pop(0)
            finalFileName = file1 + "_" + file2

            mergeTwoFiles("index/" + file1, "index/" + file2, "index/" + finalFileName)
            os.system("rm -rf index/" + file1)
            os.system("rm -rf index/" + file2)
            self.inverted_index_files.append(finalFileName)
    
    '''
    Loads the inverted index from file(where inverted index is stored as jsonlines)
    into a dictionary
    '''
    def loadInvertedIndex(self, filePath):
        print(filePath)
        data = {}
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line.rstrip('\n|\r'))
                token = list(line_data.keys())[0]
                data[token] = [Postings.from_json(value) for value in line_data[token]]
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
    def __init__(self, docId: int, count: int, importance: ImportanceEnum):
        self.count = count
        self.docId = docId
        if importance is None:
            self.importance = ImportanceEnum.NORMAL
        self.importance = importance

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    def from_json(str):
        obj = json.loads(str)
        return Postings(int(obj['count']), int(obj['docId']), ImportanceEnum(int(obj['importance'])))

    def to_json(self):
        return self.__str__()

class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Postings):
            return obj.to_json()
        return obj.__dict__
