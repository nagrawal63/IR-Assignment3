import json
import os
from json import JSONEncoder
from sortedcontainers import SortedDict

class InvertedIndex:
    def __init__(self) -> None:
        self.inverted_index = SortedDict()
        self.inverted_index_files = []

    # Add a document to inverted index
    def addDocToInvertedIndex(self, docId, tokens):
        # Calculate tokens to number of occurences hashmap
        tokens_hashmap = {}
        for token in tokens:
            if token in tokens_hashmap:
               tokens_hashmap[token] += 1
            else:
                tokens_hashmap[token] = 1
        
        # Add tokens from tokens map to inverted index
        for token in tokens_hashmap:
            if token in self.inverted_index.keys():
                self.inverted_index[token].append(Postings(docId, tokens_hashmap[token]))
            else:
                self.inverted_index[token] = [Postings(docId, tokens_hashmap[token])]

    def offloadIndex(self):
        invertedIndexFileName = "initial_" + str(len(self.inverted_index_files)) + ".json"
        self.inverted_index_files.append(invertedIndexFileName)

        data = json.dumps({"Tokens": list(self.inverted_index.keys()), "Data": self.inverted_index}, cls=PostingsEncoder)

        with open(invertedIndexFileName, 'w') as f:
            f.write(data)
        
        self.inverted_index = SortedDict()

    def mergeInvertedIndexFiles(self):
        def mergeTwoFiles(file1, file2, finalFileName):
            file1Reader = open(file1, 'r'); data1 = json.load(file1Reader)
            file2Reader = open(file2, 'r'); data2 = json.load(file2Reader)
            resultJSON = {"Data": {}}
            tokens1 = data1["Tokens"];tokens2 = data2["Tokens"]
            filePtr1 = 0; filePtr2 = 0; size1 = len(tokens1); size2 = len(tokens2) 

            while filePtr1 < size1 and filePtr2 < size2:
                if tokens1[filePtr1] == tokens2[filePtr2]:
                    if(tokens1[filePtr1] not in resultJSON["Data"].keys()):
                        resultJSON["Data"][tokens1[filePtr1]] = list()
                    resultJSON["Data"][tokens1[filePtr1]] = data1["Data"][tokens1[filePtr1]] + data2["Data"][tokens2[filePtr2]]
                    filePtr1 += 1; filePtr2 += 1
                elif tokens1[filePtr1] < tokens2[filePtr2]:
                    if(tokens1[filePtr1] not in resultJSON["Data"].keys()):
                        resultJSON["Data"][tokens1[filePtr1]] = list()
                    resultJSON["Data"][tokens1[filePtr1]] = data1["Data"][tokens1[filePtr1]]
                    filePtr1 += 1
                else:
                    if(tokens2[filePtr2] not in resultJSON["Data"].keys()):
                        resultJSON["Data"][tokens2[filePtr2]] = list()
                    resultJSON["Data"][tokens2[filePtr2]] = data2["Data"][tokens2[filePtr2]]
                    filePtr2 += 1

            while filePtr1 < size1:
                resultJSON["Data"][tokens1[filePtr1]] = data1["Data"][tokens1[filePtr1]]
                filePtr1 += 1

            while filePtr2 < size2:
                resultJSON["Data"][tokens2[filePtr2]] = data2["Data"][tokens2[filePtr2]]
                filePtr2 += 1
            
            with open(finalFileName, 'w') as resultFile:
                resultFile.write(json.dumps(resultJSON))

        
        while(len(self.inverted_index_files) != 1):
            file1 = self.inverted_index_files.pop(0)
            file2 = self.inverted_index_files.pop(0)
            finalFileName = file1 + "_" + file2

            mergeTwoFiles(file1, file2, finalFileName)
            os.system("rm -rf " + file1)
            os.system("rm -rf " + file2)
            self.inverted_index_files.append(finalFileName)

        

class Postings:
    def __init__(self, docId, count):
        self.count = count
        self.docId = docId

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_json(self):
        return self.__str__()

class PostingsEncoder(JSONEncoder):
    def default(self, obj):
        return obj.to_json()